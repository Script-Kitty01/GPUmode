package optimizer

import (
	"context"
	"os"
	"os/exec"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/harness"
	"gpumode/internal/physics"
	"gpumode/internal/telemetry"
)

// OfflineDeepSeekFixtureProposer simulates an offline LLM model proposing
// candidate CUDA C++ kernels through the pluggable KernelProposer interface, verifying that
// the hardware verification engine gates and benchmarks candidate kernels identically.
type OfflineDeepSeekFixtureProposer struct {
	fallback *DeterministicProposer
}
func NewOfflineDeepSeekFixtureProposer(hw physics.GPUHardwareSpec) *OfflineDeepSeekFixtureProposer {
	return &OfflineDeepSeekFixtureProposer{
		fallback: NewDeterministicProposerForHardware(hw),
	}
}

func (p *OfflineDeepSeekFixtureProposer) Name() string {
	return "Offline LLM Fixture KernelProposer (Verification Gate Test)"
}

func (p *OfflineDeepSeekFixtureProposer) ProposeCandidates(opCategory string, batchSize, hiddenDim, iterations int) (harness.KernelSpec, []harness.KernelSpec) {
	base, cands := p.fallback.ProposeCandidates(opCategory, batchSize, hiddenDim, iterations)
	for i := range cands {
		cands[i].Name = "[LLM Fixture] " + cands[i].Name
	}
	return base, cands
}

func TestOfflineDeepSeekFixtureWithLiveEnergyAndSanitizerOnCUDA(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	nvccPath := config.ResolveNVCCPath()
	smiPath := config.ResolveNvidiaSMIPath()
	gpuIdx := config.DefaultGPUIndex()
	hw, err := physics.DetectHardwareForDevice(ctx, gpuIdx, nvccPath, smiPath)
	if err != nil {
		t.Skipf("live CUDA GPU unavailable (%v); skipping sanitizer+energy test", err)
	}

	econ := config.DefaultEconomicsConfig()
	mon := telemetry.NewMonitorForGPU(smiPath, gpuIdx, 150*time.Millisecond, 100)
	mon.Start(ctx)

	runner := harness.NewRunner(nvccPath, config.ResolveSanitizerPath(), hw.ArchFlag, 3, 25*time.Second, mon)
	runner.GPUIndex = gpuIdx
	runner.RunSanitizer = true // Full compute-sanitizer memcheck gate enabled

	proposer := NewOfflineDeepSeekFixtureProposer(hw)
	tuner := NewAutotuner(proposer, runner)

	summary := tuner.Run(ctx, "swiglu", 128, 4096, 30, StrategyBayesian, econ)

	if !summary.BestResult.CompiledOK {
		t.Fatalf("expected fixture kernel to compile: %s", summary.BestResult.ErrorMessage)
	}
	if !summary.BestResult.SanitizerPassed {
		t.Fatalf("expected fixture kernel to pass compute-sanitizer memcheck: %s", summary.BestResult.SanitizerLog)
	}
	if !summary.BestResult.ParityPassed || !summary.BestResult.EdgeCasePassed {
		t.Fatalf("expected fixture kernel to pass numerical parity and 1027 edge case")
	}
	if summary.BestResult.Energy.AvgPowerWatts <= 0 || summary.BestResult.Energy.JoulesPer1KRequests <= 0 {
		t.Fatalf("expected live GPU power & Joules/1K integration > 0, got %+v", summary.BestResult.Energy)
	}

	t.Logf("Offline LLM Fixture Verified on %s (%s): %s | Latency %.3f ms -> %.3f ms | Power %.2f W | Energy %.2f J/1K req | Sanitizer: CLEAN",
		hw.Name, hw.ArchFlag,
		summary.BestResult.Spec.Name,
		summary.BaselineResult.MedianLatencyMS,
		summary.BestResult.MedianLatencyMS,
		summary.BestResult.Energy.AvgPowerWatts,
		summary.BestResult.Energy.JoulesPer1KRequests,
	)
}

func TestLocalDeepSeekGGUFModelLiveProposalAndVerification(t *testing.T) {
	modelPath := ResolveDefaultModelPath()
	if modelPath == "" {
		t.Skip("GPUMODE_MODEL_PATH not set; skipping live GGUF model proposal test")
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("GGUF model not present at %s: %v", modelPath, err)
	}
	if _, err := exec.LookPath("llama-cli"); err != nil {
		t.Skip("llama-cli not found in PATH; skipping live GGUF model proposal test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	nvccPath := config.ResolveNVCCPath()
	smiPath := config.ResolveNvidiaSMIPath()
	gpuIdx := config.DefaultGPUIndex()
	hw, err := physics.DetectHardwareForDevice(ctx, gpuIdx, nvccPath, smiPath)
	if err != nil {
		t.Skipf("live CUDA GPU unavailable (%v); skipping live GGUF test", err)
	}

	proposer := NewLocalDeepSeekProposerForHardware("llama-cli", modelPath, hw)
	econ := config.DefaultEconomicsConfig()
	mon := telemetry.NewMonitorForGPU(smiPath, gpuIdx, 150*time.Millisecond, 100)
	mon.Start(ctx)

	runner := harness.NewRunner(nvccPath, config.ResolveSanitizerPath(), hw.ArchFlag, 3, 25*time.Second, mon)
	runner.GPUIndex = gpuIdx
	runner.RunSanitizer = true

	tuner := NewAutotuner(proposer, runner)
	summary := tuner.Run(ctx, "rmsnorm_residual", 256, 4096, 30, StrategyBayesian, econ)

	if !summary.BestResult.CompiledOK {
		t.Fatalf("expected GGUF proposed kernel to compile: %s", summary.BestResult.ErrorMessage)
	}
	if !summary.BestResult.SanitizerPassed {
		t.Fatalf("expected GGUF proposed kernel to pass compute-sanitizer: %s", summary.BestResult.SanitizerLog)
	}
	if !summary.BestResult.ParityPassed || !summary.BestResult.EdgeCasePassed {
		t.Fatalf("expected GGUF proposed kernel to pass parity and 1027 edge case")
	}
	if proposer.LastProposal != nil {
		t.Logf("Live Local GGUF Proposal: Block=%d, Vec=float%d, Fused=%v | Reason=%q %s",
			proposer.LastProposal.BlockSize,
			proposer.LastProposal.VecWidth,
			proposer.LastProposal.Fused,
			proposer.LastProposal.Reason,
			proposer.LastPerfInfo,
		)
	}
	t.Logf("Live GGUF Verified Winner on %s: %s | Latency %.3f ms -> %.3f ms | Energy %.2f J/1K req",
		hw.Name,
		summary.BestResult.Spec.Name,
		summary.BaselineResult.MedianLatencyMS,
		summary.BestResult.MedianLatencyMS,
		summary.BestResult.Energy.JoulesPer1KRequests,
	)
}
