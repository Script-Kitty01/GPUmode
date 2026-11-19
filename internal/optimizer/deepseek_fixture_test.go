package optimizer

import (
	"context"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/harness"
	"gpumode/internal/telemetry"
)

// OfflineDeepSeekFixtureProposer simulates an offline DeepSeek-V4/Qwen3.5-9B model proposing
// candidate CUDA C++ kernels through the pluggable KernelProposer interface, verifying that
// the hardware verification engine gates and benchmarks candidate kernels identically.
type OfflineDeepSeekFixtureProposer struct {
	fallback *DeterministicProposer
}

func NewOfflineDeepSeekFixtureProposer() *OfflineDeepSeekFixtureProposer {
	return &OfflineDeepSeekFixtureProposer{
		fallback: NewDeterministicProposer(),
	}
}

func (p *OfflineDeepSeekFixtureProposer) Name() string {
	return "Offline DeepSeek-9B Fixture KernelProposer (Verification Gate Test)"
}

func (p *OfflineDeepSeekFixtureProposer) ProposeCandidates(opCategory string, batchSize, hiddenDim, iterations int) (harness.KernelSpec, []harness.KernelSpec) {
	base, cands := p.fallback.ProposeCandidates(opCategory, batchSize, hiddenDim, iterations)
	for i := range cands {
		cands[i].Name = "[DeepSeek-9B Fixture] " + cands[i].Name
	}
	return base, cands
}

func TestOfflineDeepSeekFixtureWithLiveEnergyAndSanitizerOnRTX4050(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	econ := config.DefaultEconomicsConfig()
	mon := telemetry.NewMonitor("nvidia-smi", 150*time.Millisecond, 100)
	mon.Start(ctx)

	runner := harness.NewRunner("/opt/cuda/bin/nvcc", "/opt/cuda/bin/compute-sanitizer", "sm_89", 3, 25*time.Second, mon)
	runner.RunSanitizer = true // Full compute-sanitizer memcheck gate enabled

	proposer := NewOfflineDeepSeekFixtureProposer()
	tuner := NewAutotuner(proposer, runner)

	// DeepSeek-9B hidden dimension (4096) workload test
	summary := tuner.Run(ctx, "swiglu", 128, 4096, 30, StrategyBayesian, econ)

	if !summary.BestResult.CompiledOK {
		t.Fatalf("expected DeepSeek fixture kernel to compile: %s", summary.BestResult.ErrorMessage)
	}
	if !summary.BestResult.SanitizerPassed {
		t.Fatalf("expected DeepSeek fixture kernel to pass compute-sanitizer memcheck: %s", summary.BestResult.SanitizerLog)
	}
	if !summary.BestResult.ParityPassed || !summary.BestResult.EdgeCasePassed {
		t.Fatalf("expected DeepSeek fixture kernel to pass numerical parity and 1027 edge case")
	}
	if summary.BestResult.Energy.AvgPowerWatts <= 0 || summary.BestResult.Energy.JoulesPer1KRequests <= 0 {
		t.Fatalf("expected live GPU power & Joules/1K integration > 0, got %+v", summary.BestResult.Energy)
	}

	t.Logf("Offline DeepSeek-9B Fixture Verified on RTX 4050: %s | Latency %.3f ms -> %.3f ms | Power %.2f W | Energy %.2f J/1K req | Sanitizer: CLEAN",
		summary.BestResult.Spec.Name,
		summary.BaselineResult.MedianLatencyMS,
		summary.BestResult.MedianLatencyMS,
		summary.BestResult.Energy.AvgPowerWatts,
		summary.BestResult.Energy.JoulesPer1KRequests,
	)
}
