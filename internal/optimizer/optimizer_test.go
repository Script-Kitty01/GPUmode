package optimizer

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/harness"
	"gpumode/internal/physics"
)

func TestBayesianAutotunerOnLiveCUDA(t *testing.T) {
	ctx := context.Background()
	nvccPath := config.ResolveNVCCPath()
	smiPath := config.ResolveNvidiaSMIPath()
	hw, err := physics.DetectHardwareForDevice(ctx, config.DefaultGPUIndex(), nvccPath, smiPath)
	if err != nil {
		t.Skipf("live CUDA GPU unavailable (%v); skipping live autotuner test", err)
	}

	econ := config.DefaultEconomicsConfig()
	runner := harness.NewRunner(nvccPath, config.ResolveSanitizerPath(), hw.ArchFlag, 3, 25*time.Second, nil)
	runner.GPUIndex = hw.GPUIndex
	runner.RunSanitizer = false // Speed up unit test; sanitizer gate tested separately

	tuner := NewAutotuner(NewDeterministicProposerForHardware(hw), runner)
	summary := tuner.Run(ctx, "rmsnorm_residual", 128, 4096, 40, StrategyBayesian, econ)

	if !summary.BaselineResult.CompiledOK || !summary.BaselineResult.ParityPassed {
		t.Fatalf("expected baseline kernel to compile and pass parity, got: %+v", summary.BaselineResult)
	}
	if !summary.BestResult.CompiledOK || !summary.BestResult.ParityPassed || !summary.BestResult.EdgeCasePassed {
		t.Fatalf("expected best kernel to compile and pass parity + edge case, got: %+v", summary.BestResult)
	}
	if summary.BestResult.MedianLatencyMS >= summary.BaselineResult.MedianLatencyMS {
		t.Fatalf("expected optimized fused kernel (%.4f ms) to beat unfused baseline (%.4f ms)",
			summary.BestResult.MedianLatencyMS, summary.BaselineResult.MedianLatencyMS)
	}
	if len(summary.ParetoRecommendations.Frontier) == 0 {
		t.Fatalf("expected non-empty Pareto frontier")
	}
	t.Logf("%s (%s) Live Speedup: Baseline %.3f ms -> Optimized %.3f ms (%.2fx faster, %d/%d trials evaluated)",
		hw.Name, hw.ArchFlag,
		summary.BaselineResult.MedianLatencyMS,
		summary.BestResult.MedianLatencyMS,
		summary.BaselineResult.MedianLatencyMS/summary.BestResult.MedianLatencyMS,
		summary.BayesianEvaluations,
		summary.GridTotalConfigurations,
	)
}

func TestAllSupportedCUDAKernelOperations(t *testing.T) {
	ctx := context.Background()
	nvccPath := config.ResolveNVCCPath()
	smiPath := config.ResolveNvidiaSMIPath()
	hw, err := physics.DetectHardwareForDevice(ctx, config.DefaultGPUIndex(), nvccPath, smiPath)
	if err != nil {
		t.Skipf("live CUDA GPU unavailable (%v); skipping op suite", err)
	}

	runner := harness.NewRunner(nvccPath, config.ResolveSanitizerPath(), hw.ArchFlag, 3, 25*time.Second, nil)
	runner.GPUIndex = hw.GPUIndex
	runner.RunSanitizer = false
	proposer := NewDeterministicProposerForHardware(hw)

	for _, op := range []string{"layernorm", "softmax", "swiglu", "rope_qkv", "qgemv_decode", "gqa_attention"} {
		t.Run(op, func(t *testing.T) {
			base, cands := proposer.ProposeCandidates(op, 64, 1024, 15)
			res := runner.EvaluateBatch(ctx, []harness.KernelSpec{base, cands[len(cands)-1]})
			if len(res) != 2 {
				t.Fatalf("expected 2 results, got %d", len(res))
			}
			for _, r := range res {
				if !r.CompiledOK || !r.ParityPassed || !r.EdgeCasePassed {
					t.Fatalf("%s kernel %s failed verification: compiled=%v parity=%v edge=%v err=%s",
						op, r.Spec.ID, r.CompiledOK, r.ParityPassed, r.EdgeCasePassed, r.ErrorMessage)
				}
				if r.TheoreticalOccupancyPct <= 0 || r.ActiveWarpsPerSM <= 0 {
					t.Errorf("%s kernel %s missing occupancy stats: occ=%.1f%% activeWarps=%d",
						op, r.Spec.ID, r.TheoreticalOccupancyPct, r.ActiveWarpsPerSM)
				}
			}
		})
	}
}

func TestSelfHealingRepairLoopOnBrokenKernel(t *testing.T) {
	ctx := context.Background()
	nvccPath := config.ResolveNVCCPath()
	smiPath := config.ResolveNvidiaSMIPath()
	hw, err := physics.DetectHardwareForDevice(ctx, config.DefaultGPUIndex(), nvccPath, smiPath)
	if err != nil {
		t.Skipf("live CUDA GPU unavailable (%v); skipping repair test", err)
	}

	runner := harness.NewRunner(nvccPath, config.ResolveSanitizerPath(), hw.ArchFlag, 2, 25*time.Second, nil)
	runner.GPUIndex = hw.GPUIndex
	runner.RunSanitizer = false

	brokenSpec := harness.KernelSpec{
		ID:         "broken_rmsnorm",
		Name:       "Broken Synthesized Kernel",
		OpCategory: "rmsnorm_residual",
		BlockSize:  256,
		VecWidth:   4,
		Fused:      true,
		BatchSize:  64,
		HiddenDim:  1024,
		Iterations: 10,
		CUDASource: "__global__ void bad_kernel() { syntax_error_here }",
	}
	failedList := runner.EvaluateBatch(ctx, []harness.KernelSpec{brokenSpec})
	if len(failedList) != 1 || failedList[0].CompiledOK {
		t.Fatalf("expected broken kernel to fail initial compilation")
	}

	repaired, attempts := RepairCandidateWithFeedback(ctx, runner, failedList[0], NewDeterministicProposerForHardware(hw), 2)
	if len(attempts) == 0 {
		t.Fatalf("expected at least 1 repair attempt")
	}
	if !repaired.CompiledOK || !repaired.ParityPassed || !repaired.EdgeCasePassed {
		t.Fatalf("expected self-healed kernel to compile and pass parity, got: %+v", repaired)
	}
}

// TestCompileAllCatalogKernelsWithNVCC verifies that every baseline and optimized CUDA C++ kernel
// across all 8 operations in SupportedOperations() compiles cleanly with nvcc -O3 -Xptxas -v and
// produces zero local-memory register spills. This test requires only nvcc (no physical GPU device),
// allowing full CUDA compilation and PTXAS verification inside standard CI containers.
func TestCompileAllCatalogKernelsWithNVCC(t *testing.T) {
	nvccPath := config.ResolveNVCCPath()
	if _, err := exec.LookPath(nvccPath); err != nil {
		t.Skipf("nvcc not found (%v); skipping CUDA compilation check", err)
	}

	proposer := NewDeterministicProposer()
	tmpDir := t.TempDir()

	for _, op := range SupportedOperations() {
		t.Run(op, func(t *testing.T) {
			base, cands := proposer.ProposeCandidates(op, 64, 1024, 10)
			toCheck := []harness.KernelSpec{base}
			if len(cands) > 0 {
				toCheck = append(toCheck, cands[0], cands[len(cands)-1])
			}
			for _, spec := range toCheck {
				cuFile := filepath.Join(tmpDir, spec.ID+".cu")
				binFile := filepath.Join(tmpDir, spec.ID+".bin")
				if err := os.WriteFile(cuFile, []byte(spec.CUDASource), 0o600); err != nil {
					t.Fatalf("write %s: %v", cuFile, err)
				}
				cmd := exec.Command(nvccPath, "-O3", "-arch=sm_89", "-Xptxas", "-v", cuFile, "-o", binFile)
				out, err := cmd.CombinedOutput()
				logStr := strings.TrimSpace(string(out))
				if err != nil {
					t.Fatalf("nvcc compilation failed for %s (%s): %v\n%s", op, spec.ID, err, logStr)
				}
				ptxas := harness.ParsePTXASOutput(logStr)
				if ptxas.RegistersPerThread <= 0 {
					t.Errorf("expected >0 registers parsed from ptxas for %s, got %d", spec.ID, ptxas.RegistersPerThread)
				}
				if ptxas.SpillStoresBytes > 0 || ptxas.SpillLoadsBytes > 0 {
					t.Errorf("unexpected register spills in %s: stores=%dB loads=%dB",
						spec.ID, ptxas.SpillStoresBytes, ptxas.SpillLoadsBytes)
				}
			}
		})
	}
}
