package optimizer

import (
	"context"
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

	for _, op := range []string{"layernorm", "softmax", "swiglu"} {
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
			}
		})
	}
}
