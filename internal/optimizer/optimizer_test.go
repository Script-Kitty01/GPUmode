package optimizer

import (
	"context"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/harness"
)

func TestBayesianAutotunerOnLiveRTX4050(t *testing.T) {
	econ := config.DefaultEconomicsConfig()
	runner := harness.NewRunner("/opt/cuda/bin/nvcc", "/opt/cuda/bin/compute-sanitizer", "sm_89", 3, 25*time.Second, nil)
	runner.RunSanitizer = false // Speed up unit test; sanitizer gate tested separately

	tuner := NewAutotuner(NewDeterministicProposer(), runner)
	summary := tuner.Run(context.Background(), "rmsnorm_residual", 128, 4096, 40, StrategyBayesian, econ)

	if !summary.BaselineResult.CompiledOK || !summary.BaselineResult.ParityPassed {
		t.Fatalf("expected baseline kernel to compile and pass parity, got: %+v", summary.BaselineResult)
	}
	if !summary.BestResult.CompiledOK || !summary.BestResult.ParityPassed || !summary.BestResult.EdgeCasePassed {
		t.Fatalf("expected best kernel to compile and pass parity + edge case, got: %+v", summary.BestResult)
	}
	if summary.BestResult.MedianLatencyMS >= summary.BaselineResult.MedianLatencyMS {
		t.Fatalf("expected optimized fused float4 kernel (%.4f ms) to beat unfused baseline (%.4f ms)",
			summary.BestResult.MedianLatencyMS, summary.BaselineResult.MedianLatencyMS)
	}
	if len(summary.ParetoRecommendations.Frontier) == 0 {
		t.Fatalf("expected non-empty Pareto frontier")
	}
	t.Logf("RTX 4050 Live Speedup: Baseline %.3f ms -> Optimized %.3f ms (%.2fx faster, %d/%d trials evaluated)",
		summary.BaselineResult.MedianLatencyMS,
		summary.BestResult.MedianLatencyMS,
		summary.BaselineResult.MedianLatencyMS/summary.BestResult.MedianLatencyMS,
		summary.BayesianEvaluations,
		summary.GridTotalConfigurations,
	)
}
