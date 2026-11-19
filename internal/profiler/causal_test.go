package profiler

import (
	"context"
	"strings"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/harness"
	"gpumode/internal/optimizer"
	"gpumode/internal/physics"
)

func TestFullCausalChainPipelineOnRTX4050(t *testing.T) {
	ctx := context.Background()
	econ := config.DefaultEconomicsConfig()
	hw, err := physics.DetectHardware(ctx, "/opt/cuda/bin/nvcc", "nvidia-smi")
	if err != nil {
		t.Logf("DetectHardware fallback notice: %v", err)
	}

	runner := harness.NewRunner("/opt/cuda/bin/nvcc", "/opt/cuda/bin/compute-sanitizer", hw.ArchFlag, 3, 25*time.Second, nil)
	runner.RunSanitizer = false

	pipe := NewPipeline(hw, nil, runner, "python3", "../../scripts/profile_pytorch.py")
	rep := pipe.Execute(ctx, "rmsnorm_residual", 128, 4096, 30, optimizer.StrategyBayesian, econ)

	if rep.EconomicsComparison.SpeedupFactor <= 1.2 {
		t.Fatalf("expected >1.2x speedup in causal report, got %.2fx", rep.EconomicsComparison.SpeedupFactor)
	}
	if !strings.Contains(rep.SevenAnswers.Q4IsItNumericallySafe, "VERIFIED SAFE") {
		t.Fatalf("expected Q4 numerical safety verification, got: %s", rep.SevenAnswers.Q4IsItNumericallySafe)
	}
	if rep.EconomicsComparison.Optimized.CostPer1MInferencesUSD >= rep.EconomicsComparison.Baseline.CostPer1MInferencesUSD {
		t.Fatalf("expected lower $/1M inferences after optimization")
	}

	// Test dynamic rate recomputation
	econ.GPUHourlyRateUSD = 3.50
	econ.TargetQPS = 500
	updated := RecomputeEconomics(rep, econ)
	if updated.EconomicsComparison.MonthlySavingsUSD <= 0 {
		t.Fatalf("expected positive monthly dollar savings at 500 QPS, got %.2f", updated.EconomicsComparison.MonthlySavingsUSD)
	}
}
