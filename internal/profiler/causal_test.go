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

func TestFullCausalChainPipelineOnCUDA(t *testing.T) {
	ctx := context.Background()
	econ := config.DefaultEconomicsConfig()
	nvccPath := config.ResolveNVCCPath()
	smiPath := config.ResolveNvidiaSMIPath()
	gpuIdx := config.DefaultGPUIndex()
	hw, err := physics.DetectHardwareForDevice(ctx, gpuIdx, nvccPath, smiPath)
	if err != nil {
		t.Skipf("live CUDA GPU unavailable (%v); skipping causal pipeline test", err)
	}

	runner := harness.NewRunner(nvccPath, config.ResolveSanitizerPath(), hw.ArchFlag, 3, 25*time.Second, nil)
	runner.GPUIndex = gpuIdx
	runner.RunSanitizer = false

	pipe := NewPipeline(hw, nil, runner, config.ResolvePythonPath(), "../../scripts/profile_pytorch.py")
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
