package scheduler

import (
	"testing"

	"gpumode/internal/config"
	"gpumode/internal/physics"
)

func TestBuildFleetScheduleReportVRAMAndSpotRouting(t *testing.T) {
	hw := physics.GPUHardwareSpec{
		Name:                       "NVIDIA GeForce RTX 4050 Laptop GPU",
		ComputeCapability:          "8.9",
		ArchFlag:                   "sm_89",
		SMCount:                    20,
		TotalVRAMMiB:               6141,
		PeakMemBandwidthGBps:       192.0,
		EmpiricalDRAMBandwidthGBps: 151.15,
		TDPWatts:                   65.0,
	}
	econ := config.DefaultEconomicsConfig()

	// 1. 4B Model (2.82 GiB) fits cleanly in local 6 GiB VRAM
	qwen4B := ModelDeploymentProfile{
		ModelName:            "Qwen3.5-4B-Q4_K_M",
		ModelSizeMiB:         2888,
		MeasuredPeakVRAMMiB:  4242,
		MeasuredTPOTMS:       21.06,
		MeasuredTokensPerSec: 47.48,
		MeasuredPowerWatts:   53.45,
		JoulesPer1KTokens:    1125.7,
	}
	rep4B := BuildFleetScheduleReport(hw, qwen4B, econ)

	if len(rep4B.PlacementCatalog) < 6 {
		t.Fatalf("expected >=6 placement options in catalog, got %d", len(rep4B.PlacementCatalog))
	}
	if !rep4B.RecommendedOnDemandPool.FitsFullVRAM {
		t.Errorf("expected recommended On-Demand pool for 4B model to fit full VRAM")
	}
	if rep4B.RecommendedSpotPool.CostPer1MTokensUSD >= rep4B.RecommendedOnDemandPool.CostPer1MTokensUSD {
		t.Errorf("expected Spot cost/1M tokens (%.4f) < On-Demand cost/1M tokens (%.4f)",
			rep4B.RecommendedSpotPool.CostPer1MTokensUSD, rep4B.RecommendedOnDemandPool.CostPer1MTokensUSD)
	}
	if len(rep4B.AdmissionPolicies) != 3 {
		t.Fatalf("expected 3 priority admission policies, got %d", len(rep4B.AdmissionPolicies))
	}
	if rep4B.AdmissionPolicies[2].Action != "ROUTE_TO_SPOT_POOL" {
		t.Errorf("expected BATCH_SPOT_INTERRUPTIBLE priority to route to Spot pool, got %s", rep4B.AdmissionPolicies[2].Action)
	}
	if len(rep4B.StrategyComparison) != 3 {
		t.Fatalf("expected 3 scheduling strategy simulation results, got %d", len(rep4B.StrategyComparison))
	}
	weighted := rep4B.StrategyComparison[2]
	naive := rep4B.StrategyComparison[0]
	if weighted.CostPer1MTokensUSD >= naive.CostPer1MTokensUSD || weighted.SLOCompliancePct <= naive.SLOCompliancePct {
		t.Errorf("expected COST_SLO_WEIGHTED_PARETO to beat ROUND_ROBIN on cost and SLO compliance: weighted=%+v naive=%+v", weighted, naive)
	}

	// 2. 9B Model (5.38 GiB) exceeds 88% of local 6 GiB VRAM with KV cache -> recommends 24 GiB L4/A10G pool for full offload
	deepseek9B := ModelDeploymentProfile{
		ModelName:            "DeepSeek-V4-Pro-9B-Q4_K_M",
		ModelSizeMiB:         5510,
		MeasuredPeakVRAMMiB:  5920,
		MeasuredTPOTMS:       50.64,
		MeasuredTokensPerSec: 19.75,
		MeasuredPowerWatts:   42.45,
		JoulesPer1KTokens:    2149.4,
	}
	rep9B := BuildFleetScheduleReport(hw, deepseek9B, econ)
	for _, opt := range rep9B.PlacementCatalog {
		if opt.PoolID == "pool_local_gpu" && opt.FitsFullVRAM {
			t.Errorf("expected 9B model (5.5 GiB + 768 MiB KV) to be flagged as exceeding 6 GiB local VRAM headroom")
		}
	}
	if !rep9B.RecommendedOnDemandPool.FitsFullVRAM {
		t.Errorf("expected scheduler to recommend a 24+ GiB pool that fits full 9B model VRAM, got %+v", rep9B.RecommendedOnDemandPool)
	}
}
