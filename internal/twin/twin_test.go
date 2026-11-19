package twin

import (
	"testing"

	"gpumode/internal/config"
	"gpumode/internal/physics"
)

func TestSimulateDigitalTwinAndBinPacking(t *testing.T) {
	econ := config.DefaultEconomicsConfig()
	econ.TargetQPS = 120
	econ.LatencySLAMS = 25.0

	hw := physics.GPUHardwareSpec{
		Name:                 "NVIDIA GeForce RTX 4050 Laptop GPU",
		ComputeCapability:    "8.9",
		ArchFlag:             "sm_89",
		SMCount:              20,
		TotalVRAMMiB:         6141,
		PeakFP32TFLOPS:       12.0,
		PeakMemBandwidthGBps: 192.0,
		TDPWatts:             75.0,
	}

	base := physics.WorkloadMetrics{
		LatencyMS:          16.0,
		P95LatencyMS:       18.5,
		ThroughputRPS:      62.5,
		AvgPowerWatts:      62.0,
		EnergyJoulesPerReq: 0.992,
		PeakVRAMMiB:        2800,
	}
	opt := physics.WorkloadMetrics{
		LatencyMS:          7.2,
		P95LatencyMS:       8.1,
		ThroughputRPS:      138.8,
		AvgPowerWatts:      51.0,
		EnergyJoulesPerReq: 0.367,
		PeakVRAMMiB:        1650,
	}

	scenarios := SimulateDigitalTwin(hw, base, opt, 0.85, econ)
	if len(scenarios) < 2 {
		t.Fatalf("expected multiple Digital Twin scenarios, got %d", len(scenarios))
	}
	if scenarios[1].GPUCount >= scenarios[0].GPUCount {
		t.Fatalf("expected optimized scenario to need fewer GPUs than baseline (%d vs %d)", scenarios[1].GPUCount, scenarios[0].GPUCount)
	}

	plan := ScheduleWorkloads(nil, hw.TotalVRAMMiB, econ)
	if plan.ConsolidatedGPUs >= plan.SeparateGPUsBefore {
		t.Fatalf("expected bin packing to consolidate 3 workloads onto fewer GPUs, got %d -> %d", plan.SeparateGPUsBefore, plan.ConsolidatedGPUs)
	}

	forecast := ForecastAutoscaling([]float64{90, 105, 120, 140}, 85.0, econ)
	if forecast.RecommendedGPUs < 2 {
		t.Fatalf("expected autoscaler to recommend >= 2 GPUs for rising traffic, got %d", forecast.RecommendedGPUs)
	}
}
