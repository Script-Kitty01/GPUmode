package physics

import (
	"context"
	"testing"
	"time"

	"gpumode/internal/config"
)

func TestDetectHardwareLiveAndFallback(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// 1. Test live hardware detection on the host machine
	spec, err := DetectHardware(ctx, "/opt/cuda/bin/nvcc", "nvidia-smi")
	if err != nil {
		t.Logf("DetectHardware returned fallback error (expected if GPU unavailable): %v", err)
	}

	if spec.Name == "" {
		t.Fatalf("expected non-empty GPU Name")
	}
	if spec.ComputeCapability != "8.9" {
		t.Errorf("expected ComputeCapability 8.9 on RTX 4050 host, got %s", spec.ComputeCapability)
	}
	if spec.ArchFlag != "sm_89" {
		t.Errorf("expected ArchFlag sm_89, got %s", spec.ArchFlag)
	}
	if spec.SMCount <= 0 {
		t.Errorf("expected positive SMCount, got %d", spec.SMCount)
	}
	if spec.PeakFP32TFLOPS <= 5.0 {
		t.Errorf("expected PeakFP32TFLOPS > 5.0, got %.2f", spec.PeakFP32TFLOPS)
	}
	if spec.PeakMemBandwidthGBps <= 50.0 {
		t.Errorf("expected PeakMemBandwidthGBps > 50.0, got %.2f", spec.PeakMemBandwidthGBps)
	}
	if spec.RidgePointFLOPsPerByte <= 10.0 {
		t.Errorf("expected RidgePointFLOPsPerByte > 10.0, got %.2f", spec.RidgePointFLOPsPerByte)
	}
	if spec.TotalVRAMMiB < 4000 {
		t.Errorf("expected TotalVRAMMiB >= 4000, got %.1f", spec.TotalVRAMMiB)
	}

	// 2. Test graceful fallback when invalid binary paths are provided
	fbSpec, fbErr := DetectHardware(ctx, "/nonexistent/nvcc", "/nonexistent/nvidia-smi")
	if fbErr == nil {
		t.Errorf("expected non-nil error when both nvcc and nvidia-smi paths fail")
	}
	if fbSpec.ArchFlag != "sm_89" || fbSpec.SMCount != 20 || fbSpec.PeakMemBandwidthGBps != 192.0 {
		t.Errorf("unexpected fallback spec: %+v", fbSpec)
	}
}

func TestReferenceHardwarePresets(t *testing.T) {
	presets := ReferenceHardwarePresets()
	if len(presets) != 4 {
		t.Fatalf("expected 4 reference hardware presets, got %d", len(presets))
	}
	for i, p := range presets {
		if p.Name == "" || p.PeakFP32TFLOPS <= 0 || p.PeakMemBandwidthGBps <= 0 || p.RidgePointFLOPsPerByte <= 0 {
			t.Errorf("preset[%d] has invalid physics fields: %+v", i, p)
		}
	}
}

func TestComputeBusinessKPIsAndCompareEconomics(t *testing.T) {
	econ := config.EconomicsConfig{
		GPUHourlyRateUSD:    1.50,
		ElectricityRateKWh:  0.14,
		DatacenterPUE:       1.30,
		ServerOverheadWatts: 35.0,
		CarbonIntensityGCO2: 385.0,
		TargetQPS:           150.0,
		LatencySLAMS:        20.0,
		MonthlyOperatingHrs: 730.0,
		AlphaCostWeight:     0.45,
		BetaEnergyWeight:    0.35,
		GammaMemoryWeight:   0.20,
	}

	// Baseline: slower kernel (25ms latency, 40 RPS/GPU, 62W power, violates 20ms SLA, requires 4 GPUs for 150 QPS)
	baseline := WorkloadMetrics{
		LatencyMS:          25.0,
		P95LatencyMS:       28.0,
		ThroughputRPS:      40.0,
		AvgPowerWatts:      62.0,
		EnergyJoulesPerReq: 62.0 * 0.025, // 1.55 J
		PeakVRAMMiB:        3200.0,
	}

	// Optimized: fused sm_89 kernel (10ms latency, 100 RPS/GPU, 48W power, meets 20ms SLA, requires 2 GPUs for 150 QPS)
	optimized := WorkloadMetrics{
		LatencyMS:          10.0,
		P95LatencyMS:       12.5,
		ThroughputRPS:      100.0,
		AvgPowerWatts:      48.0,
		EnergyJoulesPerReq: 48.0 * 0.010, // 0.48 J
		PeakVRAMMiB:        2100.0,
	}

	cmp := CompareWorkloadEconomics(baseline, optimized, econ)

	if cmp.Baseline.MeetsLatencySLA {
		t.Errorf("expected baseline to violate 20ms latency SLA (p95=28ms)")
	}
	if !cmp.Optimized.MeetsLatencySLA {
		t.Errorf("expected optimized to meet 20ms latency SLA (p95=12.5ms)")
	}

	if cmp.Optimized.CostPer1MInferencesUSD >= cmp.Baseline.CostPer1MInferencesUSD {
		t.Errorf("expected optimized $/1M inferences (%.4f) < baseline (%.4f)",
			cmp.Optimized.CostPer1MInferencesUSD, cmp.Baseline.CostPer1MInferencesUSD)
	}
	if cmp.Optimized.JoulesPer1KInferences >= cmp.Baseline.JoulesPer1KInferences {
		t.Errorf("expected optimized Joules/1K (%.2f) < baseline (%.2f)",
			cmp.Optimized.JoulesPer1KInferences, cmp.Baseline.JoulesPer1KInferences)
	}
	if cmp.Optimized.RequestsPerGPUHour <= cmp.Baseline.RequestsPerGPUHour {
		t.Errorf("expected optimized Requests/GPU-hr (%.0f) > baseline (%.0f)",
			cmp.Optimized.RequestsPerGPUHour, cmp.Baseline.RequestsPerGPUHour)
	}

	if cmp.Baseline.GPUsRequiredForTargetQPS != 4 {
		t.Errorf("expected baseline GPUsRequiredForTargetQPS = 4, got %d", cmp.Baseline.GPUsRequiredForTargetQPS)
	}
	if cmp.Optimized.GPUsRequiredForTargetQPS != 2 {
		t.Errorf("expected optimized GPUsRequiredForTargetQPS = 2, got %d", cmp.Optimized.GPUsRequiredForTargetQPS)
	}
	if cmp.GPUsEliminated != 2 {
		t.Errorf("expected 2 GPUs eliminated, got %d", cmp.GPUsEliminated)
	}

	if cmp.SpeedupFactor != 2.5 {
		t.Errorf("expected SpeedupFactor = 2.50, got %.2f", cmp.SpeedupFactor)
	}
	if cmp.MonthlySavingsUSD <= 2000.0 {
		t.Errorf("expected MonthlySavingsUSD > $2000, got %.2f", cmp.MonthlySavingsUSD)
	}
	if cmp.AnnualSavingsUSD <= 24000.0 {
		t.Errorf("expected AnnualSavingsUSD > $24000, got %.2f", cmp.AnnualSavingsUSD)
	}
	if cmp.MonthlyKWhSaved <= 100.0 {
		t.Errorf("expected MonthlyKWhSaved > 100 kWh, got %.2f", cmp.MonthlyKWhSaved)
	}
	if cmp.MonthlyCarbonSavedKgCO2e <= 40.0 {
		t.Errorf("expected MonthlyCarbonSavedKgCO2e > 40 kg, got %.2f", cmp.MonthlyCarbonSavedKgCO2e)
	}
	if cmp.VRAMReductionMiB != 1100.0 {
		t.Errorf("expected VRAMReductionMiB = 1100.0, got %.2f", cmp.VRAMReductionMiB)
	}
}
