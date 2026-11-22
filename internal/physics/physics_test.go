package physics

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gpumode/internal/config"
)

func TestDetectHardwareLiveAndFailure(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	nvccPath := config.ResolveNVCCPath()
	smiPath := config.ResolveNvidiaSMIPath()

	// 1. Test live hardware detection on the host machine
	spec, err := DetectHardwareForDevice(ctx, config.DefaultGPUIndex(), nvccPath, smiPath)
	if err != nil {
		if _, lookErr := exec.LookPath(nvccPath); lookErr != nil {
			t.Skipf("skipping live GPU assertions (CUDA toolchain/GPU unavailable: %v)", err)
		}
		t.Fatalf("DetectHardwareForDevice failed on CUDA host: %v", err)
	}

	if spec.Name == "" {
		t.Fatalf("expected non-empty GPU Name")
	}
	if spec.ComputeCapability == "" {
		t.Errorf("expected non-empty ComputeCapability")
	}
	if !strings.HasPrefix(spec.ArchFlag, "sm_") {
		t.Errorf("expected ArchFlag starting with sm_, got %s", spec.ArchFlag)
	}
	if spec.SMCount <= 0 {
		t.Errorf("expected positive SMCount, got %d", spec.SMCount)
	}
	if spec.PeakFP32TFLOPS <= 0 {
		t.Errorf("expected PeakFP32TFLOPS > 0, got %.2f", spec.PeakFP32TFLOPS)
	}
	if spec.PeakMemBandwidthGBps <= 0 {
		t.Errorf("expected PeakMemBandwidthGBps > 0, got %.2f", spec.PeakMemBandwidthGBps)
	}
	if spec.RidgePointFLOPsPerByte <= 0 {
		t.Errorf("expected RidgePointFLOPsPerByte > 0, got %.2f", spec.RidgePointFLOPsPerByte)
	}
	if spec.TotalVRAMMiB <= 0 {
		t.Errorf("expected TotalVRAMMiB > 0, got %.1f", spec.TotalVRAMMiB)
	}

	// 2. Invalid paths must return an error and no fabricated GPU spec.
	invalidSpec, fbErr := DetectHardware(ctx, "/nonexistent/nvcc", "/nonexistent/nvidia-smi")
	if fbErr == nil {
		t.Errorf("expected non-nil error when both nvcc and nvidia-smi paths fail")
	}
	if invalidSpec.Name != "" || invalidSpec.ArchFlag != "" || invalidSpec.SMCount != 0 ||
		invalidSpec.PeakFP32TFLOPS != 0 || invalidSpec.PeakMemBandwidthGBps != 0 {
		t.Errorf("expected no fabricated hardware spec on probe failure, got %+v", invalidSpec)
	}

	// 3. Test driver communication failure and incomplete CSV output from nvidia-smi when nvcc is missing.
	dir := t.TempDir()
	failingSMI := filepath.Join(dir, "nvidia-smi-fail")
	_ = os.WriteFile(failingSMI, []byte("#!/bin/sh\necho 'NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.' >&2\nexit 9\n"), 0o755)
	if _, err := DetectHardwareForDevice(ctx, 0, "/nonexistent/nvcc", failingSMI); err == nil {
		t.Errorf("expected error when nvidia-smi exits with driver communication failure")
	}

	malformedSMI := filepath.Join(dir, "nvidia-smi-malformed")
	_ = os.WriteFile(malformedSMI, []byte("#!/bin/sh\necho 'NVIDIA Test GPU, 8.9'\n"), 0o755)
	if _, err := DetectHardwareForDevice(ctx, 0, "/nonexistent/nvcc", malformedSMI); err == nil {
		t.Errorf("expected error when nvidia-smi returns incomplete CSV columns and nvcc is absent")
	}
}

func TestComputeBusinessKPIsAndCompareEconomics(t *testing.T) {
	econ := config.EconomicsConfig{
		TargetQPS:         150.0,
		LatencySLAMS:      20.0,
		AlphaCostWeight:   0.45,
		BetaEnergyWeight:  0.35,
		GammaMemoryWeight: 0.20,
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

	// Optimized: fused CUDA kernel (10ms latency, 100 RPS/GPU, 48W power, meets 20ms SLA, requires 2 GPUs for 150 QPS)
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
	if cmp.VRAMReductionMiB != 1100.0 {
		t.Errorf("expected VRAMReductionMiB = 1100.0, got %.2f", cmp.VRAMReductionMiB)
	}
}
