package roofline

import (
	"strings"
	"testing"

	"gpumode/internal/physics"
)

func TestDiagnoseMemoryBoundRMSNorm(t *testing.T) {
	hw := physics.GPUHardwareSpec{
		Name:                   "NVIDIA GeForce RTX 4050 Laptop GPU",
		ComputeCapability:      "8.9",
		ArchFlag:               "sm_89",
		SMCount:                20,
		PeakFP32TFLOPS:         12.0,
		PeakMemBandwidthGBps:   192.0,
		RidgePointFLOPsPerByte: 62.5,
	}
	op := EstimateOpProfile("rmsnorm_residual", 64, 4096, 0.38)
	diag := Diagnose(op, hw)

	if diag.Classification != BottleneckMemoryBound {
		t.Fatalf("expected MEMORY_BOUND classification, got %s", diag.Classification)
	}
	if diag.ArithmeticIntensity >= diag.HardwareRidgePoint {
		t.Fatalf("expected AI %.2f < ridge point %.2f", diag.ArithmeticIntensity, diag.HardwareRidgePoint)
	}
	if !strings.Contains(diag.UnderutilizationReason, "Arithmetic Intensity") {
		t.Fatalf("expected underutilization reason to mention Arithmetic Intensity, got: %s", diag.UnderutilizationReason)
	}
	if len(diag.RecommendedStrategies) < 3 {
		t.Fatalf("expected at least 3 recommended CUDA strategies, got %d", len(diag.RecommendedStrategies))
	}
	if diag.ExpectedVRAMSaveMiB <= 0 {
		t.Fatalf("expected positive VRAM savings estimate, got %.2f", diag.ExpectedVRAMSaveMiB)
	}
}

func TestDiagnoseLaunchBoundSmallOp(t *testing.T) {
	hw := physics.GPUHardwareSpec{
		Name:                   "NVIDIA GeForce RTX 4050 Laptop GPU",
		PeakFP32TFLOPS:         12.0,
		PeakMemBandwidthGBps:   192.0,
		RidgePointFLOPsPerByte: 62.5,
	}
	op := EstimateOpProfile("softmax", 2, 1024, 0.25)
	diag := Diagnose(op, hw)
	if diag.Classification != BottleneckLaunchBound {
		t.Fatalf("expected LAUNCH_BOUND for small multi-launch op, got %s", diag.Classification)
	}
	if diag.HostBoundaryProvenance != "ANALYTICAL" {
		t.Fatalf("expected default ANALYTICAL provenance, got %q", diag.HostBoundaryProvenance)
	}
}

func TestDiagnoseMeasuredEBPFHostStarvation(t *testing.T) {
	hw := physics.GPUHardwareSpec{
		Name:                   "NVIDIA GeForce RTX 4050 Laptop GPU",
		PeakFP32TFLOPS:         12.0,
		PeakMemBandwidthGBps:   192.0,
		RidgePointFLOPsPerByte: 62.5,
	}

	// Large working set (524,288 elements >= 262,144) with low BW/compute utilization:
	// without eBPF host metrics this would be MEMORY_BOUND, but high HostLaunchGapUS / HostStarvationPct
	// promotes classification to LAUNCH_BOUND with MEASURED_EBPF provenance.
	op := EstimateOpProfile("rmsnorm_residual", 128, 4096, 1.50)
	op.HostLaunchGapUS = 85.4
	op.HostSyncStallPct = 18.2
	op.HostStarvationPct = 34.6
	op.HostBoundaryProvenance = "MEASURED_EBPF"

	diag := Diagnose(op, hw)
	if diag.Classification != BottleneckLaunchBound {
		t.Fatalf("expected LAUNCH_BOUND under high eBPF host launch gap / starvation, got %s", diag.Classification)
	}
	if diag.HostBoundaryProvenance != "MEASURED_EBPF" {
		t.Fatalf("expected MEASURED_EBPF provenance, got %q", diag.HostBoundaryProvenance)
	}
	if diag.LaunchCount != op.LaunchCount || diag.HostLaunchGapUS != 85.4 || diag.HostSyncStallPct != 18.2 || diag.HostStarvationPct != 34.6 {
		t.Fatalf("expected host boundary metrics copied onto Diagnosis, got %+v", diag)
	}
	if !strings.Contains(diag.UnderutilizationReason, "85.4 µs") || !strings.Contains(diag.UnderutilizationReason, "34.6%") {
		t.Fatalf("expected UnderutilizationReason to report measured launch gap and off-CPU starvation, got: %s", diag.UnderutilizationReason)
	}

	// Memory-bound op with MEASURED_EBPF provenance preserves "Arithmetic Intensity" and appends host gap & sync stall.
	memOp := EstimateOpProfile("rmsnorm_residual", 128, 4096, 0.20)
	memOp.HostLaunchGapUS = 12.3
	memOp.HostSyncStallPct = 6.7
	memOp.HostStarvationPct = 3.1
	memOp.HostBoundaryProvenance = "MEASURED_EBPF"

	memDiag := Diagnose(memOp, hw)
	if memDiag.Classification != BottleneckMemoryBound {
		t.Fatalf("expected MEMORY_BOUND when BW util is high and host gap is small, got %s", memDiag.Classification)
	}
	if !strings.Contains(memDiag.UnderutilizationReason, "Arithmetic Intensity") {
		t.Fatalf("expected Arithmetic Intensity preserved in reason, got: %s", memDiag.UnderutilizationReason)
	}
	if !strings.Contains(memDiag.UnderutilizationReason, "12.3 µs") || !strings.Contains(memDiag.UnderutilizationReason, "6.7%") {
		t.Fatalf("expected eBPF launch gap and sync stall in memory-bound reason, got: %s", memDiag.UnderutilizationReason)
	}
}
