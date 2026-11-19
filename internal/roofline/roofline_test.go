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
}
