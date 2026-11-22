package profiler

import (
	"context"
	"strings"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/ebpftrace"
	"gpumode/internal/harness"
	"gpumode/internal/optimizer"
	"gpumode/internal/pareto"
	"gpumode/internal/physics"
	"gpumode/internal/roofline"
	"gpumode/internal/telemetry"
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

	// Test dynamic SLA/QPS recomputation
	econ.TargetQPS = 500
	updated := RecomputeEconomics(rep, econ)
	if updated.EconomicsComparison.SpeedupFactor <= 1.2 {
		t.Fatalf("expected speedup > 1.2x after recompute, got %.2f", updated.EconomicsComparison.SpeedupFactor)
	}
}

func TestRecomputeEconomicsAndSevenAnswersHostCUDATrace(t *testing.T) {
	hw := physics.GPUHardwareSpec{
		Name:                   "NVIDIA GeForce RTX 4050 Laptop GPU",
		ComputeCapability:      "8.9",
		ArchFlag:               "sm_89",
		SMCount:                20,
		TotalVRAMMiB:           6144,
		PeakFP32TFLOPS:         12.0,
		PeakMemBandwidthGBps:   192.0,
		RidgePointFLOPsPerByte: 62.5,
		MaxSMClockMHz:          2370,
		TDPWatts:               115,
	}

	pipe := NewPipeline(hw, nil, nil, "python3", "scripts/profile_pytorch.py")
	if pipe.EBPFMode != ebpftrace.ModeAuto {
		t.Fatalf("expected NewPipeline EBPFMode=%q, got %q", ebpftrace.ModeAuto, pipe.EBPFMode)
	}

	baseRes := harness.BenchmarkResult{
		Spec:            harness.KernelSpec{ID: "base", Name: "Baseline Unfused", OpCategory: "rmsnorm_residual", BlockSize: 256, VecWidth: 1},
		CompiledOK:      true,
		ParityPassed:    true,
		EdgeCasePassed:  true,
		MedianLatencyMS: 0.40,
		P95LatencyMS:    0.45,
		ThroughputRPS:   2500.0,
		PeakVRAMMiB:     128.0,
		Energy: telemetry.EnergyMeasurement{
			AvgPowerWatts:         65.0,
			IdlePowerWatts:        15.0,
			ActivePowerWatts:      50.0,
			JoulesPerRequest:      0.026,
			JoulesPer1KRequests:   26.0,
			EnergyDelayProduct:    0.0000104,
			RepeatWindows:         3,
			EnergyUncertaintyPct:  4.5,
			MeasurementConfidence: "HIGH",
			Provenance:            "MEASURED",
		},
	}
	bestRes := harness.BenchmarkResult{
		Spec:            harness.KernelSpec{ID: "opt", Name: "Fused Warp-Shuffle float4", OpCategory: "rmsnorm_residual", BlockSize: 256, VecWidth: 4, Fused: true},
		CompiledOK:      true,
		ParityPassed:    true,
		EdgeCasePassed:  true,
		MaxAbsError:     1.2e-6,
		MedianLatencyMS: 0.18,
		P95LatencyMS:    0.20,
		ThroughputRPS:   5555.5,
		PeakVRAMMiB:     72.0,
		PTXAS:           harness.PTXASStats{RegistersPerThread: 28, SharedMemBytes: 128},
		Energy: telemetry.EnergyMeasurement{
			AvgPowerWatts:         58.0,
			IdlePowerWatts:        15.0,
			ActivePowerWatts:      43.0,
			JoulesPerRequest:      0.01044,
			JoulesPer1KRequests:   10.44,
			EnergyDelayProduct:    0.00000188,
			RepeatWindows:         3,
			EnergyUncertaintyPct:  3.8,
			MeasurementConfidence: "HIGH",
			Provenance:            "MEASURED",
		},
	}

	baseProfile := roofline.EstimateOpProfile("rmsnorm_residual", 256, 4096, baseRes.MedianLatencyMS)
	baseDiag := roofline.Diagnose(baseProfile, hw)
	optProfile := roofline.EstimateOpProfile("rmsnorm_residual", 256, 4096, bestRes.MedianLatencyMS)
	optProfile.LaunchCount = 1
	optDiag := roofline.Diagnose(optProfile, hw)

	trace := PyTorchTrace{
		TorchAvailable: false,
		Provenance:     "ESTIMATED",
		Hotspots: []HotspotOp{
			{Name: "RMSNorm + Residual Add (Unfused)", Category: "rmsnorm_residual", SharePct: 34.5, Launches: 4, Provenance: "ESTIMATED"},
		},
	}

	measuredEBPF := ebpftrace.Summary{
		Enabled:        true,
		Active:         true,
		Mode:           ebpftrace.ModeAuto,
		Provenance:     ebpftrace.ProvenanceMeasuredEBPF,
		KernelLaunches: 420,
		AvgLaunchGapUS: 18.5,
		MemcpyH2DMiB:   12.50,
		MemcpyD2HMiB:   0.75,
		SyncStallMS:    3.40,
		OffCPUMS:       1.25,
	}

	rep := CausalChainReport{
		OpCategory:        "rmsnorm_residual",
		BatchSize:         256,
		HiddenDim:         4096,
		Hardware:          hw,
		TelemetryBackend:  "NVML",
		HostCUDATrace:     measuredEBPF,
		Trace:             trace,
		BaselineRoofline:  baseDiag,
		OptimizedRoofline: optDiag,
		Autotune: optimizer.AutotuneSummary{
			BaselineResult: baseRes,
			BestResult:     bestRes,
			ParetoCandidates: []pareto.Candidate{
				{
					ID:                  bestRes.Spec.ID,
					Name:                bestRes.Spec.Name,
					LatencyMS:           bestRes.MedianLatencyMS,
					P95LatencyMS:        bestRes.P95LatencyMS,
					ThroughputRPS:       bestRes.ThroughputRPS,
					PowerWatts:          bestRes.Energy.AvgPowerWatts,
					JoulesPer1KRequests: bestRes.Energy.JoulesPer1KRequests,
					PeakVRAMMiB:         bestRes.PeakVRAMMiB,
				},
			},
		},
	}

	econ := config.DefaultEconomicsConfig()
	econ.TargetQPS = 1000
	updated := RecomputeEconomics(rep, econ)

	if updated.TelemetryBackend != "NVML" {
		t.Fatalf("expected TelemetryBackend=NVML preserved, got %q", updated.TelemetryBackend)
	}
	if updated.HostCUDATrace.Provenance != ebpftrace.ProvenanceMeasuredEBPF || updated.HostCUDATrace.KernelLaunches != 420 {
		t.Fatalf("expected MEASURED_EBPF HostCUDATrace preserved, got %+v", updated.HostCUDATrace)
	}
	q2 := updated.SevenAnswers.Q2WhichOpsResponsible
	for _, want := range []string{"[MEASURED_EBPF]", "420", "18.5 µs", "12.50 / 0.75 MiB", "3.40 ms", "1.25 ms"} {
		if !strings.Contains(q2, want) {
			t.Errorf("expected Q2WhichOpsResponsible to contain %q, got: %s", want, q2)
		}
	}

	// Verify graceful unprivileged summary is preserved without fabricating MEASURED_EBPF narrative.
	repUnpriv := rep
	repUnpriv.HostCUDATrace = ebpftrace.Summary{
		Enabled:      true,
		Active:       false,
		Mode:         ebpftrace.ModeAuto,
		Provenance:   ebpftrace.ProvenanceUnavailableUnprivileged,
		StatusDetail: "eBPF requires CAP_BPF and CAP_PERFMON",
	}
	updatedUnpriv := RecomputeEconomics(repUnpriv, econ)
	if updatedUnpriv.HostCUDATrace.Provenance != ebpftrace.ProvenanceUnavailableUnprivileged {
		t.Fatalf("expected UNAVAILABLE_UNPRIVILEGED preserved, got %q", updatedUnpriv.HostCUDATrace.Provenance)
	}
	if strings.Contains(updatedUnpriv.SevenAnswers.Q2WhichOpsResponsible, "[MEASURED_EBPF]") {
		t.Fatalf("did not expect [MEASURED_EBPF] in Q2 when eBPF trace is inactive: %s", updatedUnpriv.SevenAnswers.Q2WhichOpsResponsible)
	}
}
