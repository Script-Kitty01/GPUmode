package twin

import (
	"fmt"
	"math"
	"sort"

	"gpumode/internal/config"
	"gpumode/internal/physics"
)

// DigitalTwinScenario models how the workload performs on a specific hardware profile and GPU count.
type DigitalTwinScenario struct {
	ScenarioName           string  `json:"scenarioName"`
	HardwareName           string  `json:"hardwareName"`
	GPUCount               int     `json:"gpuCount"`
	EstimatedLatencyMS     float64 `json:"estimatedLatencyMs"`
	EstimatedThroughputRPS float64 `json:"estimatedThroughputRps"`
	RequiredVRAMMiB        float64 `json:"requiredVramMib"`
	AvailableVRAMMiB       float64 `json:"availableVramMib"`
	VRAMFits               bool    `json:"vramFits"`
	MeetsQPSAndSLA         bool    `json:"meetsQpsAndSla"`
	JoulesPer1KInferences  float64 `json:"joulesPer1kInferences"`
	UnlockedByOptimization string  `json:"unlockedByOptimization"` // e.g. "CUDA Fusion + float4" or "VRAM Activation Reduction"
	Decision               string  `json:"decision"`               // "RECOMMENDED", "VIABLE", "VRAM_EXCEEDED", "SLA_OR_QPS_INSUFFICIENT"
	ConfidenceScorePct     float64 `json:"confidenceScorePct"`
	Provenance             string  `json:"provenance"` // "MEASURED" (local GPU) or "SIMULATED" (cross-SKU Roofline projection)
	CalibrationBasis       string  `json:"calibrationBasis"`
}

// WorkloadItem represents an inference service or batch job to be co-located by the bin-packing scheduler.
type WorkloadItem struct {
	Name              string  `json:"name"`
	Priority          string  `json:"priority"` // "HIGH", "MEDIUM", "LOW"
	ComputeUtilPct    float64 `json:"computeUtilPct"`
	BandwidthUtilPct  float64 `json:"bandwidthUtilPct"`
	VRAMMiB           float64 `json:"vramMib"`
	AvgPowerWatts     float64 `json:"avgPowerWatts"`
	LatencySLAMS      float64 `json:"latencySlaMs"`
	DeferrableToOffPk bool    `json:"deferrableToOffPk"`
}

// GPUBin represents a single physical GPU hosting one or more packed workloads.
type GPUBin struct {
	BinID                 int      `json:"binId"`
	AssignedWorkloads     []string `json:"assignedWorkloads"`
	TotalComputeUtilPct   float64  `json:"totalComputeUtilPct"`
	TotalBandwidthUtilPct float64  `json:"totalBandwidthUtilPct"`
	TotalVRAMMiB          float64  `json:"totalVramMib"`
	VRAMCapacityMiB       float64  `json:"vramCapacityMib"`
	EstimatedPowerWatts   float64  `json:"estimatedPowerWatts"`
}

// SchedulePlan summarizes the bin-packing and off-peak scheduling recommendation.
type SchedulePlan struct {
	SeparateGPUsBefore   int      `json:"separateGpusBefore"`
	ConsolidatedGPUs     int      `json:"consolidatedGpus"`
	GPUsSavedByPacking   int      `json:"gpusSavedByPacking"`
	Bins                 []GPUBin `json:"bins"`
	ObjectiveScoreJ      float64  `json:"objectiveScoreJ"` // J = αC + βE + γL
	OffPeakDeferralNotes []string `json:"offPeakDeferralNotes"`
}

// AutoscaleForecast predicts future traffic and GPU instance requirements.
type AutoscaleForecast struct {
	CurrentTrafficRPS     float64 `json:"currentTrafficRps"`
	PredictedTraffic15Min float64 `json:"predictedTraffic15Min"`
	PerGPUCapacityRPS     float64 `json:"perGpuCapacityRps"`
	CurrentGPUs           int     `json:"currentGpus"`
	RecommendedGPUs       int     `json:"recommendedGpus"`
	Action                string  `json:"action"`
}

// SimulateDigitalTwin evaluates the optimized workload across the local hardware physics and default reference tiers.
func SimulateDigitalTwin(
	localHW physics.GPUHardwareSpec,
	baseMetrics, optMetrics physics.WorkloadMetrics,
	memoryBoundFraction float64,
	econ config.EconomicsConfig,
) []DigitalTwinScenario {
	presets := physics.LoadReferenceHardwarePresets("", localHW)
	return SimulateDigitalTwinWithPresets(localHW, presets, baseMetrics, optMetrics, memoryBoundFraction, econ)
}

// SimulateDigitalTwinWithPresets evaluates the optimized workload across the local hardware physics
// and caller-provided reference tiers, separating whether eligibility was unlocked by CUDA
// compute/bandwidth speedup vs VRAM activation reduction.
func SimulateDigitalTwinWithPresets(
	localHW physics.GPUHardwareSpec,
	presets []physics.GPUHardwareSpec,
	baseMetrics, optMetrics physics.WorkloadMetrics,
	memoryBoundFraction float64,
	econ config.EconomicsConfig,
) []DigitalTwinScenario {
	if len(presets) > 0 {
		presets[0] = localHW
	} else {
		presets = []physics.GPUHardwareSpec{localHW}
	}

	baseKPI := physics.ComputeBusinessKPIs(baseMetrics, econ)
	scenarios := make([]DigitalTwinScenario, 0, len(presets)+1)

	// Add baseline reference row on current hardware
	scenarios = append(scenarios, DigitalTwinScenario{
		ScenarioName:           "Baseline (Unoptimized)",
		HardwareName:           localHW.Name,
		GPUCount:               baseKPI.GPUsRequiredForTargetQPS,
		EstimatedLatencyMS:     baseMetrics.LatencyMS,
		EstimatedThroughputRPS: baseMetrics.ThroughputRPS * float64(baseKPI.GPUsRequiredForTargetQPS),
		RequiredVRAMMiB:        baseMetrics.PeakVRAMMiB,
		AvailableVRAMMiB:       localHW.TotalVRAMMiB,
		VRAMFits:               baseMetrics.PeakVRAMMiB <= localHW.TotalVRAMMiB,
		MeetsQPSAndSLA:         baseKPI.MeetsLatencySLA,
		JoulesPer1KInferences:  baseKPI.JoulesPer1KInferences,
		UnlockedByOptimization: "None (Baseline)",
		Decision:               "BASELINE",
		ConfidenceScorePct:     98.0,
		Provenance:             "MEASURED",
		CalibrationBasis:       fmt.Sprintf("Direct CUDA event timing & nvidia-smi probe on %s (%s)", localHW.Name, localHW.ArchFlag),
	})

	mbFrac := math.Max(0.15, math.Min(0.95, memoryBoundFraction))
	cbFrac := 1.0 - mbFrac

	bestIdx := -1
	bestJoules := math.MaxFloat64

	for idx, spec := range presets {
		// Scale performance from localHW to candidate spec using Roofline bandwidth & FP32 scaling
		bwRatio := 1.0
		if localHW.PeakMemBandwidthGBps > 0 && spec.PeakMemBandwidthGBps > 0 {
			bwRatio = spec.PeakMemBandwidthGBps / localHW.PeakMemBandwidthGBps
		}
		fpRatio := 1.0
		if localHW.PeakFP32TFLOPS > 0 && spec.PeakFP32TFLOPS > 0 {
			fpRatio = spec.PeakFP32TFLOPS / localHW.PeakFP32TFLOPS
		}

		speedFactor := mbFrac*bwRatio + cbFrac*fpRatio
		if speedFactor <= 0 {
			speedFactor = 1.0
		}

		estLat := optMetrics.LatencyMS / speedFactor
		estP95 := optMetrics.P95LatencyMS / speedFactor
		if estP95 <= 0 {
			estP95 = estLat * 1.08
		}
		estRPSPerGPU := optMetrics.ThroughputRPS * speedFactor
		if estRPSPerGPU <= 0 && estLat > 0 {
			estRPSPerGPU = 1000.0 / estLat
		}

		powerScale := 1.0
		if localHW.TDPWatts > 0 && spec.TDPWatts > 0 {
			powerScale = spec.TDPWatts / localHW.TDPWatts
		}
		estWatts := optMetrics.AvgPowerWatts * powerScale
		if estWatts <= 0 {
			estWatts = spec.TDPWatts * 0.65
		}

		simMetrics := physics.WorkloadMetrics{
			LatencyMS:          estLat,
			P95LatencyMS:       estP95,
			ThroughputRPS:      estRPSPerGPU,
			AvgPowerWatts:      estWatts,
			EnergyJoulesPerReq: estWatts / math.Max(1e-6, estRPSPerGPU),
			PeakVRAMMiB:        optMetrics.PeakVRAMMiB,
		}
		kpi := physics.ComputeBusinessKPIs(simMetrics, econ)

		vramFits := optMetrics.PeakVRAMMiB <= spec.TotalVRAMMiB
		baseVRAMFit := baseMetrics.PeakVRAMMiB <= spec.TotalVRAMMiB

		unlockedBy := "CUDA Kernel Speedup (Fusion + float4)"
		if vramFits && !baseVRAMFit {
			unlockedBy = "VRAM Activation Footprint Reduction + Kernel Fusion"
		} else if kpi.GPUsRequiredForTargetQPS < baseKPI.GPUsRequiredForTargetQPS {
			unlockedBy = fmt.Sprintf("Throughput Speedup (%d → %d GPUs)", baseKPI.GPUsRequiredForTargetQPS, kpi.GPUsRequiredForTargetQPS)
		}

		decision := "VIABLE"
		if !vramFits {
			decision = "VRAM_EXCEEDED"
		} else if !kpi.MeetsLatencySLA {
			decision = "SLA_OR_QPS_INSUFFICIENT"
		}

		confidence := 96.0
		provenance := "MEASURED"
		calibBasis := fmt.Sprintf("Direct hardware measurement on %s (%.0f GB/s BW, %.1f TFLOPS FP32)", spec.Name, spec.PeakMemBandwidthGBps, spec.PeakFP32TFLOPS)
		if idx > 0 {
			provenance = "SIMULATED"
			confidence = 78.0 // Unvalidated cross-SKU Roofline projection
			calibBasis = fmt.Sprintf("Cross-SKU Roofline scaling (%.0f%% memory-BW ratio %.2fx + %.0f%% FP32 ratio %.2fx); requires target-GPU benchmark to validate",
				mbFrac*100.0, bwRatio, cbFrac*100.0, fpRatio)
		}

		sc := DigitalTwinScenario{
			ScenarioName:           fmt.Sprintf("Optimized on %d× %s", kpi.GPUsRequiredForTargetQPS, spec.Name),
			HardwareName:           spec.Name,
			GPUCount:               kpi.GPUsRequiredForTargetQPS,
			EstimatedLatencyMS:     estLat,
			EstimatedThroughputRPS: estRPSPerGPU * float64(kpi.GPUsRequiredForTargetQPS),
			RequiredVRAMMiB:        optMetrics.PeakVRAMMiB,
			AvailableVRAMMiB:       spec.TotalVRAMMiB,
			VRAMFits:               vramFits,
			MeetsQPSAndSLA:         vramFits && kpi.MeetsLatencySLA,
			JoulesPer1KInferences:  kpi.JoulesPer1KInferences,
			UnlockedByOptimization: unlockedBy,
			Decision:               decision,
			ConfidenceScorePct:     confidence,
			Provenance:             provenance,
			CalibrationBasis:       calibBasis,
		}
		scenarios = append(scenarios, sc)

		if sc.MeetsQPSAndSLA && sc.JoulesPer1KInferences < bestJoules {
			bestJoules = sc.JoulesPer1KInferences
			bestIdx = len(scenarios) - 1
		}
	}

	if bestIdx >= 0 {
		scenarios[bestIdx].Decision = "RECOMMENDED"
	}
	return scenarios
}

// ScheduleWorkloads packs compatible inference/batch workloads onto the minimum number of GPUs
// subject to VRAM, compute, and bandwidth headroom constraints, minimizing J = αC + βE + γL.
func ScheduleWorkloads(workloads []WorkloadItem, vramCapMiB float64, econ config.EconomicsConfig) SchedulePlan {
	if vramCapMiB <= 0 {
		return SchedulePlan{
			SeparateGPUsBefore: len(workloads),
			Bins:               nil,
			OffPeakDeferralNotes: []string{
				"No workload schedule calculated: live GPU VRAM capacity is unavailable.",
			},
		}
	}
	if len(workloads) == 0 {
		workloads = DefaultWorkloadMix(vramCapMiB)
	}

	// Sort by VRAM + Compute descending (First-Fit Decreasing multi-resource bin packing)
	sorted := make([]WorkloadItem, len(workloads))
	copy(sorted, workloads)
	sort.Slice(sorted, func(i, j int) bool {
		scoreI := sorted[i].ComputeUtilPct + (sorted[i].VRAMMiB/vramCapMiB)*100.0
		scoreJ := sorted[j].ComputeUtilPct + (sorted[j].VRAMMiB/vramCapMiB)*100.0
		return scoreI > scoreJ
	})

	const maxComputeCap = 88.0 // Keep 12% tail-latency headroom
	const maxBWCap = 85.0
	vramSafeCap := vramCapMiB * 0.92

	var bins []GPUBin
	var offPeakNotes []string

	for _, w := range sorted {
		if w.DeferrableToOffPk {
			offPeakNotes = append(offPeakNotes, fmt.Sprintf(
				"Defer '%s' (%s priority) to off-peak execution window",
				w.Name, w.Priority,
			))
		}

		placed := false
		for i := range bins {
			if bins[i].TotalComputeUtilPct+w.ComputeUtilPct <= maxComputeCap &&
				bins[i].TotalBandwidthUtilPct+w.BandwidthUtilPct <= maxBWCap &&
				bins[i].TotalVRAMMiB+w.VRAMMiB <= vramSafeCap {
				bins[i].AssignedWorkloads = append(bins[i].AssignedWorkloads, w.Name)
				bins[i].TotalComputeUtilPct += w.ComputeUtilPct
				bins[i].TotalBandwidthUtilPct += w.BandwidthUtilPct
				bins[i].TotalVRAMMiB += w.VRAMMiB
				bins[i].EstimatedPowerWatts += w.AvgPowerWatts * 0.72 // Shared idle board power savings
				placed = true
				break
			}
		}

		if !placed {
			bins = append(bins, GPUBin{
				BinID:                 len(bins) + 1,
				AssignedWorkloads:     []string{w.Name},
				TotalComputeUtilPct:   w.ComputeUtilPct,
				TotalBandwidthUtilPct: w.BandwidthUtilPct,
				TotalVRAMMiB:          w.VRAMMiB,
				VRAMCapacityMiB:       vramCapMiB,
				EstimatedPowerWatts:   w.AvgPowerWatts,
			})
		}
	}

	beforeGPUs := len(workloads)
	afterGPUs := len(bins)
	savedGPUs := beforeGPUs - afterGPUs
	if savedGPUs < 0 {
		savedGPUs = 0
	}

	// Compute J = αC + βE + γL normalized objective
	totalPower := 0.0
	maxUtil := 0.0
	for _, b := range bins {
		totalPower += b.EstimatedPowerWatts
		if b.TotalComputeUtilPct > maxUtil {
			maxUtil = b.TotalComputeUtilPct
		}
	}
	cNorm := float64(afterGPUs) / math.Max(1.0, float64(beforeGPUs))
	eNorm := totalPower / math.Max(1.0, float64(beforeGPUs)*75.0)
	lPenalty := math.Max(0.0, (maxUtil-70.0)/30.0)
	jScore := econ.AlphaCostWeight*cNorm + econ.BetaEnergyWeight*eNorm + econ.GammaMemoryWeight*lPenalty

	return SchedulePlan{
		SeparateGPUsBefore:   beforeGPUs,
		ConsolidatedGPUs:     afterGPUs,
		GPUsSavedByPacking:   savedGPUs,
		Bins:                 bins,
		ObjectiveScoreJ:      jScore,
		OffPeakDeferralNotes: offPeakNotes,
	}
}

// DefaultWorkloadMix returns a representative 3-service production workload mix scaled to the GPU's VRAM.
func DefaultWorkloadMix(vramCapMiB float64) []WorkloadItem {
	return []WorkloadItem{
		{
			Name:              "Primary LLM Inference (Optimized CUDA)",
			Priority:          "HIGH",
			ComputeUtilPct:    48.0,
			BandwidthUtilPct:  46.0,
			VRAMMiB:           math.Round(vramCapMiB * 0.45),
			AvgPowerWatts:     46.0,
			LatencySLAMS:      25.0,
			DeferrableToOffPk: false,
		},
		{
			Name:              "Embedding Vector Generation",
			Priority:          "MEDIUM",
			ComputeUtilPct:    21.0,
			BandwidthUtilPct:  20.0,
			VRAMMiB:           math.Round(vramCapMiB * 0.22),
			AvgPowerWatts:     22.0,
			LatencySLAMS:      50.0,
			DeferrableToOffPk: false,
		},
		{
			Name:              "Batch Document Reranker",
			Priority:          "LOW",
			ComputeUtilPct:    15.0,
			BandwidthUtilPct:  16.0,
			VRAMMiB:           math.Round(vramCapMiB * 0.16),
			AvgPowerWatts:     18.0,
			LatencySLAMS:      120.0,
			DeferrableToOffPk: true,
		},
	}
}

// ForecastAutoscaling uses Holt double-exponential smoothing on recent QPS history to predict
// capacity needs 15 minutes ahead and recommend proactive scaling.
func ForecastAutoscaling(qpsHistory []float64, perGPUCapacityRPS float64, econ config.EconomicsConfig) AutoscaleForecast {
	if perGPUCapacityRPS <= 0 {
		perGPUCapacityRPS = 100.0
	}
	if len(qpsHistory) == 0 {
		base := econ.TargetQPS
		if base <= 0 {
			base = 120.0
		}
		qpsHistory = []float64{base * 0.78, base * 0.85, base * 0.92, base * 1.0, base * 1.08}
	}

	alpha := 0.6
	beta := 0.4
	level := qpsHistory[0]
	trend := 0.0
	if len(qpsHistory) > 1 {
		trend = qpsHistory[1] - qpsHistory[0]
	}
	for i := 1; i < len(qpsHistory); i++ {
		prevLevel := level
		level = alpha*qpsHistory[i] + (1-alpha)*(level+trend)
		trend = beta*(level-prevLevel) + (1-beta)*trend
	}

	currentRPS := qpsHistory[len(qpsHistory)-1]
	predicted15m := math.Max(1.0, level+3.0*trend) // 3 intervals of 5-min ahead

	curGPUs := int(math.Max(1, math.Ceil(currentRPS/perGPUCapacityRPS)))
	recGPUs := int(math.Max(1, math.Ceil(predicted15m/perGPUCapacityRPS)))

	action := fmt.Sprintf("Hold steady at %d GPU instance(s) (headroom %.0f%%)", recGPUs, ((float64(recGPUs)*perGPUCapacityRPS-predicted15m)/(float64(recGPUs)*perGPUCapacityRPS))*100.0)
	if recGPUs > curGPUs {
		action = fmt.Sprintf("Proactively scale UP from %d to %d GPU(s) before 15-min traffic spike", curGPUs, recGPUs)
	} else if recGPUs < curGPUs {
		action = fmt.Sprintf("Scale DOWN from %d to %d GPU(s) to eliminate idle capacity waste", curGPUs, recGPUs)
	}

	return AutoscaleForecast{
		CurrentTrafficRPS:     currentRPS,
		PredictedTraffic15Min: predicted15m,
		PerGPUCapacityRPS:     perGPUCapacityRPS,
		CurrentGPUs:           curGPUs,
		RecommendedGPUs:       recGPUs,
		Action:                action,
	}
}
