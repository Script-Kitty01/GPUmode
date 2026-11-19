package physics

import (
	"math"

	"gpumode/internal/config"
)

// WorkloadMetrics captures measured or simulated GPU execution metrics for a kernel or model operator.
type WorkloadMetrics struct {
	LatencyMS          float64 `json:"latencyMs"`          // p50 or effective per-request GPU execution time in ms
	P95LatencyMS       float64 `json:"p95LatencyMs"`       // p95 tail latency in ms
	ThroughputRPS      float64 `json:"throughputRps"`      // requests/sec per GPU
	AvgPowerWatts      float64 `json:"avgPowerWatts"`      // measured GPU board power in Watts
	EnergyJoulesPerReq float64 `json:"energyJoulesPerReq"` // measured GPU Joules per request
	PeakVRAMMiB        float64 `json:"peakVramMib"`        // peak GPU memory footprint in MiB
}

// BusinessKPIs summarizes workload energy, capacity, and SLA metrics for a GPU run.
type BusinessKPIs struct {
	JoulesPer1KInferences    float64 `json:"joulesPer1kInferences"`
	RequestsPerGPUHour       float64 `json:"requestsPerGpuHour"`
	GPUsRequiredForTargetQPS int     `json:"gpusRequiredForTargetQps"`
	MeetsLatencySLA          bool    `json:"meetsLatencySla"`
}

// SavingsComparison quantifies the hardware and energy improvement between a baseline and optimized workload.
type SavingsComparison struct {
	Baseline          BusinessKPIs `json:"baseline"`
	Optimized         BusinessKPIs `json:"optimized"`
	SpeedupFactor     float64      `json:"speedupFactor"`
	ThroughputGainPct float64      `json:"throughputGainPct"`
	EnergySavedPct    float64      `json:"energySavedPct"`
	VRAMReductionMiB  float64      `json:"vramReductionMib"`
	GPUsEliminated    int          `json:"gpusEliminated"`
}

// ComputeBusinessKPIs summarizes measured GPU latency, energy, and capacity against target QPS and SLA.
func ComputeBusinessKPIs(m WorkloadMetrics, econ config.EconomicsConfig) BusinessKPIs {
	rps := m.ThroughputRPS
	if rps <= 0 && m.LatencyMS > 0 {
		rps = 1000.0 / m.LatencyMS
	}

	execTimeSec := 0.0
	if m.LatencyMS > 0 {
		execTimeSec = m.LatencyMS / 1000.0
	} else if rps > 0 {
		execTimeSec = 1.0 / rps
	}

	joulesPerReq := m.EnergyJoulesPerReq
	if joulesPerReq <= 0 && m.AvgPowerWatts > 0 && execTimeSec > 0 {
		joulesPerReq = m.AvgPowerWatts * execTimeSec
	}

	requestsPerGPUHour := rps * 3600.0
	joulesPer1K := joulesPerReq * 1000.0

	effectiveP95 := m.P95LatencyMS
	if effectiveP95 <= 0 {
		effectiveP95 = m.LatencyMS
	}
	meetsSLA := true
	if econ.LatencySLAMS > 0 && effectiveP95 > econ.LatencySLAMS {
		meetsSLA = false
	}

	gpusRequired := 1
	if econ.TargetQPS > 0 && rps > 0 {
		gpusRequired = int(math.Ceil(econ.TargetQPS / rps))
		if gpusRequired < 1 {
			gpusRequired = 1
		}
	}

	return BusinessKPIs{
		JoulesPer1KInferences:    roundTo2(joulesPer1K),
		RequestsPerGPUHour:       roundTo2(requestsPerGPUHour),
		GPUsRequiredForTargetQPS: gpusRequired,
		MeetsLatencySLA:          meetsSLA,
	}
}

// CompareWorkloadEconomics compares baseline and optimized hardware/energy metrics.
func CompareWorkloadEconomics(baseline, optimized WorkloadMetrics, econ config.EconomicsConfig) SavingsComparison {
	baseKPI := ComputeBusinessKPIs(baseline, econ)
	optKPI := ComputeBusinessKPIs(optimized, econ)

	speedup := 1.0
	if optimized.LatencyMS > 0 && baseline.LatencyMS > 0 {
		speedup = baseline.LatencyMS / optimized.LatencyMS
	} else if baseKPI.RequestsPerGPUHour > 0 && optKPI.RequestsPerGPUHour > 0 {
		speedup = optKPI.RequestsPerGPUHour / baseKPI.RequestsPerGPUHour
	}

	throughputGainPct := 0.0
	if baseKPI.RequestsPerGPUHour > 0 {
		throughputGainPct = ((optKPI.RequestsPerGPUHour - baseKPI.RequestsPerGPUHour) / baseKPI.RequestsPerGPUHour) * 100.0
	}

	energySavedPct := 0.0
	if baseKPI.JoulesPer1KInferences > 0 {
		energySavedPct = ((baseKPI.JoulesPer1KInferences - optKPI.JoulesPer1KInferences) / baseKPI.JoulesPer1KInferences) * 100.0
	}

	vramReduction := baseline.PeakVRAMMiB - optimized.PeakVRAMMiB
	gpusEliminated := baseKPI.GPUsRequiredForTargetQPS - optKPI.GPUsRequiredForTargetQPS

	return SavingsComparison{
		Baseline:          baseKPI,
		Optimized:         optKPI,
		SpeedupFactor:     roundTo2(speedup),
		ThroughputGainPct: roundTo2(throughputGainPct),
		EnergySavedPct:    roundTo2(energySavedPct),
		VRAMReductionMiB:  roundTo2(vramReduction),
		GPUsEliminated:    gpusEliminated,
	}
}
