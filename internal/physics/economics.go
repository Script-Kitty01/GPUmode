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

// BusinessKPIs summarizes workload energy, capacity, unit cost, and SLA metrics for a GPU run.
type BusinessKPIs struct {
	JoulesPer1KInferences    float64 `json:"joulesPer1kInferences"`
	RequestsPerGPUHour       float64 `json:"requestsPerGpuHour"`
	GPUsRequiredForTargetQPS int     `json:"gpusRequiredForTargetQps"`
	MeetsLatencySLA          bool    `json:"meetsLatencySla"`
	CostPer1KRequestsUSD     float64 `json:"costPer1kRequestsUsd"`
	SpotCostPer1KRequestsUSD float64 `json:"spotCostPer1kRequestsUsd"`
	EnergyCostPer1MReqsUSD   float64 `json:"energyCostPer1mReqsUsd"`
	HourlyFleetCostUSD       float64 `json:"hourlyFleetCostUsd"`
}

// SavingsComparison quantifies the hardware, energy, and financial improvement between a baseline and optimized workload.
type SavingsComparison struct {
	Baseline            BusinessKPIs `json:"baseline"`
	Optimized           BusinessKPIs `json:"optimized"`
	SpeedupFactor       float64      `json:"speedupFactor"`
	ThroughputGainPct   float64      `json:"throughputGainPct"`
	EnergySavedPct      float64      `json:"energySavedPct"`
	VRAMReductionMiB    float64      `json:"vramReductionMib"`
	GPUsEliminated      int          `json:"gpusEliminated"`
	UnitCostSavedPct    float64      `json:"unitCostSavedPct"`
	AnnualFleetSavedUSD float64      `json:"annualFleetSavedUsd"`
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

	gpuHourlyUSD := econ.GPUHourlyCostUSD
	if gpuHourlyUSD <= 0 {
		gpuHourlyUSD = 0.75
	}
	spotDisc := econ.SpotDiscountPct
	if spotDisc <= 0 || spotDisc >= 95 {
		spotDisc = 65.0
	}
	kwhUSD := econ.ElectricityKWhUSD
	if kwhUSD <= 0 {
		kwhUSD = 0.14
	}

	// 1 kWh = 3,600,000 Joules
	kwhPer1K := joulesPer1K / 3600000.0
	energyCostPer1K := kwhPer1K * kwhUSD
	energyCostPer1M := energyCostPer1K * 1000.0

	var costPer1K, spotCostPer1K float64
	if requestsPerGPUHour > 0 {
		computeCostPer1K := (gpuHourlyUSD / requestsPerGPUHour) * 1000.0
		costPer1K = computeCostPer1K + energyCostPer1K
		spotCostPer1K = computeCostPer1K*(1.0-spotDisc/100.0) + energyCostPer1K
	}
	hourlyFleetUSD := float64(gpusRequired) * gpuHourlyUSD

	return BusinessKPIs{
		JoulesPer1KInferences:    roundTo2(joulesPer1K),
		RequestsPerGPUHour:       roundTo2(requestsPerGPUHour),
		GPUsRequiredForTargetQPS: gpusRequired,
		MeetsLatencySLA:          meetsSLA,
		CostPer1KRequestsUSD:     math.Round(costPer1K*1e6) / 1e6,
		SpotCostPer1KRequestsUSD: math.Round(spotCostPer1K*1e6) / 1e6,
		EnergyCostPer1MReqsUSD:   roundTo2(energyCostPer1M),
		HourlyFleetCostUSD:       roundTo2(hourlyFleetUSD),
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

	unitCostSavedPct := 0.0
	if baseKPI.CostPer1KRequestsUSD > 0 {
		unitCostSavedPct = ((baseKPI.CostPer1KRequestsUSD - optKPI.CostPer1KRequestsUSD) / baseKPI.CostPer1KRequestsUSD) * 100.0
	}
	annualFleetSavedUSD := (baseKPI.HourlyFleetCostUSD - optKPI.HourlyFleetCostUSD) * 8760.0
	if annualFleetSavedUSD < 0 {
		annualFleetSavedUSD = 0
	}

	return SavingsComparison{
		Baseline:            baseKPI,
		Optimized:           optKPI,
		SpeedupFactor:       roundTo2(speedup),
		ThroughputGainPct:   roundTo2(throughputGainPct),
		EnergySavedPct:      roundTo2(energySavedPct),
		VRAMReductionMiB:    roundTo2(vramReduction),
		GPUsEliminated:      gpusEliminated,
		UnitCostSavedPct:    roundTo2(unitCostSavedPct),
		AnnualFleetSavedUSD: roundTo2(annualFleetSavedUSD),
	}
}
