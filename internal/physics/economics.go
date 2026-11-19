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

// BusinessKPIs translates low-level GPU execution metrics into business, energy, and fleet capacity KPIs.
type BusinessKPIs struct {
	CostPer1MInferencesUSD  float64 `json:"costPer1mInferencesUsd"`
	HardwareCostPer1MUSD    float64 `json:"hardwareCostPer1mUsd"`
	ElectricityCostPer1MUSD float64 `json:"electricityCostPer1mUsd"`
	JoulesPer1KInferences   float64 `json:"joulesPer1kInferences"`
	TotalServerJoulesPer1K  float64 `json:"totalServerJoulesPer1k"`
	RequestsPerGPUHour      float64 `json:"requestsPerGpuHour"`
	GPUsRequiredForTargetQPS int    `json:"gpusRequiredForTargetQps"`
	MonthlyHardwareCostUSD  float64 `json:"monthlyHardwareCostUsd"`
	MonthlyElectricityCostUSD float64 `json:"monthlyElectricityCostUsd"`
	MonthlyTotalCostUSD     float64 `json:"monthlyTotalCostUsd"`
	MonthlyEnergyKWh        float64 `json:"monthlyEnergyKwh"`
	MonthlyCarbonKgCO2e     float64 `json:"monthlyCarbonKgCo2e"`
	MeetsLatencySLA         bool    `json:"meetsLatencySla"`
}

// SavingsComparison quantifies the end-to-end physics and financial improvement between a baseline and optimized workload.
type SavingsComparison struct {
	Baseline                BusinessKPIs `json:"baseline"`
	Optimized               BusinessKPIs `json:"optimized"`
	SpeedupFactor           float64      `json:"speedupFactor"`
	ThroughputGainPct       float64      `json:"throughputGainPct"`
	EnergySavedPct          float64      `json:"energySavedPct"`
	VRAMReductionMiB        float64      `json:"vramReductionMib"`
	GPUsEliminated          int          `json:"gpusEliminated"`
	MonthlySavingsUSD       float64      `json:"monthlySavingsUsd"`
	AnnualSavingsUSD        float64      `json:"annualSavingsUsd"`
	MonthlyKWhSaved         float64      `json:"monthlyKwhSaved"`
	MonthlyCarbonSavedKgCO2e float64     `json:"monthlyCarbonSavedKgCo2e"`
	CostPer1MReductionPct   float64      `json:"costPer1mReductionPct"`
}

// ComputeBusinessKPIs translates CUDA kernel/workload telemetry into unit economics and fleet costs
// using the caller's configurable tariff, PUE, carbon intensity, and SLA parameters.
func ComputeBusinessKPIs(m WorkloadMetrics, econ config.EconomicsConfig) BusinessKPIs {
	pue := econ.DatacenterPUE
	if pue <= 0 {
		pue = 1.0
	}
	monthlyHours := econ.MonthlyOperatingHrs
	if monthlyHours <= 0 {
		monthlyHours = 730.0
	}

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

	avgPowerW := m.AvgPowerWatts
	if avgPowerW <= 0 && joulesPerReq > 0 && execTimeSec > 0 {
		avgPowerW = joulesPerReq / execTimeSec
	}

	requestsPerGPUHour := rps * 3600.0

	// Joules per 1K inferences (GPU board vs PUE-adjusted total server power)
	joulesPer1K := joulesPerReq * 1000.0
	serverOverheadJPerReq := econ.ServerOverheadWatts * execTimeSec
	totalServerJoulesPer1K := (joulesPerReq + serverOverheadJPerReq) * pue * 1000.0

	// Unit costs per 1M inferences
	var hwCostPer1M, elecCostPer1M float64
	if requestsPerGPUHour > 0 {
		gpuHoursPer1M := 1e6 / requestsPerGPUHour
		hwCostPer1M = gpuHoursPer1M * econ.GPUHourlyRateUSD
	}
	// 1 kWh = 3.6e6 Joules; totalServerJoulesPer1M = totalServerJoulesPer1K * 1000
	totalServerKWhPer1M := (totalServerJoulesPer1K * 1000.0) / 3.6e6
	elecCostPer1M = totalServerKWhPer1M * econ.ElectricityRateKWh
	costPer1M := hwCostPer1M + elecCostPer1M

	// SLA check
	effectiveP95 := m.P95LatencyMS
	if effectiveP95 <= 0 {
		effectiveP95 = m.LatencyMS
	}
	meetsSLA := true
	if econ.LatencySLAMS > 0 && effectiveP95 > econ.LatencySLAMS {
		meetsSLA = false
	}

	// Fleet sizing for TargetQPS
	gpusRequired := 1
	if econ.TargetQPS > 0 && rps > 0 {
		gpusRequired = int(math.Ceil(econ.TargetQPS / rps))
		if gpusRequired < 1 {
			gpusRequired = 1
		}
	}

	// Monthly fleet cost, energy, and carbon footprint
	monthlyHWCost := float64(gpusRequired) * monthlyHours * econ.GPUHourlyRateUSD
	perGPUTotalKW := ((avgPowerW + econ.ServerOverheadWatts) * pue) / 1000.0
	monthlyEnergyKWh := float64(gpusRequired) * perGPUTotalKW * monthlyHours
	monthlyElecCost := monthlyEnergyKWh * econ.ElectricityRateKWh
	monthlyTotalCost := monthlyHWCost + monthlyElecCost
	monthlyCarbonKg := (monthlyEnergyKWh * econ.CarbonIntensityGCO2) / 1000.0

	return BusinessKPIs{
		CostPer1MInferencesUSD:    roundTo4(costPer1M),
		HardwareCostPer1MUSD:      roundTo4(hwCostPer1M),
		ElectricityCostPer1MUSD:   roundTo4(elecCostPer1M),
		JoulesPer1KInferences:     roundTo2(joulesPer1K),
		TotalServerJoulesPer1K:    roundTo2(totalServerJoulesPer1K),
		RequestsPerGPUHour:        roundTo2(requestsPerGPUHour),
		GPUsRequiredForTargetQPS:  gpusRequired,
		MonthlyHardwareCostUSD:    roundTo2(monthlyHWCost),
		MonthlyElectricityCostUSD: roundTo2(monthlyElecCost),
		MonthlyTotalCostUSD:       roundTo2(monthlyTotalCost),
		MonthlyEnergyKWh:          roundTo2(monthlyEnergyKWh),
		MonthlyCarbonKgCO2e:       roundTo2(monthlyCarbonKg),
		MeetsLatencySLA:           meetsSLA,
	}
}

// CompareWorkloadEconomics compares baseline and optimized workload metrics under the same economics configuration.
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
	monthlySavings := baseKPI.MonthlyTotalCostUSD - optKPI.MonthlyTotalCostUSD
	monthlyKWhSaved := baseKPI.MonthlyEnergyKWh - optKPI.MonthlyEnergyKWh

	// If dedicated instance count didn't drop (e.g. both fit on 1 GPU), calculate elastic GPU-hour
	// and active energy savings over the monthly request volume so sub-fleet optimizations still quantify dollar & kWh impact.
	if monthlySavings <= 0 && baseKPI.CostPer1MInferencesUSD > optKPI.CostPer1MInferencesUSD {
		monthlyHrs := econ.MonthlyOperatingHrs
		if monthlyHrs <= 0 {
			monthlyHrs = 730.0
		}
		qps := econ.TargetQPS
		if qps <= 0 {
			qps = 120.0
		}
		monthlyMillions := (qps * monthlyHrs * 3600.0) / 1e6
		monthlySavings = (baseKPI.CostPer1MInferencesUSD - optKPI.CostPer1MInferencesUSD) * monthlyMillions
		if monthlyKWhSaved <= 0 && baseKPI.TotalServerJoulesPer1K > optKPI.TotalServerJoulesPer1K {
			monthlyThousands := monthlyMillions * 1000.0
			joulesSaved := (baseKPI.TotalServerJoulesPer1K - optKPI.TotalServerJoulesPer1K) * monthlyThousands
			monthlyKWhSaved = joulesSaved / 3.6e6
		}
	}
	annualSavings := monthlySavings * 12.0
	monthlyCarbonSaved := (monthlyKWhSaved * econ.CarbonIntensityGCO2) / 1000.0

	costPer1MReductionPct := 0.0
	if baseKPI.CostPer1MInferencesUSD > 0 {
		costPer1MReductionPct = ((baseKPI.CostPer1MInferencesUSD - optKPI.CostPer1MInferencesUSD) / baseKPI.CostPer1MInferencesUSD) * 100.0
	}

	return SavingsComparison{
		Baseline:                 baseKPI,
		Optimized:                optKPI,
		SpeedupFactor:            roundTo2(speedup),
		ThroughputGainPct:        roundTo2(throughputGainPct),
		EnergySavedPct:           roundTo2(energySavedPct),
		VRAMReductionMiB:         roundTo2(vramReduction),
		GPUsEliminated:           gpusEliminated,
		MonthlySavingsUSD:        roundTo2(monthlySavings),
		AnnualSavingsUSD:         roundTo2(annualSavings),
		MonthlyKWhSaved:          roundTo2(monthlyKWhSaved),
		MonthlyCarbonSavedKgCO2e: roundTo2(monthlyCarbonSaved),
		CostPer1MReductionPct:    roundTo2(costPer1MReductionPct),
	}
}

func roundTo4(v float64) float64 {
	return math.Round(v*10000.0) / 10000.0
}
