package config

import "time"

// EconomicsConfig holds user-configurable tariff, infrastructure, and SLA parameters
// used to translate CUDA-level latency, energy, and VRAM into business metrics.
type EconomicsConfig struct {
	GPUHourlyRateUSD    float64 `json:"gpuHourlyRateUsd"`    // e.g. $1.50/hr for GPU instance or amortized hardware rate
	ElectricityRateKWh  float64 `json:"electricityRateKwh"`  // e.g. $0.14/kWh
	DatacenterPUE       float64 `json:"datacenterPue"`       // Power Usage Effectiveness (e.g. 1.30)
	ServerOverheadWatts float64 `json:"serverOverheadWatts"` // Host CPU/RAM/cooling share per GPU (e.g. 35W)
	CarbonIntensityGCO2 float64 `json:"carbonIntensityGco2"` // Grid gCO2e per kWh (e.g. 385 g/kWh)
	TargetQPS           float64 `json:"targetQps"`           // Required workload requests/sec
	LatencySLAMS        float64 `json:"latencySlaMs"`        // Maximum acceptable p95 latency in ms
	MonthlyOperatingHrs float64 `json:"monthlyOperatingHrs"` // Default 730 hours/month
	AlphaCostWeight     float64 `json:"alphaCostWeight"`     // Multi-objective weight for Cost/Time
	BetaEnergyWeight    float64 `json:"betaEnergyWeight"`    // Multi-objective weight for Energy
	GammaMemoryWeight   float64 `json:"gammaMemoryWeight"`   // Multi-objective weight for VRAM/Latency
}

// DefaultEconomicsConfig returns sensible defaults that can be adjusted live in the Web Dashboard.
func DefaultEconomicsConfig() EconomicsConfig {
	return EconomicsConfig{
		GPUHourlyRateUSD:    1.50,
		ElectricityRateKWh:  0.14,
		DatacenterPUE:       1.30,
		ServerOverheadWatts: 35.0,
		CarbonIntensityGCO2: 385.0,
		TargetQPS:           120.0,
		LatencySLAMS:        25.0,
		MonthlyOperatingHrs: 730.0,
		AlphaCostWeight:     0.45,
		BetaEnergyWeight:    0.35,
		GammaMemoryWeight:   0.20,
	}
}

// DaemonConfig controls the Go server, CUDA paths, and sampling intervals.
type DaemonConfig struct {
	ListenAddr        string          `json:"listenAddr"`
	DataDir           string          `json:"dataDir"`
	NVCCPath          string          `json:"nvccPath"`
	SanitizerPath     string          `json:"sanitizerPath"`
	NvidiaSMIPath     string          `json:"nvidiaSmiPath"`
	PythonPath        string          `json:"pythonPath"`
	TelemetryInterval time.Duration   `json:"telemetryInterval"`
	CompileWorkers    int             `json:"compileWorkers"`
	KernelTimeout     time.Duration   `json:"kernelTimeout"`
	Economics         EconomicsConfig `json:"economics"`
}

// DefaultDaemonConfig returns a ready-to-run configuration for Linux + CUDA.
func DefaultDaemonConfig() DaemonConfig {
	return DaemonConfig{
		ListenAddr:        "127.0.0.1:8080",
		DataDir:           ".gpumode_data",
		NVCCPath:          "/opt/cuda/bin/nvcc",
		SanitizerPath:     "/opt/cuda/bin/compute-sanitizer",
		NvidiaSMIPath:     "nvidia-smi",
		PythonPath:        "python3",
		TelemetryInterval: 200 * time.Millisecond,
		CompileWorkers:    3,
		KernelTimeout:     25 * time.Second,
		Economics:         DefaultEconomicsConfig(),
	}
}
