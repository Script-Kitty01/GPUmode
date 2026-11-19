package config

import (
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

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
	GPUIndex          int             `json:"gpuIndex"`
	NVCCPath          string          `json:"nvccPath"`
	SanitizerPath     string          `json:"sanitizerPath"`
	NvidiaSMIPath     string          `json:"nvidiaSmiPath"`
	PythonPath        string          `json:"pythonPath"`
	PresetsPath       string          `json:"presetsPath,omitempty"`
	DeepSeekModelPath string          `json:"deepseekModelPath,omitempty"`
	TelemetryInterval time.Duration   `json:"telemetryInterval"`
	CompileWorkers    int             `json:"compileWorkers"`
	KernelTimeout     time.Duration   `json:"kernelTimeout"`
	Economics         EconomicsConfig `json:"economics"`
}

// ResolveExecutable finds a binary dynamically via an optional override env var,
// PATH lookup, CUDA_HOME/CUDA_PATH, and standard CUDA installation directories.
func ResolveExecutable(envVar, name string, cudaSubdirs ...string) string {
	if envVar != "" {
		if v := strings.TrimSpace(os.Getenv(envVar)); v != "" {
			return v
		}
	}
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	for _, cudaEnv := range []string{"CUDA_HOME", "CUDA_PATH"} {
		if root := strings.TrimSpace(os.Getenv(cudaEnv)); root != "" {
			cand := filepath.Join(root, "bin", name)
			if _, err := os.Stat(cand); err == nil {
				return cand
			}
		}
	}
	defaultDirs := []string{"/usr/local/cuda/bin", "/opt/cuda/bin", "/usr/bin"}
	defaultDirs = append(defaultDirs, cudaSubdirs...)
	for _, dir := range defaultDirs {
		if dir == "" {
			continue
		}
		cand := filepath.Join(dir, name)
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	return name
}

// ResolveNVCCPath locates the CUDA compiler dynamically.
func ResolveNVCCPath() string {
	return ResolveExecutable("GPUMODE_NVCC", "nvcc")
}

// ResolveSanitizerPath locates compute-sanitizer dynamically.
func ResolveSanitizerPath() string {
	return ResolveExecutable("GPUMODE_SANITIZER", "compute-sanitizer")
}

// ResolveNvidiaSMIPath locates nvidia-smi dynamically.
func ResolveNvidiaSMIPath() string {
	return ResolveExecutable("GPUMODE_NVIDIA_SMI", "nvidia-smi")
}

// ResolvePythonPath locates Python 3 dynamically.
func ResolvePythonPath() string {
	if v := strings.TrimSpace(os.Getenv("GPUMODE_PYTHON")); v != "" {
		return v
	}
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	return "python3"
}

// ResolveDeepSeekModelPath finds a configured GGUF model or the first local GGUF
// model in common model directories. It intentionally avoids a machine-specific default path.
func ResolveDeepSeekModelPath() string {
	if v := strings.TrimSpace(os.Getenv("GPUMODE_MODEL_PATH")); v != "" {
		if _, err := os.Stat(v); err == nil {
			return v
		}
	}

	searchDirs := []string{}
	if dir := strings.TrimSpace(os.Getenv("GPUMODE_MODEL_DIR")); dir != "" {
		searchDirs = append(searchDirs, dir)
	}
	if home, err := os.UserHomeDir(); err == nil {
		searchDirs = append(searchDirs,
			filepath.Join(home, "AI", "Models"),
			filepath.Join(home, "models"),
			filepath.Join(home, ".cache", "huggingface", "hub"),
		)
	}
	searchDirs = append(searchDirs, "models")

	for _, dir := range searchDirs {
		if dir == "" {
			continue
		}
		if !isDir(dir) {
			continue
		}
		if model := findGGUF(dir); model != "" {
			return model
		}
	}
	return ""
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func findGGUF(root string) string {
	var found []string
	_ = filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if entry.IsDir() {
			name := strings.ToLower(entry.Name())
			if name == ".git" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.EqualFold(filepath.Ext(entry.Name()), ".gguf") {
			found = append(found, path)
		}
		return nil
	})
	if len(found) == 0 {
		return ""
	}

	// Prefer likely DeepSeek/LLM filenames, then lexically smallest path for determinism.
	var preferred string
	for _, path := range found {
		name := strings.ToLower(filepath.Base(path))
		if strings.Contains(name, "deepseek") || strings.Contains(name, "llm") || strings.Contains(name, "qwen") {
			if preferred == "" || path < preferred {
				preferred = path
			}
		}
	}
	if preferred != "" {
		return preferred
	}
	sort.Strings(found)
	return found[0]
}

// DefaultGPUIndex resolves the target GPU index from GPUMODE_GPU_INDEX or 0.
func DefaultGPUIndex() int {
	if v := strings.TrimSpace(os.Getenv("GPUMODE_GPU_INDEX")); v != "" {
		if idx, err := strconv.Atoi(v); err == nil && idx >= 0 {
			return idx
		}
	}
	return 0
}

// DefaultDaemonConfig returns a dynamically resolved configuration for Linux + NVIDIA CUDA.
func DefaultDaemonConfig() DaemonConfig {
	return DaemonConfig{
		ListenAddr:        "127.0.0.1:8080",
		DataDir:           ".gpumode_data",
		GPUIndex:          DefaultGPUIndex(),
		NVCCPath:          ResolveNVCCPath(),
		SanitizerPath:     ResolveSanitizerPath(),
		NvidiaSMIPath:     ResolveNvidiaSMIPath(),
		PythonPath:        ResolvePythonPath(),
		PresetsPath:       strings.TrimSpace(os.Getenv("GPUMODE_PRESETS_PATH")),
		DeepSeekModelPath: ResolveDeepSeekModelPath(),
		TelemetryInterval: 200 * time.Millisecond,
		CompileWorkers:    3,
		KernelTimeout:     25 * time.Second,
		Economics:         DefaultEconomicsConfig(),
	}
}
