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

// EconomicsConfig holds workload SLA and multi-objective optimization weights.
type EconomicsConfig struct {
	TargetQPS         float64 `json:"targetQps"`         // Required workload requests/sec
	LatencySLAMS      float64 `json:"latencySlaMs"`      // Maximum acceptable p95 latency in ms
	AlphaCostWeight   float64 `json:"alphaCostWeight"`   // Multi-objective weight for Latency/Time
	BetaEnergyWeight  float64 `json:"betaEnergyWeight"`  // Multi-objective weight for Energy
	GammaMemoryWeight float64 `json:"gammaMemoryWeight"` // Multi-objective weight for VRAM
}

// DefaultEconomicsConfig returns sensible defaults for workload SLA and multi-objective weights.
func DefaultEconomicsConfig() EconomicsConfig {
	return EconomicsConfig{
		TargetQPS:         120.0,
		LatencySLAMS:      25.0,
		AlphaCostWeight:   0.45,
		BetaEnergyWeight:  0.35,
		GammaMemoryWeight: 0.20,
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
	TelemetryBackend  string          `json:"telemetryBackend"`
	EBPFMode          string          `json:"ebpfMode"`
	TracePID          int             `json:"tracePid,omitempty"`
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

// ResolveTelemetryBackend resolves GPUMODE_TELEMETRY into "auto", "nvml", or "smi" (default "auto").
func ResolveTelemetryBackend() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GPUMODE_TELEMETRY"))) {
	case "nvml":
		return "nvml"
	case "smi":
		return "smi"
	case "auto", "":
		return "auto"
	default:
		return "auto"
	}
}

// ResolveEBPFMode resolves GPUMODE_EBPF into "auto", "on", or "off" (default "auto").
func ResolveEBPFMode() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GPUMODE_EBPF"))) {
	case "on":
		return "on"
	case "off":
		return "off"
	case "auto", "":
		return "auto"
	default:
		return "auto"
	}
}

// ResolveTracePID resolves GPUMODE_TRACE_PID into a positive target PID or 0 (auto-discover).
func ResolveTracePID() int {
	if v := strings.TrimSpace(os.Getenv("GPUMODE_TRACE_PID")); v != "" {
		if pid, err := strconv.Atoi(v); err == nil && pid > 0 {
			return pid
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
		TelemetryBackend:  ResolveTelemetryBackend(),
		EBPFMode:          ResolveEBPFMode(),
		TracePID:          ResolveTracePID(),
		CompileWorkers:    3,
		KernelTimeout:     25 * time.Second,
		Economics:         DefaultEconomicsConfig(),
	}
}
