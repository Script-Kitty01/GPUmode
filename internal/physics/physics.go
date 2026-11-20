package physics

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// GPUHardwareSpec captures the hardware physics ceiling and architecture traits
// of a physical NVIDIA GPU device.
type GPUHardwareSpec struct {
	GPUIndex               int     `json:"gpuIndex"`
	Name                   string  `json:"name"`
	ComputeCapability      string  `json:"computeCapability"`
	ArchFlag               string  `json:"archFlag"`
	SMCount                int     `json:"smCount"`
	MaxSMClockMHz          float64 `json:"maxSmClockMhz"`
	MaxMemClockMHz         float64 `json:"maxMemClockMhz"`
	MemoryBusWidthBits     int     `json:"memoryBusWidthBits"`
	TotalVRAMMiB           float64 `json:"totalVramMib"`
	L2CacheBytes           int     `json:"l2CacheBytes"`
	PeakFP32TFLOPS         float64 `json:"peakFp32Tflops"`
	PeakMemBandwidthGBps   float64 `json:"peakMemBandwidthGbps"`
	RidgePointFLOPsPerByte float64 `json:"ridgePointFlopsPerByte"`
	IdlePowerWatts         float64 `json:"idlePowerWatts"`
	TDPWatts               float64 `json:"tdpWatts"`
}

const cudaDevQuerySource = `#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    int dev = (argc > 1) ? atoi(argv[1]) : 0;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return 1;
    if (dev < 0 || dev >= count) return 2;
    if (cudaSetDevice(dev) != cudaSuccess) return 3;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 4;
    int clockKHz = 0, memClockKHz = 0, busWidth = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, dev);
    printf("name=%s\nmajor=%d\nminor=%d\nsm=%d\nclockKHz=%d\nmemClockKHz=%d\nbusWidth=%d\nl2=%d\nvram=%zu\ndeviceCount=%d\n",
        p.name, p.major, p.minor, p.multiProcessorCount, clockKHz, memClockKHz, busWidth, p.l2CacheSize, p.totalGlobalMem, count);
    return 0;
}
`

var (
	devQueryMu     sync.Mutex
	cachedProbeMap = make(map[int]*cudaProbeResult)
)

type cudaProbeResult struct {
	Name        string
	Major       int
	Minor       int
	SMCount     int
	ClockMHz    float64
	MemClockMHz float64
	BusWidth    int
	L2Bytes     int
	VRAMBytes   int64
	DeviceCount int
}

// DetectHardware probes GPU 0 using nvidia-smi and a compiled CUDA device property probe.
func DetectHardware(ctx context.Context, nvccPath, nvidiaSMIPath string) (GPUHardwareSpec, error) {
	return DetectHardwareForDevice(ctx, 0, nvccPath, nvidiaSMIPath)
}

// DetectHardwareForDevice probes the specified NVIDIA GPU index using nvidia-smi and a compiled CUDA
// device property probe via nvcc. Both sources must provide a valid live GPU specification.
func DetectHardwareForDevice(ctx context.Context, gpuIndex int, nvccPath, nvidiaSMIPath string) (GPUHardwareSpec, error) {
	if gpuIndex < 0 {
		gpuIndex = 0
	}
	smiData, smiErr := queryNvidiaSMIForDevice(ctx, gpuIndex, nvidiaSMIPath)
	probeData, probeErr := runCUDAProbeForDevice(ctx, gpuIndex, nvccPath)

	if smiErr != nil && probeErr != nil {
		return GPUHardwareSpec{GPUIndex: gpuIndex}, fmt.Errorf(
			"hardware probe failed for NVIDIA GPU %d (nvidia-smi err: %v, nvcc err: %v)",
			gpuIndex, smiErr, probeErr,
		)
	}

	spec := GPUHardwareSpec{
		GPUIndex: gpuIndex,
	}

	if probeErr == nil && probeData != nil {
		if probeData.Name != "" {
			spec.Name = probeData.Name
		}
		if probeData.Major > 0 {
			spec.ComputeCapability = fmt.Sprintf("%d.%d", probeData.Major, probeData.Minor)
			spec.ArchFlag = fmt.Sprintf("sm_%d%d", probeData.Major, probeData.Minor)
		}
		if probeData.SMCount > 0 {
			spec.SMCount = probeData.SMCount
		}
		if probeData.ClockMHz > 0 {
			spec.MaxSMClockMHz = probeData.ClockMHz
		}
		if probeData.MemClockMHz > 0 {
			spec.MaxMemClockMHz = probeData.MemClockMHz
		}
		if probeData.BusWidth > 0 {
			spec.MemoryBusWidthBits = probeData.BusWidth
		}
		if probeData.L2Bytes > 0 {
			spec.L2CacheBytes = probeData.L2Bytes
		}
		if probeData.VRAMBytes > 0 {
			spec.TotalVRAMMiB = math.Round(float64(probeData.VRAMBytes) / (1024.0 * 1024.0))
		}
	}

	if smiErr == nil && smiData != nil {
		if smiData.Name != "" {
			spec.Name = smiData.Name
		}
		if smiData.ComputeCap != "" {
			spec.ComputeCapability = smiData.ComputeCap
			archClean := strings.ReplaceAll(smiData.ComputeCap, ".", "")
			if archClean != "" {
				spec.ArchFlag = "sm_" + archClean
			}
		}
		if smiData.TotalVRAMMiB > 0 {
			spec.TotalVRAMMiB = smiData.TotalVRAMMiB
		}
		// Prefer boost clock from CUDA device properties if available, else nvidia-smi clock
		if (probeErr != nil || spec.MaxSMClockMHz <= 0) && smiData.MaxSMClockMHz > 0 {
			spec.MaxSMClockMHz = smiData.MaxSMClockMHz
		}
		if smiData.MaxMemClockMHz > 0 {
			spec.MaxMemClockMHz = smiData.MaxMemClockMHz
		}
		if smiData.PowerDrawWatts > 0 {
			spec.IdlePowerWatts = smiData.PowerDrawWatts
		}
		if smiData.TDPWatts > 0 {
			spec.TDPWatts = smiData.TDPWatts
		}
	}

	coresPerSM := fp32CoresPerSM(spec.ComputeCapability)
	smClockGHz := spec.MaxSMClockMHz / 1000.0
	memClockGHz := spec.MaxMemClockMHz / 1000.0

	// PeakFP32TFLOPS = (CoresPerSM * 2 FLOPs/cycle * SMCount * MaxSMClockGHz) / 1000
	if spec.SMCount > 0 && smClockGHz > 0 {
		spec.PeakFP32TFLOPS = roundTo2((float64(coresPerSM) * 2.0 * float64(spec.SMCount) * smClockGHz) / 1000.0)
	}
	// PeakMemBandwidthGBps = 2 * MaxMemClockGHz * (MemoryBusWidthBits / 8)
	if spec.MemoryBusWidthBits > 0 && memClockGHz > 0 {
		spec.PeakMemBandwidthGBps = roundTo2(2.0 * memClockGHz * (float64(spec.MemoryBusWidthBits) / 8.0))
	}

	spec.RidgePointFLOPsPerByte = computeRidgePoint(spec.PeakFP32TFLOPS, spec.PeakMemBandwidthGBps)
	if spec.Name == "" || spec.ComputeCapability == "" || spec.ArchFlag == "" ||
		spec.SMCount <= 0 || spec.TotalVRAMMiB <= 0 ||
		spec.PeakFP32TFLOPS <= 0 || spec.PeakMemBandwidthGBps <= 0 ||
		spec.RidgePointFLOPsPerByte <= 0 {
		return GPUHardwareSpec{GPUIndex: gpuIndex}, fmt.Errorf(
			"hardware probe produced incomplete NVIDIA GPU %d specification: %+v",
			gpuIndex, spec,
		)
	}
	return spec, nil
}

type smiQueryResult struct {
	Name           string
	ComputeCap     string
	TotalVRAMMiB   float64
	MaxSMClockMHz  float64
	MaxMemClockMHz float64
	PowerDrawWatts float64
	TDPWatts       float64
}

func queryNvidiaSMIForDevice(ctx context.Context, gpuIndex int, nvidiaSMIPath string) (*smiQueryResult, error) {
	if strings.TrimSpace(nvidiaSMIPath) == "" {
		return nil, fmt.Errorf("empty nvidia-smi path")
	}
	idxArg := strconv.Itoa(gpuIndex)
	cmd := exec.CommandContext(ctx, nvidiaSMIPath,
		"-i", idxArg,
		"--query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.memory,power.draw,enforced.power.limit,power.default_limit",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		// Retry with the basic 6-column query in case enforced.power.limit is unsupported on older drivers
		cmdBasic := exec.CommandContext(ctx, nvidiaSMIPath,
			"-i", idxArg,
			"--query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.memory,power.draw",
			"--format=csv,noheader,nounits",
		)
		out, err = cmdBasic.Output()
		if err != nil {
			return nil, err
		}
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		return nil, fmt.Errorf("empty nvidia-smi output")
	}

	parts := strings.Split(lines[0], ",")
	if len(parts) < 6 {
		return nil, fmt.Errorf("unexpected nvidia-smi csv format: %s", lines[0])
	}

	res := &smiQueryResult{
		Name:       strings.TrimSpace(parts[0]),
		ComputeCap: strings.TrimSpace(parts[1]),
	}
	if v, err := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64); err == nil {
		res.TotalVRAMMiB = v
	}
	if v, err := strconv.ParseFloat(strings.TrimSpace(parts[3]), 64); err == nil {
		res.MaxSMClockMHz = v
	}
	if v, err := strconv.ParseFloat(strings.TrimSpace(parts[4]), 64); err == nil {
		res.MaxMemClockMHz = v
	}
	if v, err := strconv.ParseFloat(strings.TrimSpace(parts[5]), 64); err == nil {
		res.PowerDrawWatts = v
	}
	if len(parts) >= 7 {
		if v, err := strconv.ParseFloat(strings.TrimSpace(parts[6]), 64); err == nil && v > 0 {
			res.TDPWatts = v
		}
	}
	if res.TDPWatts == 0 && len(parts) >= 8 {
		if v, err := strconv.ParseFloat(strings.TrimSpace(parts[7]), 64); err == nil && v > 0 {
			res.TDPWatts = v
		}
	}
	return res, nil
}

func runCUDAProbeForDevice(ctx context.Context, gpuIndex int, nvccPath string) (*cudaProbeResult, error) {
	if strings.TrimSpace(nvccPath) == "" {
		return nil, fmt.Errorf("empty nvcc path")
	}
	if _, err := exec.LookPath(nvccPath); err != nil {
		return nil, err
	}

	devQueryMu.Lock()
	defer devQueryMu.Unlock()

	if cached, ok := cachedProbeMap[gpuIndex]; ok && cached != nil {
		return cached, nil
	}

	cacheDir := filepath.Join(os.TempDir(), "gpumode_cache")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return nil, err
	}
	srcPath := filepath.Join(cacheDir, "gpumode_devquery_v2.cu")
	binPath := filepath.Join(cacheDir, "gpumode_devquery_v2")

	if _, err := os.Stat(binPath); err != nil {
		if err := os.WriteFile(srcPath, []byte(cudaDevQuerySource), 0o644); err != nil {
			return nil, err
		}
		compileCmd := exec.CommandContext(ctx, nvccPath, "-O2", srcPath, "-o", binPath)
		if out, err := compileCmd.CombinedOutput(); err != nil {
			return nil, fmt.Errorf("nvcc compile failed: %w (%s)", err, strings.TrimSpace(string(out)))
		}
	}

	runCmd := exec.CommandContext(ctx, binPath, strconv.Itoa(gpuIndex))
	out, err := runCmd.Output()
	if err != nil {
		return nil, fmt.Errorf("devquery execution failed: %w", err)
	}

	res := &cudaProbeResult{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		kv := strings.SplitN(strings.TrimSpace(line), "=", 2)
		if len(kv) != 2 {
			continue
		}
		key, val := kv[0], kv[1]
		switch key {
		case "name":
			res.Name = val
		case "major":
			res.Major, _ = strconv.Atoi(val)
		case "minor":
			res.Minor, _ = strconv.Atoi(val)
		case "sm":
			res.SMCount, _ = strconv.Atoi(val)
		case "clockKHz":
			if k, err := strconv.ParseFloat(val, 64); err == nil {
				res.ClockMHz = k / 1000.0
			}
		case "memClockKHz":
			if k, err := strconv.ParseFloat(val, 64); err == nil {
				res.MemClockMHz = k / 1000.0
			}
		case "busWidth":
			res.BusWidth, _ = strconv.Atoi(val)
		case "l2":
			res.L2Bytes, _ = strconv.Atoi(val)
		case "vram":
			res.VRAMBytes, _ = strconv.ParseInt(val, 10, 64)
		case "deviceCount":
			res.DeviceCount, _ = strconv.Atoi(val)
		}
	}

	if res.SMCount == 0 {
		return nil, fmt.Errorf("invalid devquery output")
	}
	cachedProbeMap[gpuIndex] = res
	return res, nil
}

func fp32CoresPerSM(computeCap string) int {
	switch computeCap {
	case "12.0", "10.0", "10.1": // Blackwell
		return 128
	case "9.0": // Hopper H100/H200
		return 128
	case "8.9", "8.6", "8.7": // Ada Lovelace & Ampere GA10x
		return 128
	case "8.0", "7.5", "7.0", "6.0": // A100, Turing, Volta, P100
		return 64
	case "6.1", "6.2", "5.0", "5.2", "5.3": // Pascal GP10x, Maxwell
		return 128
	case "3.0", "3.5", "3.7": // Kepler
		return 192
	default:
		return 128
	}
}

func computeRidgePoint(peakTFLOPS, peakBWGBps float64) float64 {
	if peakBWGBps <= 0 {
		return 0
	}
	// TFLOPS = 1e12 FLOPs/s; GB/s = 1e9 Bytes/s -> ratio * 1000 FLOPs/Byte
	return roundTo2((peakTFLOPS * 1000.0) / peakBWGBps)
}

func roundTo2(v float64) float64 {
	return math.Round(v*100.0) / 100.0
}
