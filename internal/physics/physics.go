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
// of a physical or reference GPU device.
type GPUHardwareSpec struct {
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
	HourlyRateHintUSD      float64 `json:"hourlyRateHintUsd,omitempty"`
}

const cudaDevQuerySource = `#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;
    int clockKHz = 0, memClockKHz = 0, busWidth = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("name=%s\nmajor=%d\nminor=%d\nsm=%d\nclockKHz=%d\nmemClockKHz=%d\nbusWidth=%d\nl2=%d\nvram=%zu\n",
        p.name, p.major, p.minor, p.multiProcessorCount, clockKHz, memClockKHz, busWidth, p.l2CacheSize, p.totalGlobalMem);
    return 0;
}
`

var (
	devQueryMu     sync.Mutex
	cachedProbeVal *cudaProbeResult
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
}

// FallbackRTX4050Spec returns the realistic fallback spec for an NVIDIA GeForce RTX 4050 Laptop GPU (sm_89).
func FallbackRTX4050Spec() GPUHardwareSpec {
	spec := GPUHardwareSpec{
		Name:                 "NVIDIA GeForce RTX 4050 Laptop GPU",
		ComputeCapability:    "8.9",
		ArchFlag:             "sm_89",
		SMCount:              20,
		MaxSMClockMHz:        2355.0,
		MaxMemClockMHz:       8001.0,
		MemoryBusWidthBits:   96,
		TotalVRAMMiB:         6141.0,
		L2CacheBytes:         24 * 1024 * 1024, // 24 MiB
		PeakFP32TFLOPS:       12.0,
		PeakMemBandwidthGBps: 192.0,
		IdlePowerWatts:       8.0,
		TDPWatts:             65.0,
		HourlyRateHintUSD:    0.35,
	}
	spec.RidgePointFLOPsPerByte = computeRidgePoint(spec.PeakFP32TFLOPS, spec.PeakMemBandwidthGBps)
	return spec
}

// DetectHardware probes the live system GPU using nvidia-smi and an optional compiled CUDA
// device property probe via nvcc. If neither tool succeeds, it gracefully falls back to a
// realistic RTX 4050 Laptop GPU (sm_89) specification.
func DetectHardware(ctx context.Context, nvccPath, nvidiaSMIPath string) (GPUHardwareSpec, error) {
	fallback := FallbackRTX4050Spec()

	smiData, smiErr := queryNvidiaSMI(ctx, nvidiaSMIPath)
	probeData, probeErr := runCUDAProbe(ctx, nvccPath)

	if smiErr != nil && probeErr != nil {
		return fallback, fmt.Errorf("hardware probe fallback used (nvidia-smi err: %v, nvcc err: %v)", smiErr, probeErr)
	}

	spec := fallback

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
		// Prefer boost clock from CUDA device properties if reasonable, or use nvidia-smi clock
		if probeErr != nil && smiData.MaxSMClockMHz > 0 {
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
	return spec, nil
}

// ReferenceHardwarePresets returns 4 reference hardware specs:
// 1. Local Detected GPU (or RTX 4050 Laptop GPU fallback)
// 2. NVIDIA L4 24GB
// 3. NVIDIA L40S 48GB
// 4. NVIDIA A100 80GB
func ReferenceHardwarePresets() []GPUHardwareSpec {
	local, _ := DetectHardware(context.Background(), "/opt/cuda/bin/nvcc", "nvidia-smi")
	if local.HourlyRateHintUSD == 0 {
		local.HourlyRateHintUSD = 0.35
	}

	l4 := GPUHardwareSpec{
		Name:                 "NVIDIA L4 24GB",
		ComputeCapability:    "8.9",
		ArchFlag:             "sm_89",
		SMCount:              58,
		MaxSMClockMHz:        2040.0,
		MaxMemClockMHz:       6250.0,
		MemoryBusWidthBits:   192,
		TotalVRAMMiB:         24576.0,
		L2CacheBytes:         48 * 1024 * 1024,
		PeakFP32TFLOPS:       30.29,
		PeakMemBandwidthGBps: 300.0,
		IdlePowerWatts:       18.0,
		TDPWatts:             72.0,
		HourlyRateHintUSD:    0.85,
	}
	l4.RidgePointFLOPsPerByte = computeRidgePoint(l4.PeakFP32TFLOPS, l4.PeakMemBandwidthGBps)

	l40s := GPUHardwareSpec{
		Name:                 "NVIDIA L40S 48GB",
		ComputeCapability:    "8.9",
		ArchFlag:             "sm_89",
		SMCount:              142,
		MaxSMClockMHz:        2520.0,
		MaxMemClockMHz:       9000.0,
		MemoryBusWidthBits:   384,
		TotalVRAMMiB:         49152.0,
		L2CacheBytes:         96 * 1024 * 1024,
		PeakFP32TFLOPS:       91.61,
		PeakMemBandwidthGBps: 864.0,
		IdlePowerWatts:       45.0,
		TDPWatts:             350.0,
		HourlyRateHintUSD:    1.95,
	}
	l40s.RidgePointFLOPsPerByte = computeRidgePoint(l40s.PeakFP32TFLOPS, l40s.PeakMemBandwidthGBps)

	a100 := GPUHardwareSpec{
		Name:                 "NVIDIA A100 80GB SXM",
		ComputeCapability:    "8.0",
		ArchFlag:             "sm_80",
		SMCount:              108,
		MaxSMClockMHz:        1410.0,
		MaxMemClockMHz:       1593.0,
		MemoryBusWidthBits:   5120,
		TotalVRAMMiB:         81920.0,
		L2CacheBytes:         40 * 1024 * 1024,
		PeakFP32TFLOPS:       19.49,
		PeakMemBandwidthGBps: 2039.0,
		IdlePowerWatts:       55.0,
		TDPWatts:             400.0,
		HourlyRateHintUSD:    3.40,
	}
	a100.RidgePointFLOPsPerByte = computeRidgePoint(a100.PeakFP32TFLOPS, a100.PeakMemBandwidthGBps)

	return []GPUHardwareSpec{local, l4, l40s, a100}
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

func queryNvidiaSMI(ctx context.Context, nvidiaSMIPath string) (*smiQueryResult, error) {
	if strings.TrimSpace(nvidiaSMIPath) == "" {
		return nil, fmt.Errorf("empty nvidia-smi path")
	}
	cmd := exec.CommandContext(ctx, nvidiaSMIPath,
		"--query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.memory,power.draw,enforced.power.limit,power.default_limit",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		// Retry with the basic 6-column query in case enforced.power.limit is unsupported on older drivers
		cmdBasic := exec.CommandContext(ctx, nvidiaSMIPath,
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

func runCUDAProbe(ctx context.Context, nvccPath string) (*cudaProbeResult, error) {
	if strings.TrimSpace(nvccPath) == "" {
		return nil, fmt.Errorf("empty nvcc path")
	}
	if _, err := exec.LookPath(nvccPath); err != nil {
		return nil, err
	}

	devQueryMu.Lock()
	defer devQueryMu.Unlock()

	if cachedProbeVal != nil {
		return cachedProbeVal, nil
	}

	cacheDir := ".gpumode_cache"
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return nil, err
	}
	srcPath := filepath.Join(cacheDir, "gpumode_devquery.cu")
	binPath := filepath.Join(cacheDir, "gpumode_devquery")

	if _, err := os.Stat(binPath); err != nil {
		if err := os.WriteFile(srcPath, []byte(cudaDevQuerySource), 0o644); err != nil {
			return nil, err
		}
		compileCmd := exec.CommandContext(ctx, nvccPath, "-O2", srcPath, "-o", binPath)
		if out, err := compileCmd.CombinedOutput(); err != nil {
			return nil, fmt.Errorf("nvcc compile failed: %w (%s)", err, strings.TrimSpace(string(out)))
		}
	}

	runCmd := exec.CommandContext(ctx, "./"+binPath)
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
		}
	}

	if res.SMCount == 0 {
		return nil, fmt.Errorf("invalid devquery output")
	}
	cachedProbeVal = res
	return res, nil
}

func fp32CoresPerSM(computeCap string) int {
	switch computeCap {
	case "8.9", "8.6", "8.7": // Ada Lovelace & Ampere GA10x
		return 128
	case "8.0", "9.0", "7.5", "7.0": // A100, H100, Turing, Volta
		return 64
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
