package roofline

import (
	"fmt"
	"math"
	"strings"

	"gpumode/internal/physics"
)

// BottleneckClass categorizes why an operation or GPU workload is underutilizing hardware.
type BottleneckClass string

const (
	BottleneckMemoryBound  BottleneckClass = "MEMORY_BOUND"
	BottleneckComputeBound BottleneckClass = "COMPUTE_BOUND"
	BottleneckLaunchBound  BottleneckClass = "LAUNCH_BOUND"
	BottleneckBalanced     BottleneckClass = "BALANCED"
)

// OpProfile describes a neural network or CUDA operation's workload characteristics.
type OpProfile struct {
	Name              string  `json:"name"`
	Category          string  `json:"category"` // e.g. "rmsnorm", "rmsnorm_residual", "layernorm", "softmax", "swiglu", "gemm"
	Elements          int64   `json:"elements"`
	HiddenDim         int     `json:"hiddenDim"`
	BatchSize         int     `json:"batchSize"`
	FLOPs             float64 `json:"flops"`
	BytesTransferred  float64 `json:"bytesTransferred"`
	MeasuredLatencyMS float64 `json:"measuredLatencyMs"`
	LaunchCount       int     `json:"launchCount"`
	ActivationVRAMMiB float64 `json:"activationVramMib"`
}

// Diagnosis explains why the GPU is underutilized on a given operation and prescribes targeted CUDA optimizations.
type Diagnosis struct {
	OpName                  string          `json:"opName"`
	Category                string          `json:"category"`
	Classification          BottleneckClass `json:"classification"`
	ArithmeticIntensity     float64         `json:"arithmeticIntensity"`     // FLOPs / Byte
	HardwareRidgePoint      float64         `json:"hardwareRidgePoint"`      // Peak FLOPs / Peak Bandwidth
	RooflineCeilingTFLOPS   float64         `json:"rooflineCeilingTflops"`   // Min(PeakFP32, AI * PeakBW)
	AchievedTFLOPS          float64         `json:"achievedTflops"`          // FLOPs / (LatencySec * 1e12)
	AchievedBandwidthGBps   float64         `json:"achievedBandwidthGbps"`   // Bytes / (LatencySec * 1e9)
	ComputeUtilizationPct   float64         `json:"computeUtilizationPct"`   // AchievedTFLOPS / PeakFP32TFLOPS * 100
	BandwidthUtilizationPct float64         `json:"bandwidthUtilizationPct"` // AchievedBW / PeakBW * 100
	RooflineEfficiencyPct   float64         `json:"rooflineEfficiencyPct"`   // AchievedTFLOPS / RooflineCeilingTFLOPS * 100
	UnderutilizationReason  string          `json:"underutilizationReason"`
	RecommendedStrategies   []string        `json:"recommendedStrategies"`
	ExpectedVRAMSaveMiB     float64         `json:"expectedVramSaveMib"`
}

// RooflinePoint is used by the Web Dashboard Canvas chart to plot operations against the Roofline curve.
type RooflinePoint struct {
	Label               string          `json:"label"`
	ArithmeticIntensity float64         `json:"arithmeticIntensity"`
	AchievedTFLOPS      float64         `json:"achievedTflops"`
	CeilingTFLOPS       float64         `json:"ceilingTflops"`
	Classification      BottleneckClass `json:"classification"`
	IsOptimized         bool            `json:"isOptimized"`
}

// EstimateOpProfile builds an analytical OpProfile for standard LLM/inference operations when given dimensions and latency.
func EstimateOpProfile(category string, batchSize, hiddenDim int, latencyMS float64) OpProfile {
	if batchSize <= 0 {
		batchSize = 32
	}
	if hiddenDim <= 0 {
		hiddenDim = 4096
	}
	n := int64(batchSize) * int64(hiddenDim)
	cat := strings.ToLower(strings.TrimSpace(category))

	var flops, bytes float64
	var launches int
	var vramMiB float64
	name := category

	switch cat {
	case "rmsnorm":
		name = "RMSNorm (Unfused)"
		// square + sum + rsqrt + mul weight: ~4 FLOPs/elem
		// Unfused PyTorch reads x twice, writes intermediate variance/norm, reads weight: ~16 bytes/elem
		flops = float64(n) * 4.0
		bytes = float64(n) * 16.0
		launches = 3
		vramMiB = float64(n*4*3) / (1024.0 * 1024.0)
	case "rmsnorm_residual":
		name = "RMSNorm + Residual Add"
		// residual add + square + sum + rsqrt + mul weight: ~5 FLOPs/elem
		// Unfused: add kernel (read x, read res, write sum = 12B) + rmsnorm (16B) = 28 bytes/elem
		flops = float64(n) * 5.0
		bytes = float64(n) * 28.0
		launches = 4
		vramMiB = float64(n*4*4) / (1024.0 * 1024.0)
	case "layernorm":
		name = "LayerNorm"
		// mean + sub + square + var + rsqrt + affine: ~7 FLOPs/elem
		// Unfused reads/writes: ~20 bytes/elem
		flops = float64(n) * 7.0
		bytes = float64(n) * 20.0
		launches = 4
		vramMiB = float64(n*4*3) / (1024.0 * 1024.0)
	case "softmax":
		name = "Softmax (3-Pass Unfused)"
		// max + sub + exp + sum + div: ~5 FLOPs/elem
		// 3-pass global memory traversal: read for max (4B), read+write exp (8B), read+write div (8B) = 20 bytes/elem
		flops = float64(n) * 5.0
		bytes = float64(n) * 20.0
		launches = 3
		vramMiB = float64(n*4*3) / (1024.0 * 1024.0)
	case "swiglu":
		name = "SwiGLU Activation (SiLU * Gate)"
		// sigmoid + mul + gate mul: ~6 FLOPs/elem
		// Unfused: read gate, write silu (8B), read silu + read up, write out (12B) = 20 bytes/elem
		flops = float64(n) * 6.0
		bytes = float64(n) * 20.0
		launches = 2
		vramMiB = float64(n*4*4) / (1024.0 * 1024.0)
	default:
		name = category
		flops = float64(n) * 5.0
		bytes = float64(n) * 16.0
		launches = 2
		vramMiB = float64(n*4*2) / (1024.0 * 1024.0)
	}

	return OpProfile{
		Name:              name,
		Category:          cat,
		Elements:          n,
		HiddenDim:         hiddenDim,
		BatchSize:         batchSize,
		FLOPs:             flops,
		BytesTransferred:  bytes,
		MeasuredLatencyMS: latencyMS,
		LaunchCount:       launches,
		ActivationVRAMMiB: vramMiB,
	}
}

// Diagnose analyzes an OpProfile against the GPUHardwareSpec using the Roofline Performance Model.
func Diagnose(op OpProfile, hw physics.GPUHardwareSpec) Diagnosis {
	bytes := op.BytesTransferred
	if bytes <= 0 {
		bytes = float64(op.Elements) * 12.0
	}
	flops := op.FLOPs
	if flops <= 0 {
		flops = float64(op.Elements) * 4.0
	}

	ai := flops / bytes
	ridge := hw.RidgePointFLOPsPerByte
	if ridge <= 0 && hw.PeakMemBandwidthGBps > 0 {
		ridge = (hw.PeakFP32TFLOPS * 1000.0) / hw.PeakMemBandwidthGBps
	}

	// Roofline ceiling in TFLOPS = min(PeakFP32TFLOPS, AI * PeakMemBandwidthGBps / 1000)
	bwBoundTFLOPS := (ai * hw.PeakMemBandwidthGBps) / 1000.0
	ceilingTFLOPS := math.Min(hw.PeakFP32TFLOPS, bwBoundTFLOPS)
	if ceilingTFLOPS <= 0 {
		ceilingTFLOPS = bwBoundTFLOPS
	}

	var achievedTFLOPS, achievedBWGBps float64
	if op.MeasuredLatencyMS > 0 {
		sec := op.MeasuredLatencyMS / 1000.0
		achievedTFLOPS = (flops / sec) / 1e12
		achievedBWGBps = (bytes / sec) / 1e9
	}

	computeUtil := 0.0
	if hw.PeakFP32TFLOPS > 0 {
		computeUtil = math.Min(100.0, (achievedTFLOPS/hw.PeakFP32TFLOPS)*100.0)
	}
	bwUtil := 0.0
	if hw.PeakMemBandwidthGBps > 0 {
		bwUtil = math.Min(100.0, (achievedBWGBps/hw.PeakMemBandwidthGBps)*100.0)
	}
	rooflineEff := 0.0
	if ceilingTFLOPS > 0 {
		rooflineEff = math.Min(100.0, (achievedTFLOPS/ceilingTFLOPS)*100.0)
	}

	// Determine primary bottleneck classification
	var class BottleneckClass
	var reason string
	var strategies []string
	var vramSave float64

	// Launch-bound check: small tensor or multiple unfused kernel launches with low bandwidth & compute saturation
	if op.LaunchCount >= 3 && op.Elements < 262144 && bwUtil < 25.0 && computeUtil < 15.0 {
		class = BottleneckLaunchBound
		reason = fmt.Sprintf(
			"GPU SMs are starved by %d separate kernel launches on a small working set (%.1f%% compute util, %.1f%% memory bandwidth util). Kernel launch overhead and global memory round-trips dominate execution time.",
			op.LaunchCount, computeUtil, bwUtil,
		)
	} else if ai < ridge*0.85 {
		class = BottleneckMemoryBound
		reason = fmt.Sprintf(
			"Operation has low Arithmetic Intensity (%.2f FLOPs/Byte vs GPU ridge point %.1f FLOPs/Byte). Floating-point cores sit at %.1f%% utilization waiting on DRAM memory transfers (%.1f GB/s achieved of %.1f GB/s peak).",
			ai, ridge, computeUtil, achievedBWGBps, hw.PeakMemBandwidthGBps,
		)
	} else if ai > ridge*1.15 {
		class = BottleneckComputeBound
		reason = fmt.Sprintf(
			"Operation Arithmetic Intensity (%.2f FLOPs/Byte) exceeds the GPU ridge point (%.1f FLOPs/Byte); performance is limited by SM execution pipelines and instruction throughput.",
			ai, ridge,
		)
	} else {
		class = BottleneckBalanced
		reason = fmt.Sprintf(
			"Operation sits near the hardware ridge point (AI %.2f vs %.1f FLOPs/Byte) balancing DRAM traffic and SM arithmetic.",
			ai, ridge,
		)
	}

	switch strings.ToLower(op.Category) {
	case "rmsnorm", "rmsnorm_residual":
		strategies = []string{
			"Fuse residual addition directly into the RMSNorm load phase to eliminate an entire DRAM read/write pass",
			"Vectorize global memory loads/stores using 128-bit float4 transactions for full memory coalescing",
			"Replace shared-memory reduction tree with register-level __shfl_xor_sync warp shuffles",
			"Autotune thread block size (128..512) and warps-per-row via Bayesian optimization",
		}
		vramSave = op.ActivationVRAMMiB * 0.50
	case "layernorm":
		strategies = []string{
			"Fuse mean and variance reduction into a single Welford/two-moment warp-shuffle pass",
			"Use float4 vectorized global memory loads and stores to reduce memory transaction overhead",
			"Cache row inputs in registers/shared memory to eliminate redundant DRAM re-reads",
		}
		vramSave = op.ActivationVRAMMiB * 0.40
	case "softmax":
		strategies = []string{
			"Replace 3-pass naive Softmax (max, exp+sum, div) with Single-Pass Online Softmax in shared memory/registers",
			"Use warp-level __shfl_xor_sync reductions for row max and normalization factor",
			"Eliminate intermediate exponential tensor allocation in global VRAM",
		}
		vramSave = op.ActivationVRAMMiB * 0.55
	case "swiglu":
		strategies = []string{
			"Fuse SiLU(x) = x * sigmoid(x) and elementwise gate multiplication into one kernel launch",
			"Vectorize reads of gate and up-projection tensors with aligned float4 loads",
			"Eliminate intermediate activation buffer writeback to global memory",
		}
		vramSave = op.ActivationVRAMMiB * 0.50
	default:
		if class == BottleneckMemoryBound || class == BottleneckLaunchBound {
			strategies = []string{
				"Apply float4 128-bit vectorized memory coalescing",
				"Fuse adjacent elementwise/reduction kernels to cut DRAM traffic",
				"Autotune block dimensions to maximize memory-level parallelism",
			}
			vramSave = op.ActivationVRAMMiB * 0.35
		} else {
			strategies = []string{
				"Unroll inner compute loops and reduce register spill pressure",
				"Tile working set into shared memory to improve L1/shared reuse",
			}
			vramSave = op.ActivationVRAMMiB * 0.15
		}
	}

	return Diagnosis{
		OpName:                  op.Name,
		Category:                op.Category,
		Classification:          class,
		ArithmeticIntensity:     ai,
		HardwareRidgePoint:      ridge,
		RooflineCeilingTFLOPS:   ceilingTFLOPS,
		AchievedTFLOPS:          achievedTFLOPS,
		AchievedBandwidthGBps:   achievedBWGBps,
		ComputeUtilizationPct:   computeUtil,
		BandwidthUtilizationPct: bwUtil,
		RooflineEfficiencyPct:   rooflineEff,
		UnderutilizationReason:  reason,
		RecommendedStrategies:   strategies,
		ExpectedVRAMSaveMiB:     vramSave,
	}
}
