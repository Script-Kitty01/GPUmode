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
	Name                   string  `json:"name"`
	Category               string  `json:"category"` // e.g. "rmsnorm", "rmsnorm_residual", "layernorm", "softmax", "swiglu", "gemm"
	Elements               int64   `json:"elements"`
	HiddenDim              int     `json:"hiddenDim"`
	BatchSize              int     `json:"batchSize"`
	FLOPs                  float64 `json:"flops"`
	BytesTransferred       float64 `json:"bytesTransferred"`
	MeasuredLatencyMS      float64 `json:"measuredLatencyMs"`
	LaunchCount            int     `json:"launchCount"`
	ActivationVRAMMiB      float64 `json:"activationVramMib"`
	HostLaunchGapUS        float64 `json:"hostLaunchGapUs,omitempty"`
	HostSyncStallPct       float64 `json:"hostSyncStallPct,omitempty"`
	HostStarvationPct      float64 `json:"hostStarvationPct,omitempty"`
	HostBoundaryProvenance string  `json:"hostBoundaryProvenance,omitempty"`
}

// Diagnosis explains why the GPU is underutilized on a given operation and prescribes targeted CUDA optimizations.
type Diagnosis struct {
	OpName                      string          `json:"opName"`
	Category                    string          `json:"category"`
	Classification              BottleneckClass `json:"classification"`
	ArithmeticIntensity         float64         `json:"arithmeticIntensity"`         // FLOPs / Byte
	HardwareRidgePoint          float64         `json:"hardwareRidgePoint"`          // Peak FLOPs / Peak Bandwidth
	RooflineCeilingTFLOPS       float64         `json:"rooflineCeilingTflops"`       // Min(PeakFP32, AI * PeakBW)
	TensorCoreFP16CeilingTFLOPS float64         `json:"tensorCoreFp16CeilingTflops,omitempty"`
	EmpiricalDRAMCeilingGBps    float64         `json:"empiricalDramCeilingGbps,omitempty"`
	EmpiricalL2CeilingGBps      float64         `json:"empiricalL2CeilingGbps,omitempty"`
	AchievedTFLOPS              float64         `json:"achievedTflops"`              // FLOPs / (LatencySec * 1e12)
	AchievedBandwidthGBps       float64         `json:"achievedBandwidthGbps"`       // Bytes / (LatencySec * 1e9)
	ComputeUtilizationPct       float64         `json:"computeUtilizationPct"`       // AchievedTFLOPS / PeakFP32TFLOPS * 100
	BandwidthUtilizationPct     float64         `json:"bandwidthUtilizationPct"`     // AchievedBW / PeakBW * 100
	L2BandwidthUtilizationPct   float64         `json:"l2BandwidthUtilizationPct,omitempty"`
	RooflineEfficiencyPct       float64         `json:"rooflineEfficiencyPct"`       // AchievedTFLOPS / RooflineCeilingTFLOPS * 100
	LaunchCount                 int             `json:"launchCount,omitempty"`
	HostLaunchGapUS             float64         `json:"hostLaunchGapUs,omitempty"`
	HostSyncStallPct            float64         `json:"hostSyncStallPct,omitempty"`
	HostStarvationPct           float64         `json:"hostStarvationPct,omitempty"`
	HostBoundaryProvenance      string          `json:"hostBoundaryProvenance,omitempty"`
	UnderutilizationReason      string          `json:"underutilizationReason"`
	RecommendedStrategies       []string        `json:"recommendedStrategies"`
	ExpectedVRAMSaveMiB         float64         `json:"expectedVramSaveMib"`
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
	case "rope_qkv":
		name = "Rotary Positional Embedding (RoPE + QKV Bias)"
		// complex rotation (4 mul + 2 add) + bias add: ~7 FLOPs/elem
		// Unfused: bias pass (read+write 8B) + sin/cos table + RoPE read/write (16B) = 24 bytes/elem
		flops = float64(n) * 7.0
		bytes = float64(n) * 24.0
		launches = 3
		vramMiB = float64(n*4*3) / (1024.0 * 1024.0)
	case "qgemv_decode":
		name = "Quantized Decode GEMV (Q8_0 Weight Dequant + Dot)"
		// scale mul + FMA per weight element: ~3 FLOPs/elem
		// Unfused: read Q8_0 (1.125B) + write FP32 dequant buffer (4B) + read FP32 + vec (8B) = 13.125 bytes/elem
		flops = float64(n) * 3.0
		bytes = float64(n) * 13.125
		launches = 2
		vramMiB = float64(n*4*2) / (1024.0 * 1024.0)
	case "gqa_attention":
		name = "Grouped-Query Attention Decode (FlashDecoding Split-K)"
		// Q*K dot + online softmax + V accumulate: ~8 FLOPs/elem
		// Unfused: QK score writeback + softmax + V weighted sum = 24 bytes/elem
		flops = float64(n) * 8.0
		bytes = float64(n) * 24.0
		launches = 3
		vramMiB = float64(n*4*3) / (1024.0 * 1024.0)
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
	empDRAM := hw.EmpiricalDRAMBandwidthGBps
	if empDRAM <= 0 && hw.PeakMemBandwidthGBps > 0 {
		empDRAM = hw.PeakMemBandwidthGBps * 0.84
	}
	empL2 := hw.EmpiricalL2BandwidthGBps
	if empL2 <= 0 && hw.PeakMemBandwidthGBps > 0 {
		empL2 = hw.PeakMemBandwidthGBps * 3.10
	}
	l2Util := 0.0
	if empL2 > 0 {
		l2Util = math.Min(100.0, (achievedBWGBps/empL2)*100.0)
	}
	tensorCeiling := hw.PeakTensorFP16TFLOPS
	if tensorCeiling <= 0 && hw.PeakFP32TFLOPS > 0 {
		tensorCeiling = hw.PeakFP32TFLOPS * 2.0
	}
	rooflineEff := 0.0
	if ceilingTFLOPS > 0 {
		rooflineEff = math.Min(100.0, (achievedTFLOPS/ceilingTFLOPS)*100.0)
	}

	hostProv := strings.TrimSpace(op.HostBoundaryProvenance)
	if hostProv == "" {
		hostProv = "ANALYTICAL"
	}

	// Determine primary bottleneck classification
	var class BottleneckClass
	var reason string
	var strategies []string
	var vramSave float64

	// Launch-bound check: small tensor, multiple unfused kernel launches, or host-side launch gap / scheduler starvation
	smallMultiLaunch := op.LaunchCount >= 3 && op.Elements < 262144
	hostStarvedLaunch := op.HostLaunchGapUS >= 40.0 || op.HostStarvationPct >= 25.0
	if bwUtil < 25.0 && computeUtil < 15.0 && (smallMultiLaunch || hostStarvedLaunch) {
		class = BottleneckLaunchBound
		if hostProv == "MEASURED_EBPF" {
			reason = fmt.Sprintf(
				"GPU SMs are starved at the host-device launch boundary (%.1f%% compute util, %.1f%% memory bandwidth util): eBPF measured %.1f µs inter-launch host gap and %.1f%% off-CPU scheduler starvation across %d kernel launches.",
				computeUtil, bwUtil, op.HostLaunchGapUS, op.HostStarvationPct, op.LaunchCount,
			)
		} else if smallMultiLaunch {
			reason = fmt.Sprintf(
				"GPU SMs are starved by %d separate kernel launches on a small working set (%.1f%% compute util, %.1f%% memory bandwidth util). Kernel launch overhead and global memory round-trips dominate execution time.",
				op.LaunchCount, computeUtil, bwUtil,
			)
		} else {
			reason = fmt.Sprintf(
				"GPU SMs are starved by host-side kernel launch overhead (%.1f µs inter-launch host gap, %.1f%% scheduler starvation, %.1f%% compute util, %.1f%% memory bandwidth util).",
				op.HostLaunchGapUS, op.HostStarvationPct, computeUtil, bwUtil,
			)
		}
	} else if ai < ridge*0.85 {
		class = BottleneckMemoryBound
		reason = fmt.Sprintf(
			"Operation has low Arithmetic Intensity (%.2f FLOPs/Byte vs GPU ridge point %.1f FLOPs/Byte). Floating-point cores sit at %.1f%% utilization waiting on DRAM memory transfers (%.1f GB/s achieved of %.1f GB/s peak).",
			ai, ridge, computeUtil, achievedBWGBps, hw.PeakMemBandwidthGBps,
		)
		if hostProv == "MEASURED_EBPF" {
			reason += fmt.Sprintf(
				" eBPF host trace measured %.1f µs inter-launch gap and %.1f%% CUDA sync stall.",
				op.HostLaunchGapUS, op.HostSyncStallPct,
			)
		}
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
		OpName:                      op.Name,
		Category:                    op.Category,
		Classification:              class,
		ArithmeticIntensity:         ai,
		HardwareRidgePoint:          ridge,
		RooflineCeilingTFLOPS:       ceilingTFLOPS,
		TensorCoreFP16CeilingTFLOPS: tensorCeiling,
		EmpiricalDRAMCeilingGBps:    empDRAM,
		EmpiricalL2CeilingGBps:      empL2,
		AchievedTFLOPS:              achievedTFLOPS,
		AchievedBandwidthGBps:       achievedBWGBps,
		ComputeUtilizationPct:       computeUtil,
		BandwidthUtilizationPct:     bwUtil,
		L2BandwidthUtilizationPct:   l2Util,
		RooflineEfficiencyPct:       rooflineEff,
		LaunchCount:                 op.LaunchCount,
		HostLaunchGapUS:             op.HostLaunchGapUS,
		HostSyncStallPct:            op.HostSyncStallPct,
		HostStarvationPct:           op.HostStarvationPct,
		HostBoundaryProvenance:      hostProv,
		UnderutilizationReason:      reason,
		RecommendedStrategies:       strategies,
		ExpectedVRAMSaveMiB:         vramSave,
	}
}
