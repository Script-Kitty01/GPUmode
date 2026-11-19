package optimizer

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"gpumode/internal/harness"
	"gpumode/internal/physics"
)

// KernelProposer is the pluggable interface for proposing candidate CUDA kernels for a given operation.
// DeterministicProposer uses Roofline bottleneck rules + parameterized CUDA templates targeting the detected GPU;
// an offline or online LLM proposer can implement this same interface without changing verification or economics.
type KernelProposer interface {
	Name() string
	ProposeCandidates(opCategory string, batchSize, hiddenDim, iterations int) (baseline harness.KernelSpec, candidates []harness.KernelSpec)
}

// DeterministicProposer generates parameterized CUDA C++ kernels guided by Roofline optimization strategies
// and the detected GPU's hardware properties.
type DeterministicProposer struct {
	Hardware     physics.GPUHardwareSpec
	PassesPerReq int
}

func NewDeterministicProposer() *DeterministicProposer {
	return &DeterministicProposer{
		PassesPerReq: defaultPassesPerReq(),
	}
}

func NewDeterministicProposerForHardware(hw physics.GPUHardwareSpec) *DeterministicProposer {
	return &DeterministicProposer{
		Hardware:     hw,
		PassesPerReq: defaultPassesPerReq(),
	}
}

func defaultPassesPerReq() int {
	if v := strings.TrimSpace(os.Getenv("GPUMODE_PASSES_PER_REQ")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 64
}

func (p *DeterministicProposer) effectivePasses() int {
	if p.PassesPerReq > 0 {
		return p.PassesPerReq
	}
	return 64
}

func (p *DeterministicProposer) Name() string {
	arch := p.Hardware.ArchFlag
	if arch == "" {
		arch = "dynamic-cuda"
	}
	return fmt.Sprintf("Deterministic Roofline & CUDA Template Proposer (%s)", arch)
}

// SupportedOperations lists the built-in LLM/inference operations available for live GPU optimization.
func SupportedOperations() []string {
	return []string{
		"rmsnorm_residual",
		"rmsnorm",
		"layernorm",
		"softmax",
		"swiglu",
	}
}

func (p *DeterministicProposer) ProposeCandidates(opCategory string, batchSize, hiddenDim, iterations int) (harness.KernelSpec, []harness.KernelSpec) {
	if batchSize <= 0 {
		batchSize = 256
	}
	if hiddenDim <= 0 {
		hiddenDim = 4096
	}
	if iterations <= 0 {
		iterations = 120
	}
	passes := p.effectivePasses()
	cat := strings.ToLower(strings.TrimSpace(opCategory))
	switch cat {
	case "rmsnorm":
		return buildRMSNormSpecs(batchSize, hiddenDim, iterations, passes, false)
	case "layernorm":
		return buildLayerNormSpecs(batchSize, hiddenDim, iterations, passes)
	case "softmax":
		return buildSoftmaxSpecs(batchSize, hiddenDim, iterations, passes)
	case "swiglu":
		return buildSwiGLUSpecs(batchSize, hiddenDim, iterations, passes)
	default:
		return buildRMSNormSpecs(batchSize, hiddenDim, iterations, passes, true)
	}
}

// buildRMSNormSpecs constructs the baseline unfused RMSNorm (+ optional residual add) and
// candidate fused/vectorized warp-shuffle kernels across block sizes and vector widths.
func buildRMSNormSpecs(batchSize, hiddenDim, iterations, passes int, withResidual bool) (harness.KernelSpec, []harness.KernelSpec) {
	cat := "rmsnorm"
	title := "RMSNorm"
	if withResidual {
		cat = "rmsnorm_residual"
		title = "RMSNorm + Residual"
	}

	baseDisplay := `// Baseline Unfused CUDA Implementation (Separate Residual Add + Shared-Memory Reduction)
__global__ void residual_add_naive(const float* x, const float* res, float* h, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) h[idx] = x[idx] + res[idx];
}

__global__ void rmsnorm_unfused_naive(const float* h, const float* w, float* out, int cols, float eps) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = h + row * cols;
    float* row_out = out + row * cols;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = row_in[i];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)cols + eps);
    for (int i = tid; i < cols; i += blockDim.x) {
        row_out[i] = row_in[i] * inv_rms * w[i];
    }
}`

	baseSpec := harness.KernelSpec{
		ID:            cat + "_baseline_unfused",
		Name:          fmt.Sprintf("Baseline Unfused %s (Shared-Mem Tree, Block 256)", title),
		OpCategory:    cat,
		Variant:       "baseline_unfused",
		BlockSize:     256,
		VecWidth:      1,
		Fused:         false,
		CUDASource:    generateRMSNormHarnessCU(batchSize, hiddenDim, iterations, passes, 256, 1, false, withResidual),
		DisplayKernel: baseDisplay,
		BatchSize:     batchSize,
		HiddenDim:     hiddenDim,
		Iterations:    iterations,
	}

	configs := []struct {
		block int
		vec   int
		fused bool
		label string
	}{
		{128, 1, true, "Fused Scalar + Warp Shuffle (Block 128)"},
		{256, 1, true, "Fused Scalar + Warp Shuffle (Block 256)"},
		{128, 4, true, "Fused float4 Vectorized + Warp Shuffle (Block 128)"},
		{256, 4, true, "Fused float4 Vectorized + Warp Shuffle (Block 256)"},
		{512, 4, true, "Fused float4 Vectorized + Warp Shuffle (Block 512)"},
	}

	candidates := make([]harness.KernelSpec, 0, len(configs))
	for _, cfg := range configs {
		optDisplay := fmt.Sprintf(`// Optimized Fused %s Kernel (Block=%d, VecWidth=float%d, Warp-Shuffle Reduction)
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void rmsnorm_fused_opt(
    const float* __restrict__ x,
    const float* __restrict__ res,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int cols, float eps
) {
    __shared__ float warp_sums[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* rx = x + row * cols;
    const float* rr = res + row * cols;
    float* rout = out + row * cols;

    float sum_sq = 0.0f;
    // Vectorized 128-bit float%d coalesced loads with safe tail handling for cols %% 4 != 0
    int vec_cols = cols / %d;
    for (int i = tid; i < vec_cols; i += %d) {
        // Fused residual addition + sum-of-squares accumulation in registers
        // Eliminates intermediate global memory tensor read/write
    }
    sum_sq = warp_reduce_sum(sum_sq);
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();
    if (tid < 32) {
        float bsum = (tid < (%d >> 5)) ? warp_sums[tid] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        if (tid == 0) warp_sums[0] = rsqrtf(bsum / (float)cols + eps);
    }
    __syncthreads();
    float inv_rms = warp_sums[0];
    // Coalesced float%d vectorized output write
}`, title, cfg.block, cfg.vec, cfg.vec, cfg.vec, cfg.block, cfg.block, cfg.vec)

		id := fmt.Sprintf("%s_b%d_v%d_fused", cat, cfg.block, cfg.vec)
		candidates = append(candidates, harness.KernelSpec{
			ID:            id,
			Name:          fmt.Sprintf("%s — %s", title, cfg.label),
			OpCategory:    cat,
			Variant:       fmt.Sprintf("block_%d_vec_%d", cfg.block, cfg.vec),
			BlockSize:     cfg.block,
			VecWidth:      cfg.vec,
			Fused:         cfg.fused,
			CUDASource:    generateRMSNormHarnessCU(batchSize, hiddenDim, iterations, passes, cfg.block, cfg.vec, cfg.fused, withResidual),
			DisplayKernel: optDisplay,
			BatchSize:     batchSize,
			HiddenDim:     hiddenDim,
			Iterations:    iterations,
		})
	}

	return baseSpec, candidates
}

func generateRMSNormHarnessCU(batchSize, hiddenDim, iterations, passes, blockSize, vecWidth int, fused, withResidual bool) string {
	resFlag := 0
	if withResidual {
		resFlag = 1
	}
	fusedFlag := 0
	if fused {
		fusedFlag = 1
	}
	return fmt.Sprintf(`#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define BLOCK_SIZE %d
#define VEC_WIDTH %d
#define IS_FUSED %d
#define WITH_RESIDUAL %d

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void residual_add_kernel(const float* x, const float* res, float* h, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        h[idx] = x[idx] + res[idx];
    }
}

__global__ void rmsnorm_unfused_kernel(const float* h, const float* w, float* out, int cols, float eps) {
    __shared__ float sdata[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = h + row * cols;
    float* row_out = out + row * cols;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float v = row_in[i];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)cols + eps);
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        row_out[i] = row_in[i] * inv_rms * w[i];
    }
}

__global__ void rmsnorm_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ res,
    const float* __restrict__ w,
    float* __restrict__ out,
    int cols,
    float eps
) {
    __shared__ float warp_sums[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* rx = x + row * cols;
    const float* rr = res + row * cols;
    float* rout = out + row * cols;

    float sum_sq = 0.0f;
#if VEC_WIDTH == 4
    int vec_cols = ((cols & 3) == 0) ? (cols / 4) : 0;
    const float4* rx4 = reinterpret_cast<const float4*>(rx);
    const float4* rr4 = reinterpret_cast<const float4*>(rr);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 vx = rx4[i];
#if WITH_RESIDUAL
        float4 vr = rr4[i];
        vx.x += vr.x; vx.y += vr.y; vx.z += vr.z; vx.w += vr.w;
#endif
        sum_sq += vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w;
    }
    for (int i = vec_cols * 4 + tid; i < cols; i += BLOCK_SIZE) {
        float v = rx[i];
#if WITH_RESIDUAL
        v += rr[i];
#endif
        sum_sq += v * v;
    }
#else
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float v = rx[i];
#if WITH_RESIDUAL
        v += rr[i];
#endif
        sum_sq += v * v;
    }
#endif

    sum_sq = warp_reduce_sum(sum_sq);
    int lane = tid & 31;
    int wid = tid >> 5;
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();

    int num_warps = BLOCK_SIZE >> 5;
    if (tid < 32) {
        float bsum = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        if (tid == 0) warp_sums[0] = rsqrtf(bsum / (float)cols + eps);
    }
    __syncthreads();
    float inv_rms = warp_sums[0];

#if VEC_WIDTH == 4
    const float4* rw4 = reinterpret_cast<const float4*>(w);
    float4* rout4 = reinterpret_cast<float4*>(rout);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 vx = rx4[i];
#if WITH_RESIDUAL
        float4 vr = rr4[i];
        vx.x += vr.x; vx.y += vr.y; vx.z += vr.z; vx.w += vr.w;
#endif
        float4 vw = rw4[i];
        float4 vo;
        vo.x = vx.x * inv_rms * vw.x;
        vo.y = vx.y * inv_rms * vw.y;
        vo.z = vx.z * inv_rms * vw.z;
        vo.w = vx.w * inv_rms * vw.w;
        rout4[i] = vo;
    }
    for (int i = vec_cols * 4 + tid; i < cols; i += BLOCK_SIZE) {
        float v = rx[i];
#if WITH_RESIDUAL
        v += rr[i];
#endif
        rout[i] = v * inv_rms * w[i];
    }
#else
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float v = rx[i];
#if WITH_RESIDUAL
        v += rr[i];
#endif
        rout[i] = v * inv_rms * w[i];
    }
#endif
}

void run_candidate(const float* d_x, const float* d_res, const float* d_w, float* d_tmp, float* d_out, int rows, int cols, float eps) {
#if IS_FUSED
    rmsnorm_fused_kernel<<<rows, BLOCK_SIZE>>>(d_x, d_res, d_w, d_out, cols, eps);
#else
    int total = rows * cols;
#if WITH_RESIDUAL
    residual_add_kernel<<<(total + 255) / 256, 256>>>(d_x, d_res, d_tmp, total);
    residual_add_kernel<<<(total + 255) / 256, 256>>>(d_x, d_res, d_tmp, total);
    rmsnorm_unfused_kernel<<<rows, BLOCK_SIZE>>>(d_tmp, d_w, d_out, cols, eps);
#else
    cudaMemcpyAsync(d_tmp, d_x, total * sizeof(float), cudaMemcpyDeviceToDevice);
    rmsnorm_unfused_kernel<<<rows, BLOCK_SIZE>>>(d_tmp, d_w, d_out, cols, eps);
#endif
#endif
}

bool verify_shape(int rows, int cols, float* max_err_out) {
    int total = rows * cols;
    float *h_x = (float*)malloc(total * sizeof(float));
    float *h_res = (float*)malloc(total * sizeof(float));
    float *h_w = (float*)malloc(cols * sizeof(float));
    float *h_out = (float*)malloc(total * sizeof(float));

    for (int i = 0; i < total; i++) {
        h_x[i] = 0.2f * sinf((float)(i %% 97)) + 0.5f;
        h_res[i] = 0.15f * cosf((float)(i %% 53)) - 0.1f;
    }
    for (int i = 0; i < cols; i++) {
        h_w[i] = 0.8f + 0.1f * sinf((float)(i %% 31));
    }

    float *d_x, *d_res, *d_w, *d_tmp, *d_out;
    cudaMalloc(&d_x, total * sizeof(float));
    cudaMalloc(&d_res, total * sizeof(float));
    cudaMalloc(&d_w, cols * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));

    cudaMemcpy(d_x, h_x, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, cols * sizeof(float), cudaMemcpyHostToDevice);

    run_candidate(d_x, d_res, d_w, d_tmp, d_out, rows, cols, 1e-5f);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        double ss = 0.0;
        for (int c = 0; c < cols; c++) {
            double v = h_x[r * cols + c];
#if WITH_RESIDUAL
            v += h_res[r * cols + c];
#endif
            ss += v * v;
        }
        double inv = 1.0 / sqrt(ss / (double)cols + 1e-5);
        for (int c = 0; c < cols; c++) {
            double v = h_x[r * cols + c];
#if WITH_RESIDUAL
            v += h_res[r * cols + c];
#endif
            float ref = (float)(v * inv * (double)h_w[c]);
            float err = fabsf(h_out[r * cols + c] - ref);
            if (isnan(err) || isinf(err)) err = 999.0f;
            if (err > max_err) max_err = err;
        }
    }

    cudaFree(d_x); cudaFree(d_res); cudaFree(d_w); cudaFree(d_tmp); cudaFree(d_out);
    free(h_x); free(h_res); free(h_w); free(h_out);
    if (max_err_out) *max_err_out = max_err;
    return max_err < 1e-3f;
}

int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int main(int argc, char** argv) {
    float max_err_main = 0.0f, max_err_edge = 0.0f;
    bool edge_ok = verify_shape(5, 1027, &max_err_edge);
    if (argc > 1 && strcmp(argv[1], "--verify-only") == 0) {
        return edge_ok ? 0 : 2;
    }
    bool parity_ok = verify_shape(16, %d, &max_err_main);
    if (max_err_edge > max_err_main) max_err_main = max_err_edge;

    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    cudaGetDeviceProperties(&prop, 0);
    size_t flush_bytes = (prop.l2CacheSize > 0) ? (size_t)prop.l2CacheSize * 2 : (size_t)(8 * 1024 * 1024);

    size_t free_before_ctx = 0, total_mem = 0, free_after_ctx = 0;
    cudaMemGetInfo(&free_before_ctx, &total_mem);
    cudaFree(0);
    cudaMemGetInfo(&free_after_ctx, &total_mem);
    double ctx_mib = (free_before_ctx > free_after_ctx) ? (double)(free_before_ctx - free_after_ctx) / (1024.0 * 1024.0) : 0.0;

    int rows = %d;
    int cols = %d;
    int iters = %d;
    int passes_per_req = %d;
    size_t total = (size_t)rows * (size_t)cols;

    float *d_x, *d_res, *d_w, *d_tmp, *d_out, *d_l2flush;
    cudaMalloc(&d_x, total * sizeof(float));
    cudaMalloc(&d_res, total * sizeof(float));
    cudaMalloc(&d_w, cols * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_l2flush, flush_bytes);

    cudaMemset(d_x, 1, total * sizeof(float));
    cudaMemset(d_res, 1, total * sizeof(float));
    cudaMemset(d_w, 1, cols * sizeof(float));

    for (int i = 0; i < 10; i++) {
        run_candidate(d_x, d_res, d_w, d_tmp, d_out, rows, cols, 1e-5f);
    }
    cudaDeviceSynchronize();

    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < iters; i++) {
        cudaMemsetAsync(d_l2flush, i & 255, flush_bytes);
        cudaEventRecord(start);
        for (int p = 0; p < passes_per_req; p++) {
            run_candidate(d_x, d_res, d_w, d_tmp, d_out, rows, cols, 1e-5f);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }

    qsort(times, iters, sizeof(float), cmp_float);
    float p50 = times[iters / 2];
    float p95 = times[(int)(iters * 0.95)];
    float pmin = times[0];

    double bytes_per_pass = (double)total * sizeof(float) * (WITH_RESIDUAL ? 3.0 : 2.0);
    double total_bytes = bytes_per_pass * passes_per_req;
    double bw_gbps = (total_bytes / ((p50 / 1000.0))) / 1e9;

    double active_buffers = IS_FUSED ? (WITH_RESIDUAL ? 3.0 : 2.0) : (WITH_RESIDUAL ? 5.0 : 3.5);
    double vram_mib = (active_buffers * (double)total * sizeof(float)) / (1024.0 * 1024.0) + ctx_mib;

    printf("{\"parityPassed\":%%s,\"maxAbsError\":%%.7f,\"edgeCasePassed\":%%s,\"medianLatencyMs\":%%.4f,\"p95LatencyMs\":%%.4f,\"minLatencyMs\":%%.4f,\"effectiveBwGbps\":%%.2f,\"completedRequests\":%%d,\"peakVramMib\":%%.2f}\n",
        parity_ok ? "true" : "false", max_err_main, edge_ok ? "true" : "false",
        p50, p95, pmin, bw_gbps, iters, vram_mib);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_res); cudaFree(d_w); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_l2flush);
    free(times);
    return 0;
}
`, blockSize, vecWidth, fusedFlag, resFlag, hiddenDim, batchSize, hiddenDim, iterations, passes)
}

func buildLayerNormSpecs(batchSize, hiddenDim, iterations, passes int) (harness.KernelSpec, []harness.KernelSpec) {
	baseDisplay := `// Baseline Unfused LayerNorm (2-Pass Mean & Variance + Global Memory Intermediate)
__global__ void layernorm_mean_sub_naive(const float* x, float* centered, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* rx = x + row * cols;
    float* rc = centered + row * cols;
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += rx[i];
    sdata[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float mean = sdata[0] / (float)cols;
    for (int i = tid; i < cols; i += blockDim.x) rc[i] = rx[i] - mean;
}
__global__ void layernorm_var_scale_naive(const float* centered, const float* w, const float* b, float* out, int cols, float eps) { ... }`

	baseSpec := harness.KernelSpec{
		ID:            "layernorm_baseline_unfused",
		Name:          "Baseline Unfused LayerNorm (2-Pass Shared-Mem Mean & Var)",
		OpCategory:    "layernorm",
		Variant:       "baseline_unfused",
		BlockSize:     256,
		VecWidth:      1,
		Fused:         false,
		CUDASource:    generateLayerNormHarnessCU(batchSize, hiddenDim, iterations, passes, 256, 1, false),
		DisplayKernel: baseDisplay,
		BatchSize:     batchSize,
		HiddenDim:     hiddenDim,
		Iterations:    iterations,
	}

	configs := []struct {
		block int
		vec   int
		label string
	}{
		{128, 1, "Fused Single-Pass Scalar + Warp Shuffle (Block 128)"},
		{256, 1, "Fused Single-Pass Scalar + Warp Shuffle (Block 256)"},
		{128, 4, "Fused float4 Vectorized + Warp Shuffle (Block 128)"},
		{256, 4, "Fused float4 Vectorized + Warp Shuffle (Block 256)"},
		{512, 4, "Fused float4 Vectorized + Warp Shuffle (Block 512)"},
	}

	candidates := make([]harness.KernelSpec, 0, len(configs))
	for _, cfg := range configs {
		optDisplay := fmt.Sprintf(`// Optimized Fused Single-Pass LayerNorm (Block=%d, VecWidth=float%d, Dual Warp-Shuffle)
__global__ void layernorm_fused_opt(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ out, int cols, float eps) {
    // Single-pass E[x] and E[x^2] via vectorized float%d loads + __shfl_xor_sync reductions
    // Eliminates intermediate centered tensor allocation and global memory writeback
}`, cfg.block, cfg.vec, cfg.vec)

		candidates = append(candidates, harness.KernelSpec{
			ID:            fmt.Sprintf("layernorm_b%d_v%d_fused", cfg.block, cfg.vec),
			Name:          fmt.Sprintf("LayerNorm — %s", cfg.label),
			OpCategory:    "layernorm",
			Variant:       fmt.Sprintf("block_%d_vec_%d", cfg.block, cfg.vec),
			BlockSize:     cfg.block,
			VecWidth:      cfg.vec,
			Fused:         true,
			CUDASource:    generateLayerNormHarnessCU(batchSize, hiddenDim, iterations, passes, cfg.block, cfg.vec, true),
			DisplayKernel: optDisplay,
			BatchSize:     batchSize,
			HiddenDim:     hiddenDim,
			Iterations:    iterations,
		})
	}
	return baseSpec, candidates
}

func generateLayerNormHarnessCU(batchSize, hiddenDim, iterations, passes, blockSize, vecWidth int, fused bool) string {
	fusedFlag := 0
	if fused {
		fusedFlag = 1
	}
	return fmt.Sprintf(`#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define BLOCK_SIZE %d
#define VEC_WIDTH %d
#define IS_FUSED %d

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layernorm_pass1_mean(const float* x, float* centered, int cols) {
    __shared__ float sdata[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* rx = x + row * cols;
    float* rc = centered + row * cols;

    float sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) sum += rx[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)cols;
    for (int i = tid; i < cols; i += BLOCK_SIZE) rc[i] = rx[i] - mean;
}

__global__ void layernorm_pass2_var(const float* centered, const float* w, const float* b, float* out, int cols, float eps) {
    __shared__ float sdata[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* rc = centered + row * cols;
    float* rout = out + row * cols;

    float sq = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float v = rc[i];
        sq += v * v;
    }
    sdata[tid] = sq;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / (float)cols + eps);
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        rout[i] = rc[i] * inv_std * w[i] + b[i];
    }
}

__global__ void layernorm_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int cols,
    float eps
) {
    __shared__ float warp_sum[32];
    __shared__ float warp_sq[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* rx = x + row * cols;
    float* rout = out + row * cols;

    float sum = 0.0f;
    float sum_sq = 0.0f;
#if VEC_WIDTH == 4
    int vec_cols = ((cols & 3) == 0) ? (cols / 4) : 0;
    const float4* rx4 = reinterpret_cast<const float4*>(rx);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 vx = rx4[i];
        sum += vx.x + vx.y + vx.z + vx.w;
        sum_sq += vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w;
    }
    for (int i = vec_cols * 4 + tid; i < cols; i += BLOCK_SIZE) {
        float v = rx[i];
        sum += v;
        sum_sq += v * v;
    }
#else
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float v = rx[i];
        sum += v;
        sum_sq += v * v;
    }
#endif

    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);
    int lane = tid & 31;
    int wid = tid >> 5;
    if (lane == 0) {
        warp_sum[wid] = sum;
        warp_sq[wid] = sum_sq;
    }
    __syncthreads();

    int num_warps = BLOCK_SIZE >> 5;
    if (tid < 32) {
        float bs = (tid < num_warps) ? warp_sum[tid] : 0.0f;
        float bq = (tid < num_warps) ? warp_sq[tid] : 0.0f;
        bs = warp_reduce_sum(bs);
        bq = warp_reduce_sum(bq);
        if (tid == 0) {
            float mean = bs / (float)cols;
            float var = fmaxf(0.0f, (bq / (float)cols) - mean * mean);
            warp_sum[0] = mean;
            warp_sq[0] = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    float mean = warp_sum[0];
    float inv_std = warp_sq[0];

#if VEC_WIDTH == 4
    const float4* rw4 = reinterpret_cast<const float4*>(w);
    const float4* rb4 = reinterpret_cast<const float4*>(b);
    float4* rout4 = reinterpret_cast<float4*>(rout);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 vx = rx4[i];
        float4 vw = rw4[i];
        float4 vb = rb4[i];
        float4 vo;
        vo.x = (vx.x - mean) * inv_std * vw.x + vb.x;
        vo.y = (vx.y - mean) * inv_std * vw.y + vb.y;
        vo.z = (vx.z - mean) * inv_std * vw.z + vb.z;
        vo.w = (vx.w - mean) * inv_std * vw.w + vb.w;
        rout4[i] = vo;
    }
    for (int i = vec_cols * 4 + tid; i < cols; i += BLOCK_SIZE) {
        rout[i] = (rx[i] - mean) * inv_std * w[i] + b[i];
    }
#else
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        rout[i] = (rx[i] - mean) * inv_std * w[i] + b[i];
    }
#endif
}

void run_candidate(const float* d_x, const float* d_w, const float* d_b, float* d_tmp, float* d_out, int rows, int cols, float eps) {
#if IS_FUSED
    layernorm_fused_kernel<<<rows, BLOCK_SIZE>>>(d_x, d_w, d_b, d_out, cols, eps);
#else
    layernorm_pass1_mean<<<rows, BLOCK_SIZE>>>(d_x, d_tmp, cols);
    layernorm_pass2_var<<<rows, BLOCK_SIZE>>>(d_tmp, d_w, d_b, d_out, cols, eps);
#endif
}

bool verify_shape(int rows, int cols, float* max_err_out) {
    int total = rows * cols;
    float *h_x = (float*)malloc(total * sizeof(float));
    float *h_w = (float*)malloc(cols * sizeof(float));
    float *h_b = (float*)malloc(cols * sizeof(float));
    float *h_out = (float*)malloc(total * sizeof(float));

    for (int i = 0; i < total; i++) h_x[i] = 0.2f * sinf((float)(i %% 97)) + 0.5f;
    for (int i = 0; i < cols; i++) {
        h_w[i] = 0.8f + 0.1f * sinf((float)(i %% 31));
        h_b[i] = 0.05f * cosf((float)(i %% 19));
    }

    float *d_x, *d_w, *d_b, *d_tmp, *d_out;
    cudaMalloc(&d_x, total * sizeof(float));
    cudaMalloc(&d_w, cols * sizeof(float));
    cudaMalloc(&d_b, cols * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));

    cudaMemcpy(d_x, h_x, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, cols * sizeof(float), cudaMemcpyHostToDevice);

    run_candidate(d_x, d_w, d_b, d_tmp, d_out, rows, cols, 1e-5f);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        double sum = 0.0;
        for (int c = 0; c < cols; c++) sum += h_x[r * cols + c];
        double mean = sum / (double)cols;
        double sq = 0.0;
        for (int c = 0; c < cols; c++) {
            double d = h_x[r * cols + c] - mean;
            sq += d * d;
        }
        double inv = 1.0 / sqrt(sq / (double)cols + 1e-5);
        for (int c = 0; c < cols; c++) {
            float ref = (float)((h_x[r * cols + c] - mean) * inv * (double)h_w[c] + (double)h_b[c]);
            float err = fabsf(h_out[r * cols + c] - ref);
            if (isnan(err) || isinf(err)) err = 999.0f;
            if (err > max_err) max_err = err;
        }
    }

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_tmp); cudaFree(d_out);
    free(h_x); free(h_w); free(h_b); free(h_out);
    if (max_err_out) *max_err_out = max_err;
    return max_err < 1e-3f;
}

int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int main(int argc, char** argv) {
    float max_err_main = 0.0f, max_err_edge = 0.0f;
    bool edge_ok = verify_shape(5, 1027, &max_err_edge);
    if (argc > 1 && strcmp(argv[1], "--verify-only") == 0) {
        return edge_ok ? 0 : 2;
    }
    bool parity_ok = verify_shape(16, %d, &max_err_main);
    if (max_err_edge > max_err_main) max_err_main = max_err_edge;

    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    cudaGetDeviceProperties(&prop, 0);
    size_t flush_bytes = (prop.l2CacheSize > 0) ? (size_t)prop.l2CacheSize * 2 : (size_t)(8 * 1024 * 1024);

    size_t free_before_ctx = 0, total_mem = 0, free_after_ctx = 0;
    cudaMemGetInfo(&free_before_ctx, &total_mem);
    cudaFree(0);
    cudaMemGetInfo(&free_after_ctx, &total_mem);
    double ctx_mib = (free_before_ctx > free_after_ctx) ? (double)(free_before_ctx - free_after_ctx) / (1024.0 * 1024.0) : 0.0;

    int rows = %d;
    int cols = %d;
    int iters = %d;
    int passes_per_req = %d;
    size_t total = (size_t)rows * (size_t)cols;

    float *d_x, *d_w, *d_b, *d_tmp, *d_out, *d_l2flush;
    cudaMalloc(&d_x, total * sizeof(float));
    cudaMalloc(&d_w, cols * sizeof(float));
    cudaMalloc(&d_b, cols * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_l2flush, flush_bytes);

    cudaMemset(d_x, 1, total * sizeof(float));
    cudaMemset(d_w, 1, cols * sizeof(float));
    cudaMemset(d_b, 0, cols * sizeof(float));

    for (int i = 0; i < 10; i++) {
        run_candidate(d_x, d_w, d_b, d_tmp, d_out, rows, cols, 1e-5f);
    }
    cudaDeviceSynchronize();

    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < iters; i++) {
        cudaMemsetAsync(d_l2flush, i & 255, flush_bytes);
        cudaEventRecord(start);
        for (int p = 0; p < passes_per_req; p++) {
            run_candidate(d_x, d_w, d_b, d_tmp, d_out, rows, cols, 1e-5f);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }

    qsort(times, iters, sizeof(float), cmp_float);
    float p50 = times[iters / 2];
    float p95 = times[(int)(iters * 0.95)];
    float pmin = times[0];

    double bytes_per_pass = (double)total * sizeof(float) * (IS_FUSED ? 2.0 : 4.0);
    double total_bytes = bytes_per_pass * passes_per_req;
    double bw_gbps = (total_bytes / ((p50 / 1000.0))) / 1e9;

    double active_buffers = IS_FUSED ? 2.0 : 3.0;
    double vram_mib = (active_buffers * (double)total * sizeof(float)) / (1024.0 * 1024.0) + ctx_mib;

    printf("{\"parityPassed\":%%s,\"maxAbsError\":%%.7f,\"edgeCasePassed\":%%s,\"medianLatencyMs\":%%.4f,\"p95LatencyMs\":%%.4f,\"minLatencyMs\":%%.4f,\"effectiveBwGbps\":%%.2f,\"completedRequests\":%%d,\"peakVramMib\":%%.2f}\n",
        parity_ok ? "true" : "false", max_err_main, edge_ok ? "true" : "false",
        p50, p95, pmin, bw_gbps, iters, vram_mib);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_l2flush);
    free(times);
    return 0;
}
`, blockSize, vecWidth, fusedFlag, hiddenDim, batchSize, hiddenDim, iterations, passes)
}

func buildSoftmaxSpecs(batchSize, hiddenDim, iterations, passes int) (harness.KernelSpec, []harness.KernelSpec) {
	baseDisplay := `// Baseline 3-Pass Naive Softmax (Pass 1: Row Max, Pass 2: Exp + Sum, Pass 3: Normalize)
__global__ void softmax_pass1_max(const float* x, float* row_max, int cols) { ... }
__global__ void softmax_pass2_exp(const float* x, const float* row_max, float* exp_buf, float* row_sum, int cols) { ... }
__global__ void softmax_pass3_div(const float* exp_buf, const float* row_sum, float* out, int cols) { ... }`

	baseSpec := harness.KernelSpec{
		ID:            "softmax_baseline_3pass",
		Name:          "Baseline 3-Pass Naive Softmax (Max -> Exp -> Div)",
		OpCategory:    "softmax",
		Variant:       "baseline_unfused",
		BlockSize:     256,
		VecWidth:      1,
		Fused:         false,
		CUDASource:    generateSoftmaxHarnessCU(batchSize, hiddenDim, iterations, passes, 256, 1, false),
		DisplayKernel: baseDisplay,
		BatchSize:     batchSize,
		HiddenDim:     hiddenDim,
		Iterations:    iterations,
	}

	configs := []struct {
		block int
		vec   int
		label string
	}{
		{128, 1, "Fused Online Scalar + Warp Shuffle (Block 128)"},
		{256, 1, "Fused Online Scalar + Warp Shuffle (Block 256)"},
		{128, 4, "Fused Online float4 Vectorized + Warp Shuffle (Block 128)"},
		{256, 4, "Fused Online float4 Vectorized + Warp Shuffle (Block 256)"},
		{512, 4, "Fused Online float4 Vectorized + Warp Shuffle (Block 512)"},
	}

	candidates := make([]harness.KernelSpec, 0, len(configs))
	for _, cfg := range configs {
		optDisplay := fmt.Sprintf(`// Optimized Fused Online Softmax Kernel (Block=%d, VecWidth=float%d, Warp-Shuffle)
__global__ void softmax_fused_opt(const float* __restrict__ x, float* __restrict__ out, int cols) {
    // Online max + exp sum tracking in registers using __shfl_xor_sync reductions
    // Vectorized float%d coalesced loads/stores with zero intermediate exp buffer
}`, cfg.block, cfg.vec, cfg.vec)

		candidates = append(candidates, harness.KernelSpec{
			ID:            fmt.Sprintf("softmax_b%d_v%d_fused", cfg.block, cfg.vec),
			Name:          fmt.Sprintf("Online Softmax — %s", cfg.label),
			OpCategory:    "softmax",
			Variant:       fmt.Sprintf("block_%d_vec_%d", cfg.block, cfg.vec),
			BlockSize:     cfg.block,
			VecWidth:      cfg.vec,
			Fused:         true,
			CUDASource:    generateSoftmaxHarnessCU(batchSize, hiddenDim, iterations, passes, cfg.block, cfg.vec, true),
			DisplayKernel: optDisplay,
			BatchSize:     batchSize,
			HiddenDim:     hiddenDim,
			Iterations:    iterations,
		})
	}
	return baseSpec, candidates
}

func generateSoftmaxHarnessCU(batchSize, hiddenDim, iterations, passes, blockSize, vecWidth int, fused bool) string {
	fusedFlag := 0
	if fused {
		fusedFlag = 1
	}
	return fmt.Sprintf(`#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define BLOCK_SIZE %d
#define VEC_WIDTH %d
#define IS_FUSED %d

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_pass1_max(const float* x, float* row_max, int cols) {
    __shared__ float sdata[BLOCK_SIZE];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* rx = x + row * cols;
    float m = -FLT_MAX;
    for (int i = tid; i < cols; i += BLOCK_SIZE) m = fmaxf(m, rx[i]);
    sdata[tid] = m;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) row_max[row] = sdata[0];
}

__global__ void softmax_pass2_exp(const float* x, const float* row_max, float* exp_buf, float* row_sum, int cols) {
    __shared__ float sdata[BLOCK_SIZE];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* rx = x + row * cols;
    float* re = exp_buf + row * cols;
    float m = row_max[row];
    float sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float ev = expf(rx[i] - m);
        re[i] = ev;
        sum += ev;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) row_sum[row] = sdata[0];
}

__global__ void softmax_pass3_div(const float* exp_buf, const float* row_sum, float* out, int cols) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float* re = exp_buf + row * cols;
    float* rout = out + row * cols;
    float inv_sum = 1.0f / row_sum[row];
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        rout[i] = re[i] * inv_sum;
    }
}

__global__ void softmax_fused_kernel(const float* __restrict__ x, float* __restrict__ out, int cols) {
    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* rx = x + row * cols;
    float* rout = out + row * cols;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
#if VEC_WIDTH == 4
    int vec_cols = ((cols & 3) == 0) ? (cols / 4) : 0;
    const float4* rx4 = reinterpret_cast<const float4*>(rx);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 vx = rx4[i];
        float m_old = local_max;
        float m_new = fmaxf(m_old, fmaxf(fmaxf(vx.x, vx.y), fmaxf(vx.z, vx.w)));
        local_sum = local_sum * expf(m_old - m_new) +
                    expf(vx.x - m_new) + expf(vx.y - m_new) + expf(vx.z - m_new) + expf(vx.w - m_new);
        local_max = m_new;
    }
    for (int i = vec_cols * 4 + tid; i < cols; i += BLOCK_SIZE) {
        float v = rx[i];
        float m_new = fmaxf(local_max, v);
        local_sum = local_sum * expf(local_max - m_new) + expf(v - m_new);
        local_max = m_new;
    }
#else
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float v = rx[i];
        float m_new = fmaxf(local_max, v);
        local_sum = local_sum * expf(local_max - m_new) + expf(v - m_new);
        local_max = m_new;
    }
#endif

    float w_max = warp_reduce_max(local_max);
    float w_sum = warp_reduce_sum(local_sum * expf(local_max - w_max));
    int lane = tid & 31;
    int wid = tid >> 5;
    if (lane == 0) {
        warp_max[wid] = w_max;
        warp_sum[wid] = w_sum;
    }
    __syncthreads();

    int num_warps = BLOCK_SIZE >> 5;
    if (tid < 32) {
        float bm = (tid < num_warps) ? warp_max[tid] : -FLT_MAX;
        float global_m = warp_reduce_max(bm);
        float bs = (tid < num_warps) ? warp_sum[tid] * expf(bm - global_m) : 0.0f;
        float global_s = warp_reduce_sum(bs);
        if (tid == 0) {
            warp_max[0] = global_m;
            warp_sum[0] = 1.0f / global_s;
        }
    }
    __syncthreads();

    float row_m = warp_max[0];
    float inv_s = warp_sum[0];

#if VEC_WIDTH == 4
    float4* rout4 = reinterpret_cast<float4*>(rout);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 vx = rx4[i];
        float4 vo;
        vo.x = expf(vx.x - row_m) * inv_s;
        vo.y = expf(vx.y - row_m) * inv_s;
        vo.z = expf(vx.z - row_m) * inv_s;
        vo.w = expf(vx.w - row_m) * inv_s;
        rout4[i] = vo;
    }
    for (int i = vec_cols * 4 + tid; i < cols; i += BLOCK_SIZE) {
        rout[i] = expf(rx[i] - row_m) * inv_s;
    }
#else
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        rout[i] = expf(rx[i] - row_m) * inv_s;
    }
#endif
}

void run_candidate(const float* d_x, float* d_tmp, float* d_row_max, float* d_row_sum, float* d_out, int rows, int cols) {
#if IS_FUSED
    softmax_fused_kernel<<<rows, BLOCK_SIZE>>>(d_x, d_out, cols);
#else
    softmax_pass1_max<<<rows, BLOCK_SIZE>>>(d_x, d_row_max, cols);
    softmax_pass2_exp<<<rows, BLOCK_SIZE>>>(d_x, d_row_max, d_tmp, d_row_sum, cols);
    softmax_pass3_div<<<rows, BLOCK_SIZE>>>(d_tmp, d_row_sum, d_out, cols);
#endif
}

bool verify_shape(int rows, int cols, float* max_err_out) {
    int total = rows * cols;
    float *h_x = (float*)malloc(total * sizeof(float));
    float *h_out = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) h_x[i] = 1.5f * sinf((float)(i %% 79));

    float *d_x, *d_tmp, *d_row_max, *d_row_sum, *d_out;
    cudaMalloc(&d_x, total * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_row_max, rows * sizeof(float));
    cudaMalloc(&d_row_sum, rows * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMemcpy(d_x, h_x, total * sizeof(float), cudaMemcpyHostToDevice);

    run_candidate(d_x, d_tmp, d_row_max, d_row_sum, d_out, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        double m = -1e30;
        for (int c = 0; c < cols; c++) {
            double v = h_x[r * cols + c];
            if (v > m) m = v;
        }
        double s = 0.0;
        for (int c = 0; c < cols; c++) s += exp(h_x[r * cols + c] - m);
        for (int c = 0; c < cols; c++) {
            float ref = (float)(exp(h_x[r * cols + c] - m) / s);
            float err = fabsf(h_out[r * cols + c] - ref);
            if (isnan(err) || isinf(err)) err = 999.0f;
            if (err > max_err) max_err = err;
        }
    }

    cudaFree(d_x); cudaFree(d_tmp); cudaFree(d_row_max); cudaFree(d_row_sum); cudaFree(d_out);
    free(h_x); free(h_out);
    if (max_err_out) *max_err_out = max_err;
    return max_err < 1e-3f;
}

int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int main(int argc, char** argv) {
    float max_err_main = 0.0f, max_err_edge = 0.0f;
    bool edge_ok = verify_shape(5, 1027, &max_err_edge);
    if (argc > 1 && strcmp(argv[1], "--verify-only") == 0) {
        return edge_ok ? 0 : 2;
    }
    bool parity_ok = verify_shape(16, %d, &max_err_main);
    if (max_err_edge > max_err_main) max_err_main = max_err_edge;

    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    cudaGetDeviceProperties(&prop, 0);
    size_t flush_bytes = (prop.l2CacheSize > 0) ? (size_t)prop.l2CacheSize * 2 : (size_t)(8 * 1024 * 1024);

    size_t free_before_ctx = 0, total_mem = 0, free_after_ctx = 0;
    cudaMemGetInfo(&free_before_ctx, &total_mem);
    cudaFree(0);
    cudaMemGetInfo(&free_after_ctx, &total_mem);
    double ctx_mib = (free_before_ctx > free_after_ctx) ? (double)(free_before_ctx - free_after_ctx) / (1024.0 * 1024.0) : 0.0;

    int rows = %d;
    int cols = %d;
    int iters = %d;
    int passes_per_req = %d;
    size_t total = (size_t)rows * (size_t)cols;

    float *d_x, *d_tmp, *d_row_max, *d_row_sum, *d_out, *d_l2flush;
    cudaMalloc(&d_x, total * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_row_max, rows * sizeof(float));
    cudaMalloc(&d_row_sum, rows * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_l2flush, flush_bytes);
    cudaMemset(d_x, 0, total * sizeof(float));

    for (int i = 0; i < 10; i++) {
        run_candidate(d_x, d_tmp, d_row_max, d_row_sum, d_out, rows, cols);
    }
    cudaDeviceSynchronize();

    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < iters; i++) {
        cudaMemsetAsync(d_l2flush, i & 255, flush_bytes);
        cudaEventRecord(start);
        for (int p = 0; p < passes_per_req; p++) {
            run_candidate(d_x, d_tmp, d_row_max, d_row_sum, d_out, rows, cols);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }

    qsort(times, iters, sizeof(float), cmp_float);
    float p50 = times[iters / 2];
    float p95 = times[(int)(iters * 0.95)];
    float pmin = times[0];

    double bytes_per_pass = (double)total * sizeof(float) * (IS_FUSED ? 2.0 : 5.0);
    double total_bytes = bytes_per_pass * passes_per_req;
    double bw_gbps = (total_bytes / ((p50 / 1000.0))) / 1e9;

    double active_buffers = IS_FUSED ? 2.0 : 3.0;
    double vram_mib = (active_buffers * (double)total * sizeof(float)) / (1024.0 * 1024.0) + ctx_mib;

    printf("{\"parityPassed\":%%s,\"maxAbsError\":%%.7f,\"edgeCasePassed\":%%s,\"medianLatencyMs\":%%.4f,\"p95LatencyMs\":%%.4f,\"minLatencyMs\":%%.4f,\"effectiveBwGbps\":%%.2f,\"completedRequests\":%%d,\"peakVramMib\":%%.2f}\n",
        parity_ok ? "true" : "false", max_err_main, edge_ok ? "true" : "false",
        p50, p95, pmin, bw_gbps, iters, vram_mib);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_tmp); cudaFree(d_row_max); cudaFree(d_row_sum); cudaFree(d_out); cudaFree(d_l2flush);
    free(times);
    return 0;
}
`, blockSize, vecWidth, fusedFlag, hiddenDim, batchSize, hiddenDim, iterations, passes)
}

func buildSwiGLUSpecs(batchSize, hiddenDim, iterations, passes int) (harness.KernelSpec, []harness.KernelSpec) {
	baseDisplay := `// Baseline 2-Pass Unfused SwiGLU (Pass 1: SiLU(gate) -> tmp, Pass 2: tmp * up -> out)
__global__ void silu_naive(const float* gate, float* tmp, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) { float g = gate[idx]; tmp[idx] = g / (1.0f + expf(-g)); }
}
__global__ void gate_mul_naive(const float* tmp, const float* up, float* out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = tmp[idx] * up[idx];
}`

	baseSpec := harness.KernelSpec{
		ID:            "swiglu_baseline_unfused",
		Name:          "Baseline 2-Pass Unfused SwiGLU (SiLU + Gate Multiply)",
		OpCategory:    "swiglu",
		Variant:       "baseline_unfused",
		BlockSize:     256,
		VecWidth:      1,
		Fused:         false,
		CUDASource:    generateSwiGLUHarnessCU(batchSize, hiddenDim, iterations, passes, 256, 1, false),
		DisplayKernel: baseDisplay,
		BatchSize:     batchSize,
		HiddenDim:     hiddenDim,
		Iterations:    iterations,
	}

	configs := []struct {
		block int
		vec   int
		label string
	}{
		{128, 1, "Fused Scalar (Block 128)"},
		{256, 1, "Fused Scalar (Block 256)"},
		{128, 4, "Fused float4 Vectorized (Block 128)"},
		{256, 4, "Fused float4 Vectorized (Block 256)"},
		{512, 4, "Fused float4 Vectorized (Block 512)"},
	}

	candidates := make([]harness.KernelSpec, 0, len(configs))
	for _, cfg := range configs {
		optDisplay := fmt.Sprintf(`// Optimized Fused SwiGLU Kernel (Block=%d, VecWidth=float%d)
__global__ void swiglu_fused_opt(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int cols) {
    // Coalesced float%d loads of gate & up, register SiLU(gate) * up, and coalesced store
}`, cfg.block, cfg.vec, cfg.vec)

		candidates = append(candidates, harness.KernelSpec{
			ID:            fmt.Sprintf("swiglu_b%d_v%d_fused", cfg.block, cfg.vec),
			Name:          fmt.Sprintf("Fused SwiGLU — %s", cfg.label),
			OpCategory:    "swiglu",
			Variant:       fmt.Sprintf("block_%d_vec_%d", cfg.block, cfg.vec),
			BlockSize:     cfg.block,
			VecWidth:      cfg.vec,
			Fused:         true,
			CUDASource:    generateSwiGLUHarnessCU(batchSize, hiddenDim, iterations, passes, cfg.block, cfg.vec, true),
			DisplayKernel: optDisplay,
			BatchSize:     batchSize,
			HiddenDim:     hiddenDim,
			Iterations:    iterations,
		})
	}
	return baseSpec, candidates
}

func generateSwiGLUHarnessCU(batchSize, hiddenDim, iterations, passes, blockSize, vecWidth int, fused bool) string {
	fusedFlag := 0
	if fused {
		fusedFlag = 1
	}
	return fmt.Sprintf(`#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define BLOCK_SIZE %d
#define VEC_WIDTH %d
#define IS_FUSED %d

__global__ void silu_unfused_kernel(const float* gate, float* tmp, int total) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < total) {
        float g = gate[idx];
        tmp[idx] = g / (1.0f + expf(-g));
    }
}

__global__ void gate_mul_unfused_kernel(const float* tmp, const float* up, float* out, int total) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < total) {
        out[idx] = tmp[idx] * up[idx];
    }
}

__global__ void swiglu_fused_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int cols) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float* rg = gate + row * cols;
    const float* ru = up + row * cols;
    float* ro = out + row * cols;

#if VEC_WIDTH == 4
    int vec_cols = ((cols & 3) == 0) ? (cols / 4) : 0;
    const float4* rg4 = reinterpret_cast<const float4*>(rg);
    const float4* ru4 = reinterpret_cast<const float4*>(ru);
    float4* ro4 = reinterpret_cast<float4*>(ro);
    for (int i = tid; i < vec_cols; i += BLOCK_SIZE) {
        float4 vg = rg4[i];
        float4 vu = ru4[i];
        float4 vo;
        vo.x = (vg.x / (1.0f + expf(-vg.x))) * vu.x;
        vo.y = (vg.y / (1.0f + expf(-vg.y))) * vu.y;
        vo.z = (vg.z / (1.0f + expf(-vg.z))) * vu.z;
        vo.w = (vg.w / (1.0f + expf(-vg.w))) * vu.w;
        ro4[i] = vo;
    }
    for (int i = vec_cols * 4 + tid; i < cols; i += BLOCK_SIZE) {
        float g = rg[i];
        ro[i] = (g / (1.0f + expf(-g))) * ru[i];
    }
#else
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float g = rg[i];
        ro[i] = (g / (1.0f + expf(-g))) * ru[i];
    }
#endif
}

void run_candidate(const float* d_gate, const float* d_up, float* d_tmp, float* d_out, int rows, int cols) {
#if IS_FUSED
    swiglu_fused_kernel<<<rows, BLOCK_SIZE>>>(d_gate, d_up, d_out, cols);
#else
    int total = rows * cols;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_unfused_kernel<<<grid, BLOCK_SIZE>>>(d_gate, d_tmp, total);
    gate_mul_unfused_kernel<<<grid, BLOCK_SIZE>>>(d_tmp, d_up, d_out, total);
#endif
}

bool verify_shape(int rows, int cols, float* max_err_out) {
    int total = rows * cols;
    float *h_gate = (float*)malloc(total * sizeof(float));
    float *h_up = (float*)malloc(total * sizeof(float));
    float *h_out = (float*)malloc(total * sizeof(float));

    for (int i = 0; i < total; i++) {
        h_gate[i] = 1.2f * sinf((float)(i %% 73));
        h_up[i] = 0.9f * cosf((float)(i %% 59));
    }

    float *d_gate, *d_up, *d_tmp, *d_out;
    cudaMalloc(&d_gate, total * sizeof(float));
    cudaMalloc(&d_up, total * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));

    cudaMemcpy(d_gate, h_gate, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up, total * sizeof(float), cudaMemcpyHostToDevice);

    run_candidate(d_gate, d_up, d_tmp, d_out, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        double g = h_gate[i];
        float ref = (float)((g / (1.0 + exp(-g))) * (double)h_up[i]);
        float err = fabsf(h_out[i] - ref);
        if (isnan(err) || isinf(err)) err = 999.0f;
        if (err > max_err) max_err = err;
    }

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_tmp); cudaFree(d_out);
    free(h_gate); free(h_up); free(h_out);
    if (max_err_out) *max_err_out = max_err;
    return max_err < 1e-3f;
}

int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int main(int argc, char** argv) {
    float max_err_main = 0.0f, max_err_edge = 0.0f;
    bool edge_ok = verify_shape(5, 1027, &max_err_edge);
    if (argc > 1 && strcmp(argv[1], "--verify-only") == 0) {
        return edge_ok ? 0 : 2;
    }
    bool parity_ok = verify_shape(16, %d, &max_err_main);
    if (max_err_edge > max_err_main) max_err_main = max_err_edge;

    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    cudaGetDeviceProperties(&prop, 0);
    size_t flush_bytes = (prop.l2CacheSize > 0) ? (size_t)prop.l2CacheSize * 2 : (size_t)(8 * 1024 * 1024);

    size_t free_before_ctx = 0, total_mem = 0, free_after_ctx = 0;
    cudaMemGetInfo(&free_before_ctx, &total_mem);
    cudaFree(0);
    cudaMemGetInfo(&free_after_ctx, &total_mem);
    double ctx_mib = (free_before_ctx > free_after_ctx) ? (double)(free_before_ctx - free_after_ctx) / (1024.0 * 1024.0) : 0.0;

    int rows = %d;
    int cols = %d;
    int iters = %d;
    int passes_per_req = %d;
    size_t total = (size_t)rows * (size_t)cols;

    float *d_gate, *d_up, *d_tmp, *d_out, *d_l2flush;
    cudaMalloc(&d_gate, total * sizeof(float));
    cudaMalloc(&d_up, total * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_l2flush, flush_bytes);
    cudaMemset(d_gate, 0, total * sizeof(float));
    cudaMemset(d_up, 0, total * sizeof(float));

    for (int i = 0; i < 10; i++) {
        run_candidate(d_gate, d_up, d_tmp, d_out, rows, cols);
    }
    cudaDeviceSynchronize();

    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < iters; i++) {
        cudaMemsetAsync(d_l2flush, i & 255, flush_bytes);
        cudaEventRecord(start);
        for (int p = 0; p < passes_per_req; p++) {
            run_candidate(d_gate, d_up, d_tmp, d_out, rows, cols);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }

    qsort(times, iters, sizeof(float), cmp_float);
    float p50 = times[iters / 2];
    float p95 = times[(int)(iters * 0.95)];
    float pmin = times[0];

    double bytes_per_pass = (double)total * sizeof(float) * (IS_FUSED ? 3.0 : 5.0);
    double total_bytes = bytes_per_pass * passes_per_req;
    double bw_gbps = (total_bytes / ((p50 / 1000.0))) / 1e9;

    double active_buffers = IS_FUSED ? 3.0 : 4.0;
    double vram_mib = (active_buffers * (double)total * sizeof(float)) / (1024.0 * 1024.0) + ctx_mib;

    printf("{\"parityPassed\":%%s,\"maxAbsError\":%%.7f,\"edgeCasePassed\":%%s,\"medianLatencyMs\":%%.4f,\"p95LatencyMs\":%%.4f,\"minLatencyMs\":%%.4f,\"effectiveBwGbps\":%%.2f,\"completedRequests\":%%d,\"peakVramMib\":%%.2f}\n",
        parity_ok ? "true" : "false", max_err_main, edge_ok ? "true" : "false",
        p50, p95, pmin, bw_gbps, iters, vram_mib);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_l2flush);
    free(times);
    return 0;
}
`, blockSize, vecWidth, fusedFlag, hiddenDim, batchSize, hiddenDim, iterations, passes)
}
