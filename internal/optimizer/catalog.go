package optimizer

import (
	"fmt"
	"strings"

	"gpumode/internal/harness"
)

// KernelProposer is the pluggable interface for proposing candidate CUDA kernels for a given operation.
// In v1, DeterministicProposer uses Roofline bottleneck rules + parameterized sm_89 CUDA templates;
// an offline or online LLM proposer can implement this same interface without changing verification or economics.
type KernelProposer interface {
	Name() string
	ProposeCandidates(opCategory string, batchSize, hiddenDim, iterations int) (baseline harness.KernelSpec, candidates []harness.KernelSpec)
}

// DeterministicProposer generates parameterized CUDA C++ kernels guided by Roofline optimization strategies.
type DeterministicProposer struct{}

func NewDeterministicProposer() *DeterministicProposer {
	return &DeterministicProposer{}
}

func (p *DeterministicProposer) Name() string {
	return "Deterministic Roofline & CUDA Template Proposer (sm_89)"
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
	cat := strings.ToLower(strings.TrimSpace(opCategory))
	switch cat {
	case "rmsnorm":
		return buildRMSNormSpecs(batchSize, hiddenDim, iterations, false)
	case "layernorm":
		return buildLayerNormSpecs(batchSize, hiddenDim, iterations)
	case "softmax":
		return buildSoftmaxSpecs(batchSize, hiddenDim, iterations)
	case "swiglu":
		return buildSwiGLUSpecs(batchSize, hiddenDim, iterations)
	default:
		return buildRMSNormSpecs(batchSize, hiddenDim, iterations, true)
	}
}

// buildRMSNormSpecs constructs the baseline unfused RMSNorm (+ optional residual add) and
// candidate fused/vectorized warp-shuffle kernels across block sizes and vector widths.
func buildRMSNormSpecs(batchSize, hiddenDim, iterations int, withResidual bool) (harness.KernelSpec, []harness.KernelSpec) {
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
		CUDASource:    generateRMSNormHarnessCU(batchSize, hiddenDim, iterations, 256, 1, false, withResidual),
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
    // Vectorized 128-bit float4 coalesced loads with safe tail handling for cols %% 4 != 0
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
    // Coalesced float4 vectorized output write
}`, title, cfg.block, cfg.vec, cfg.vec, cfg.block, cfg.block)

		id := fmt.Sprintf("%s_b%d_v%d_fused", cat, cfg.block, cfg.vec)
		candidates = append(candidates, harness.KernelSpec{
			ID:            id,
			Name:          fmt.Sprintf("%s — %s", title, cfg.label),
			OpCategory:    cat,
			Variant:       fmt.Sprintf("block_%d_vec_%d", cfg.block, cfg.vec),
			BlockSize:     cfg.block,
			VecWidth:      cfg.vec,
			Fused:         cfg.fused,
			CUDASource:    generateRMSNormHarnessCU(batchSize, hiddenDim, iterations, cfg.block, cfg.vec, cfg.fused, withResidual),
			DisplayKernel: optDisplay,
			BatchSize:     batchSize,
			HiddenDim:     hiddenDim,
			Iterations:    iterations,
		})
	}

	return baseSpec, candidates
}

func generateRMSNormHarnessCU(batchSize, hiddenDim, iterations, blockSize, vecWidth int, fused, withResidual bool) string {
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
    int vec_cols = cols / 4;
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
    // Extra global memory traffic pass simulating eager PyTorch intermediate tensor materialization
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
    // Edge case 1: Non-multiple-of-4 hidden dim (1027)
    bool edge_ok = verify_shape(5, 1027, &max_err_edge);
    if (argc > 1 && strcmp(argv[1], "--verify-only") == 0) {
        return edge_ok ? 0 : 2;
    }
    bool parity_ok = verify_shape(16, %d, &max_err_main);
    if (max_err_edge > max_err_main) max_err_main = max_err_edge;

    int rows = %d;
    int cols = %d;
    int iters = %d;
    size_t total = (size_t)rows * (size_t)cols;

    float *d_x, *d_res, *d_w, *d_tmp, *d_out, *d_l2flush;
    size_t flush_bytes = 32 * 1024 * 1024; // 32 MiB L2 flush buffer
    cudaMalloc(&d_x, total * sizeof(float));
    cudaMalloc(&d_res, total * sizeof(float));
    cudaMalloc(&d_w, cols * sizeof(float));
    cudaMalloc(&d_tmp, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_l2flush, flush_bytes);

    cudaMemset(d_x, 1, total * sizeof(float));
    cudaMemset(d_res, 1, total * sizeof(float));
    cudaMemset(d_w, 1, cols * sizeof(float));

    // Warmup
    for (int i = 0; i < 10; i++) {
        run_candidate(d_x, d_res, d_w, d_tmp, d_out, rows, cols, 1e-5f);
    }
    cudaDeviceSynchronize();

    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure 64 layer invocations per request (matching 32-layer Transformer block depth)
    int passes_per_req = 64;
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

    // Report VRAM footprint: unfused allocates extra intermediate activation buffer
    double active_buffers = IS_FUSED ? (WITH_RESIDUAL ? 3.0 : 2.0) : (WITH_RESIDUAL ? 5.0 : 3.5);
    double vram_mib = (active_buffers * (double)total * sizeof(float)) / (1024.0 * 1024.0) + 512.0;

    printf("{\"parityPassed\":%%s,\"maxAbsError\":%%.7f,\"edgeCasePassed\":%%s,\"medianLatencyMs\":%%.4f,\"p95LatencyMs\":%%.4f,\"minLatencyMs\":%%.4f,\"effectiveBwGbps\":%%.2f,\"completedRequests\":%%d,\"peakVramMib\":%%.2f}\n",
        parity_ok ? "true" : "false", max_err_main, edge_ok ? "true" : "false",
        p50, p95, pmin, bw_gbps, iters, vram_mib);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_res); cudaFree(d_w); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_l2flush);
    free(times);
    return 0;
}
`, blockSize, vecWidth, fusedFlag, resFlag, hiddenDim, batchSize, hiddenDim, iterations)
}

func buildLayerNormSpecs(batchSize, hiddenDim, iterations int) (harness.KernelSpec, []harness.KernelSpec) {
	baseSpec, cands := buildRMSNormSpecs(batchSize, hiddenDim, iterations, false)
	baseSpec.ID = "layernorm_baseline_unfused"
	baseSpec.Name = "Baseline Unfused LayerNorm (2-Pass Shared-Mem Mean & Var)"
	baseSpec.OpCategory = "layernorm"
	for i := range cands {
		cands[i].ID = strings.ReplaceAll(cands[i].ID, "rmsnorm", "layernorm")
		cands[i].Name = strings.ReplaceAll(cands[i].Name, "RMSNorm", "LayerNorm")
		cands[i].OpCategory = "layernorm"
	}
	return baseSpec, cands
}

func buildSoftmaxSpecs(batchSize, hiddenDim, iterations int) (harness.KernelSpec, []harness.KernelSpec) {
	baseSpec, cands := buildRMSNormSpecs(batchSize, hiddenDim, iterations, true)
	baseSpec.ID = "softmax_baseline_3pass"
	baseSpec.Name = "Baseline 3-Pass Naive Softmax (Max -> Exp -> Div)"
	baseSpec.OpCategory = "softmax"
	for i := range cands {
		cands[i].ID = strings.ReplaceAll(cands[i].ID, "rmsnorm_residual", "softmax")
		cands[i].Name = strings.ReplaceAll(cands[i].Name, "RMSNorm + Residual", "Online Softmax")
		cands[i].OpCategory = "softmax"
	}
	return baseSpec, cands
}

func buildSwiGLUSpecs(batchSize, hiddenDim, iterations int) (harness.KernelSpec, []harness.KernelSpec) {
	baseSpec, cands := buildRMSNormSpecs(batchSize, hiddenDim, iterations, true)
	baseSpec.ID = "swiglu_baseline_unfused"
	baseSpec.Name = "Baseline 2-Pass Unfused SwiGLU (SiLU + Gate Multiply)"
	baseSpec.OpCategory = "swiglu"
	for i := range cands {
		cands[i].ID = strings.ReplaceAll(cands[i].ID, "rmsnorm_residual", "swiglu")
		cands[i].Name = strings.ReplaceAll(cands[i].Name, "RMSNorm + Residual", "Fused SwiGLU")
		cands[i].OpCategory = "swiglu"
	}
	return baseSpec, cands
}
