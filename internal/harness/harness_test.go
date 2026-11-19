package harness

import (
	"context"
	"os/exec"
	"testing"
	"time"

	"gpumode/internal/config"
)

func TestParsePTXASOutput(t *testing.T) {
	sampleLog := `ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z13rmsnorm_fusedPKfS0_S0_Pfii' for 'sm_89'
ptxas info    : Function properties for _Z13rmsnorm_fusedPKfS0_S0_Pfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, 1024 bytes smem, 384 bytes cmem[0]`

	stats := ParsePTXASOutput(sampleLog)
	if stats.RegistersPerThread != 32 {
		t.Fatalf("expected 32 registers, got %d", stats.RegistersPerThread)
	}
	if stats.SharedMemBytes != 1024 {
		t.Fatalf("expected 1024 shared mem bytes, got %d", stats.SharedMemBytes)
	}
	if stats.ConstantMemBytes != 384 {
		t.Fatalf("expected 384 constant mem bytes, got %d", stats.ConstantMemBytes)
	}
}

func TestLiveNVCCCompileAndExecuteOnCUDA(t *testing.T) {
	nvccPath := config.ResolveNVCCPath()
	if _, err := exec.LookPath(nvccPath); err != nil {
		t.Skipf("nvcc not available (%v); skipping live CUDA test", err)
	}
	sanitizerPath := config.ResolveSanitizerPath()

	src := `#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void test_kern(const float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = a[idx] * 2.0f + 1.0f;
}

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        fprintf(stderr, "no CUDA device available\n");
        return 1;
    }
    int n = 1027;
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMemset(d_a, 0, n * sizeof(float));
    test_kern<<<(n+255)/256, 256>>>(d_a, d_b, n);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    printf("{\"parityPassed\":true,\"maxAbsError\":0.0,\"edgeCasePassed\":true,\"medianLatencyMs\":0.12,\"p95LatencyMs\":0.14,\"minLatencyMs\":0.11,\"effectiveBwGbps\":145.0,\"completedRequests\":100,\"peakVramMib\":64.0}\n");
    return 0;
}
`
	r := NewRunner(nvccPath, sanitizerPath, "native", 2, 20*time.Second, nil)
	r.RunSanitizer = false // Fast unit test; sanitizer is tested in integration

	results := r.EvaluateBatch(context.Background(), []KernelSpec{
		{
			ID:         "unit_test_kern",
			Name:       "Unit Test Kernel",
			OpCategory: "rmsnorm",
			BlockSize:  256,
			VecWidth:   1,
			CUDASource: src,
			Iterations: 100,
		},
	})

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	res := results[0]
	if !res.CompiledOK {
		t.Fatalf("expected CompiledOK=true, got err: %s", res.ErrorMessage)
	}
	if !res.ParityPassed || !res.EdgeCasePassed {
		t.Fatalf("expected parity and edge-case pass, got %+v", res)
	}
	if res.PTXAS.RegistersPerThread <= 0 {
		t.Fatalf("expected >0 registers parsed from ptxas, got %d (log: %s)", res.PTXAS.RegistersPerThread, res.CompilerLog)
	}
}
