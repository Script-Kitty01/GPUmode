package harness

import (
	"context"
	"os/exec"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/telemetry"
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

func TestExecuteAndMeasureUsesBurstIntervalForUncertainty(t *testing.T) {
	nvccPath := config.ResolveNVCCPath()
	if _, err := exec.LookPath(nvccPath); err != nil {
		t.Skipf("nvcc not available (%v); skipping live CUDA burst telemetry test", err)
	}

	// Use a deterministic sampler on a 200ms monitor so inter-window power CV is 0%.
	// 100 requests * 0.60ms = 60ms kernel time per window:
	// - Against the 200ms background interval, aliasing uncertainty would be 18.6% (MEDIUM confidence).
	// - Against the 50ms burst interval, 60ms >= 50ms yields 3.5% uncertainty (HIGH confidence).
	mon := telemetry.NewMonitor("nvidia-smi", 200*time.Millisecond, 256)
	defer mon.Close()
	mon.SetSampler(func(ctx context.Context) (telemetry.Sample, error) {
		return telemetry.Sample{
			Timestamp:    time.Now(),
			GPUName:      "Deterministic Test GPU",
			PowerWatts:   45.0,
			SMClockMHz:   2400,
			MemClockMHz:  8001,
			TemperatureC: 55,
			VRAMUsedMiB:  512,
			VRAMTotalMiB: 6144,
		}, nil
	})

	src := `#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

__global__ void burst_test_kern(float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = float(idx);
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return 1;
    }
    float* d_b = nullptr;
    cudaMalloc(&d_b, 1024 * sizeof(float));
    burst_test_kern<<<4, 256>>>(d_b, 1024);
    cudaDeviceSynchronize();
    usleep(65000);
    cudaFree(d_b);
    printf("{\"parityPassed\":true,\"maxAbsError\":0.0,\"edgeCasePassed\":true,\"medianLatencyMs\":0.60,\"p95LatencyMs\":0.65,\"minLatencyMs\":0.58,\"effectiveBwGbps\":120.0,\"completedRequests\":100,\"peakVramMib\":64.0}\n");
    return 0;
}
`
	r := NewRunner(nvccPath, "", "native", 1, 20*time.Second, mon)
	r.RunSanitizer = false

	results := r.EvaluateBatch(context.Background(), []KernelSpec{
		{
			ID:         "burst_test_kern",
			Name:       "Burst Telemetry Kernel",
			OpCategory: "rmsnorm",
			BlockSize:  256,
			VecWidth:   1,
			CUDASource: src,
			Iterations: 100,
		},
	})
	if len(results) != 1 || !results[0].CompiledOK {
		t.Fatalf("unexpected batch results: %+v", results)
	}
	em := results[0].Energy
	if em.SampleCount < 4 {
		t.Fatalf("expected >=4 burst samples across 3 windows, got %d", em.SampleCount)
	}
	if em.EnergyUncertaintyPct > 5.0 {
		t.Errorf("expected burst-sampled EnergyUncertaintyPct <= 5.0%% (3.5%% aliasing floor), got %.2f%%", em.EnergyUncertaintyPct)
	}
	if em.MeasurementConfidence != "HIGH" {
		t.Errorf("expected HIGH MeasurementConfidence with burst polling, got %q (unc=%.2f%%, samples=%d)",
			em.MeasurementConfidence, em.EnergyUncertaintyPct, em.SampleCount)
	}
}
