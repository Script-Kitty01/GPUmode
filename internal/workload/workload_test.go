package workload

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/ebpftrace"
	"gpumode/internal/optimizer"
	"gpumode/internal/telemetry"
)

func TestResolveGGUFModelPath(t *testing.T) {
	t.Setenv("GPUMODE_MODEL_PATH", "")
	t.Setenv("GPUMODE_MODEL_DIR", "")

	modelDir := t.TempDir()
	modelPath := filepath.Join(modelDir, "Qwen3.5-9B-Q4_K_M.gguf")
	if err := os.WriteFile(modelPath, []byte("test model"), 0o600); err != nil {
		t.Fatalf("failed to write test GGUF: %v", err)
	}
	t.Setenv("GPUMODE_MODEL_DIR", modelDir)
	if got := config.ResolveDeepSeekModelPath(); got != modelPath {
		t.Fatalf("config model discovery = %q, want %q", got, modelPath)
	}
	if got := optimizer.ResolveDefaultModelPath(); got != modelPath {
		t.Fatalf("optimizer model discovery = %q, want %q", got, modelPath)
	}

	overrideDir := t.TempDir()
	overridePath := filepath.Join(overrideDir, "custom.gguf")
	if err := os.WriteFile(overridePath, []byte("override"), 0o600); err != nil {
		t.Fatalf("failed to write override GGUF: %v", err)
	}
	t.Setenv("GPUMODE_MODEL_PATH", overridePath)
	if got := config.ResolveDeepSeekModelPath(); got != overridePath {
		t.Fatalf("configured model override = %q, want %q", got, overridePath)
	}
}

func TestRunInferenceBenchmarkWithMockStreamingServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			tokens := []string{"CUDA", " fused", " kernels", " reduce", " DRAM", " traffic."}
			for _, tok := range tokens {
				time.Sleep(8 * time.Millisecond)
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", tok)
				if flusher != nil {
					flusher.Flush()
				}
			}
			fmt.Fprintf(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	mon := telemetry.NewMonitor("nvidia-smi", 20*time.Millisecond, 100)
	mon.SetSampler(func(ctx context.Context) (telemetry.Sample, error) {
		return telemetry.Sample{
			Timestamp:   time.Now(),
			GPUName:     "RTX 4050 Mock",
			PowerWatts:  42.0,
			GPUUtilPct:  55.0,
			MemUtilPct:  35.0,
			VRAMUsedMiB: 5420.0,
		}, nil
	})
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	mon.Start(ctx)

	cfg := BenchmarkConfig{
		ServerURL:   srv.URL,
		ModelName:   "deepseek-test",
		AutoServer:  false,
		NumRequests: 3,
		MaxTokens:   16,
	}
	rep, err := RunInferenceBenchmark(ctx, cfg, mon, config.DefaultEconomicsConfig())
	if err != nil {
		t.Fatalf("RunInferenceBenchmark failed: %v", err)
	}
	if rep.Provenance != "MEASURED" {
		t.Errorf("expected MEASURED provenance, got %s", rep.Provenance)
	}
	if rep.CompletedRequests != 3 {
		t.Errorf("expected 3 completed requests, got %d", rep.CompletedRequests)
	}
	if rep.TotalOutputTokens != 18 {
		t.Errorf("expected 18 output tokens, got %d", rep.TotalOutputTokens)
	}
	if rep.MedianTTFTMS <= 0 || rep.ThroughputTokensPerS <= 0 {
		t.Errorf("expected positive TTFT and tokens/sec, got ttft=%.2f tps=%.2f", rep.MedianTTFTMS, rep.ThroughputTokensPerS)
	}
	if rep.MeasuredPeakVRAMMiB != 5420.0 {
		t.Errorf("expected MeasuredPeakVRAMMiB=5420, got %.1f", rep.MeasuredPeakVRAMMiB)
	}
	if rep.TelemetryBackend != "CUSTOM" {
		t.Errorf("expected TelemetryBackend=CUSTOM, got %q", rep.TelemetryBackend)
	}
	if rep.HostCUDATrace.Mode != ebpftrace.ModeAuto || !rep.HostCUDATrace.Enabled {
		t.Errorf("expected default ModeAuto enabled HostCUDATrace, got %+v", rep.HostCUDATrace)
	}
	validAutoProv := rep.HostCUDATrace.Provenance == ebpftrace.ProvenanceMeasuredEBPF ||
		rep.HostCUDATrace.Provenance == ebpftrace.ProvenanceUnavailableUnprivileged ||
		rep.HostCUDATrace.Provenance == ebpftrace.ProvenanceUnavailableUnsupported
	if !validAutoProv {
		t.Errorf("unexpected HostCUDATrace provenance in ModeAuto: %q", rep.HostCUDATrace.Provenance)
	}

	// Verify DefaultBenchmarkConfig sets EBPFMode=ModeAuto and explicit ModeOff produces DISABLED provenance.
	if def := DefaultBenchmarkConfig(); def.EBPFMode != ebpftrace.ModeAuto {
		t.Errorf("DefaultBenchmarkConfig().EBPFMode = %q, want %q", def.EBPFMode, ebpftrace.ModeAuto)
	}

	cfgOff := cfg
	cfgOff.NumRequests = 1
	cfgOff.EBPFMode = ebpftrace.ModeOff
	repOff, err := RunInferenceBenchmark(ctx, cfgOff, mon, config.DefaultEconomicsConfig())
	if err != nil {
		t.Fatalf("RunInferenceBenchmark(ModeOff) failed: %v", err)
	}
	if repOff.HostCUDATrace.Provenance != ebpftrace.ProvenanceDisabled || repOff.HostCUDATrace.Enabled || repOff.HostCUDATrace.Active {
		t.Errorf("expected DISABLED HostCUDATrace for ModeOff, got %+v", repOff.HostCUDATrace)
	}
	if repOff.TelemetryBackend != "CUSTOM" {
		t.Errorf("expected TelemetryBackend=CUSTOM for ModeOff, got %q", repOff.TelemetryBackend)
	}
	if rep.RunMode != "baseline" || math.Abs(rep.AvgGPUUtilPct-55.0) > 1e-6 || math.Abs(rep.PeakGPUUtilPct-55.0) > 1e-6 {
		t.Errorf("unexpected baseline runMode/GPU util: mode=%q avg=%.6f peak=%.6f", rep.RunMode, rep.AvgGPUUtilPct, rep.PeakGPUUtilPct)
	}

	// Verify buildLlamaServerArgs distinguishes baseline vs optimized modes.
	tmpGGUF := filepath.Join(t.TempDir(), "test.gguf")
	_ = os.WriteFile(tmpGGUF, make([]byte, 1024*1024), 0o600)
	_, baseFlags, baseProf, baseNGL := buildLlamaServerArgs(BenchmarkConfig{ModelPath: tmpGGUF, CtxSize: 1024}, "baseline", "18081", 6141, 500, 0)
	_, optFlags, optProf, optNGL := buildLlamaServerArgs(BenchmarkConfig{ModelPath: tmpGGUF, CtxSize: 1024}, "optimized", "18082", 6141, 500, 0)
	if baseNGL != 24 || optNGL != 99 {
		t.Errorf("expected baseline ngl=24 and optimized ngl=99 for small GGUF on 6GiB GPU, got %d / %d", baseNGL, optNGL)
	}
	if baseProf == optProf || fmt.Sprint(baseFlags) == fmt.Sprint(optFlags) {
		t.Errorf("expected distinct baseline vs optimized llama-server flags")
	}

	// Verify concurrent multi-client workload execution (Concurrency = 3).
	cfgConc := cfg
	cfgConc.NumRequests = 6
	cfgConc.Concurrency = 3
	cfgConc.EBPFMode = ebpftrace.ModeOff
	repConc, err := RunInferenceBenchmark(ctx, cfgConc, mon, config.DefaultEconomicsConfig())
	if err != nil {
		t.Fatalf("RunInferenceBenchmark(Concurrency=3) failed: %v", err)
	}
	if repConc.CompletedRequests != 6 || repConc.Concurrency != 3 {
		t.Errorf("expected 6 completed requests at concurrency=3, got reqs=%d conc=%d", repConc.CompletedRequests, repConc.Concurrency)
	}
}
