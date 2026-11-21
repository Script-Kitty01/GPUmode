package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/ebpftrace"
	"gpumode/internal/harness"
	"gpumode/internal/physics"
	"gpumode/internal/profiler"
	"gpumode/internal/store"
	"gpumode/internal/telemetry"
)

func TestServerDashboardStateAndRecompute(t *testing.T) {
	ctx := context.Background()
	cfg := config.DefaultDaemonConfig()
	cfg.DataDir = t.TempDir()

	hw, _ := physics.DetectHardware(ctx, cfg.NVCCPath, cfg.NvidiaSMIPath)
	mon := telemetry.NewMonitorWithBackend(cfg.NvidiaSMIPath, cfg.GPUIndex, 200*time.Millisecond, 30, telemetry.ParseBackendMode(cfg.TelemetryBackend))
	defer mon.Close()
	runner := harness.NewRunner(cfg.NVCCPath, cfg.SanitizerPath, hw.ArchFlag, 2, 20*time.Second, mon)
	runner.RunSanitizer = false

	pipe := profiler.NewPipeline(hw, mon, runner, cfg.PythonPath, "../../scripts/profile_pytorch.py")
	pipe.EBPFMode = ebpftrace.ParseMode(cfg.EBPFMode)
	st, err := store.NewStore(cfg.DataDir)
	if err != nil {
		t.Fatalf("NewStore failed: %v", err)
	}

	srv := NewServer(cfg, hw, mon, pipe, st)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 1. GET / dashboard HTML
	res, err := http.Get(ts.URL + "/")
	if err != nil {
		t.Fatalf("GET / failed: %v", err)
	}
	htmlBytes, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		t.Fatalf("failed reading dashboard HTML: %v", err)
	}
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on GET /, got %d", res.StatusCode)
	}
	html := string(htmlBytes)
	for _, marker := range []string{"telemetrySourceBadge", "ebpfProvBadge", "CUDA Boundary (eBPF + Scheduler)", "directModelPathInput", "modelSelect", "runBaselineBtn", "runOptimizedBtn"} {
		if !strings.Contains(html, marker) {
			t.Fatalf("expected dashboard HTML to contain %q", marker)
		}
	}

	// 1a-0. POST /api/workload with missing or invalid directModelPath returns 400 Bad Request
	badDirectRes, err := http.Post(ts.URL+"/api/workload", "application/json", strings.NewReader(`{"modelId":"__direct_path__","directModelPath":"/nonexistent/path/model.gguf","mode":"baseline"}`))
	if err != nil {
		t.Fatalf("POST /api/workload invalid directModelPath failed: %v", err)
	}
	badDirectRes.Body.Close()
	if badDirectRes.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 on nonexistent directModelPath, got %d", badDirectRes.StatusCode)
	}
	emptyDirectRes, err := http.Post(ts.URL+"/api/workload", "application/json", strings.NewReader(`{"modelId":"__direct_path__","mode":"baseline"}`))
	if err != nil {
		t.Fatalf("POST /api/workload empty directModelPath failed: %v", err)
	}
	emptyDirectRes.Body.Close()
	if emptyDirectRes.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 on empty directModelPath when __direct_path__ selected, got %d", emptyDirectRes.StatusCode)
	}

	// 1a. GET /api/models with custom dir
	customDir := t.TempDir()
	_ = os.WriteFile(filepath.Join(customDir, "Sample-Test-Q4_K_M.gguf"), []byte("gguf"), 0o600)
	modelsRes, err := http.Get(ts.URL + "/api/models?dir=" + customDir)
	if err != nil {
		t.Fatalf("GET /api/models failed: %v", err)
	}
	defer modelsRes.Body.Close()
	if modelsRes.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on GET /api/models, got %d", modelsRes.StatusCode)
	}
	var modelsPayload struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(modelsRes.Body).Decode(&modelsPayload); err != nil {
		t.Fatalf("failed decoding /api/models: %v", err)
	}
	foundSample := false
	for _, m := range modelsPayload.Models {
		if m.Name == "Sample-Test-Q4_K_M" {
			foundSample = true
			break
		}
	}
	if !foundSample {
		t.Fatalf("expected /api/models to return Sample-Test-Q4_K_M, got %+v", modelsPayload.Models)
	}

	// 1b. GET /api/state verifies telemetryBackend and cached ebpfStatus
	stateRes, err := http.Get(ts.URL + "/api/state")
	if err != nil {
		t.Fatalf("GET /api/state failed: %v", err)
	}
	defer stateRes.Body.Close()
	if stateRes.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on GET /api/state, got %d", stateRes.StatusCode)
	}
	var statePayload struct {
		TelemetryBackend string            `json:"telemetryBackend"`
		EBPFStatus       ebpftrace.Summary `json:"ebpfStatus"`
	}
	if err := json.NewDecoder(stateRes.Body).Decode(&statePayload); err != nil {
		t.Fatalf("failed decoding /api/state: %v", err)
	}
	if statePayload.TelemetryBackend != "NVML" && statePayload.TelemetryBackend != "NVIDIA_SMI" {
		t.Fatalf("expected telemetryBackend NVML or NVIDIA_SMI, got %q", statePayload.TelemetryBackend)
	}
	if statePayload.EBPFStatus.Provenance == "" {
		t.Fatalf("expected non-empty ebpfStatus.provenance in /api/state")
	}
	if statePayload.EBPFStatus.Mode == "" {
		t.Fatalf("expected non-empty ebpfStatus.mode in /api/state")
	}

	// 2. POST /api/optimize
	optReq := `{"opCategory":"rmsnorm_residual","batchSize":64,"hiddenDim":2048,"iterations":20,"strategy":"BAYESIAN_UCB"}`
	res2, err := http.Post(ts.URL+"/api/optimize", "application/json", strings.NewReader(optReq))
	if err != nil {
		t.Fatalf("POST /api/optimize failed: %v", err)
	}
	defer res2.Body.Close()
	if res2.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on POST /api/optimize, got %d", res2.StatusCode)
	}

	var rep profiler.CausalChainReport
	if err := json.NewDecoder(res2.Body).Decode(&rep); err != nil {
		t.Fatalf("failed to decode optimize report: %v", err)
	}
	if rep.EconomicsComparison.SpeedupFactor <= 1.0 {
		t.Fatalf("expected >1.0x speedup from POST /api/optimize, got %.2f", rep.EconomicsComparison.SpeedupFactor)
	}

	// 3. POST /api/recompute with updated QPS/SLA
	newEcon := cfg.Economics
	newEcon.TargetQPS = 800
	b, _ := json.Marshal(newEcon)
	res3, err := http.Post(ts.URL+"/api/recompute", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatalf("POST /api/recompute failed: %v", err)
	}
	defer res3.Body.Close()
	if res3.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on POST /api/recompute, got %d", res3.StatusCode)
	}

	// 4. POST /api/workload baseline vs optimized against a mock OpenAI streaming server
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path == "/v1/models" {
			fmt.Fprint(w, `{"data":[{"id":"mock-llm"}]}`)
			return
		}
		if r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		http.NotFound(w, r)
	}))
	defer llmSrv.Close()

	baseBody := fmt.Sprintf(`{"serverUrl":%q,"modelName":"mock-llm","autoServer":false,"mode":"baseline","numRequests":1,"maxTokens":8}`, llmSrv.URL)
	baseRes, err := http.Post(ts.URL+"/api/workload", "application/json", strings.NewReader(baseBody))
	if err != nil {
		t.Fatalf("POST /api/workload (baseline) failed: %v", err)
	}
	defer baseRes.Body.Close()
	if baseRes.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on baseline /api/workload, got %d", baseRes.StatusCode)
	}
	var baseResp map[string]any
	if err := json.NewDecoder(baseRes.Body).Decode(&baseResp); err != nil {
		t.Fatalf("decode baseline /api/workload: %v", err)
	}
	if baseResp["mode"] != "baseline" {
		t.Fatalf("expected mode=baseline, got %v", baseResp["mode"])
	}

	optBody := fmt.Sprintf(`{"serverUrl":%q,"modelName":"mock-llm","autoServer":false,"mode":"optimized","numRequests":1,"maxTokens":8,"opCategory":"rmsnorm_residual","batchSize":64,"hiddenDim":2048,"iterations":20}`, llmSrv.URL)
	optWLRes, err := http.Post(ts.URL+"/api/workload", "application/json", strings.NewReader(optBody))
	if err != nil {
		t.Fatalf("POST /api/workload (optimized) failed: %v", err)
	}
	defer optWLRes.Body.Close()
	if optWLRes.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on optimized /api/workload, got %d", optWLRes.StatusCode)
	}
	var optResp map[string]any
	if err := json.NewDecoder(optWLRes.Body).Decode(&optResp); err != nil {
		t.Fatalf("decode optimized /api/workload: %v", err)
	}
	if optResp["mode"] != "optimized" || optResp["baseline"] == nil || optResp["optimized"] == nil || optResp["causalReport"] == nil {
		t.Fatalf("expected mode=optimized, baseline, optimized, and causalReport in response, got keys=%v", optResp)
	}
}
