package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"gpumode/internal/config"
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
	mon := telemetry.NewMonitor(cfg.NvidiaSMIPath, 200*time.Millisecond, 30)
	runner := harness.NewRunner(cfg.NVCCPath, cfg.SanitizerPath, hw.ArchFlag, 2, 20*time.Second, mon)
	runner.RunSanitizer = false

	pipe := profiler.NewPipeline(hw, mon, runner, cfg.PythonPath, "../../scripts/profile_pytorch.py")
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
	res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 on GET /, got %d", res.StatusCode)
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

	// 3. POST /api/recompute with updated tariff
	newEcon := cfg.Economics
	newEcon.GPUHourlyRateUSD = 4.00
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
}
