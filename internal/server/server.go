package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/ebpftrace"
	"gpumode/internal/optimizer"
	"gpumode/internal/physics"
	"gpumode/internal/profiler"
	"gpumode/internal/store"
	"gpumode/internal/telemetry"
	"gpumode/internal/workload"
)

// Server is the real-time Go daemon serving REST endpoints, live SSE telemetry streams, and the Web Dashboard.
type Server struct {
	mu         sync.RWMutex
	cfg        config.DaemonConfig
	hardware   physics.GPUHardwareSpec
	monitor    *telemetry.Monitor
	pipeline   *profiler.Pipeline
	store      *store.Store
	ebpfStatus ebpftrace.Summary
	runningOp  bool
}

func NewServer(
	cfg config.DaemonConfig,
	hw physics.GPUHardwareSpec,
	mon *telemetry.Monitor,
	pipe *profiler.Pipeline,
	st *store.Store,
) *Server {
	return &Server{
		cfg:        cfg,
		hardware:   hw,
		monitor:    mon,
		pipeline:   pipe,
		store:      st,
		ebpfStatus: ebpftrace.ProbeAvailability(ebpftrace.ParseMode(cfg.EBPFMode)),
	}
}

// Handler returns the HTTP router for the daemon.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleDashboard)
	mux.HandleFunc("/api/state", s.handleState)
	mux.HandleFunc("/api/telemetry/history", s.handleTelemetryHistory)
	mux.HandleFunc("/api/stream/telemetry", s.handleTelemetrySSE)
	mux.HandleFunc("/api/optimize", s.handleOptimize)
	mux.HandleFunc("/api/recompute", s.handleRecompute)
	mux.HandleFunc("/api/workload", s.handleWorkload)
	return mux
}

func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(dashboardHTML))
}

func (s *Server) handleState(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	econ := s.cfg.Economics
	hw := s.hardware
	busy := s.runningOp
	ebpfStatus := s.ebpfStatus
	s.mu.RUnlock()

	latest, hasLatest := s.store.Latest()
	history := s.store.List()
	var latestSample telemetry.Sample
	var samples []telemetry.Sample
	telemetryBackend := ""
	if s.monitor != nil {
		telemetryBackend = s.monitor.Backend()
		latestSample = s.monitor.Latest()
		samples = s.monitor.History(60)
	}

	resp := map[string]any{
		"hardware":         hw,
		"economics":        econ,
		"telemetryBackend": telemetryBackend,
		"ebpfStatus":       ebpfStatus,
		"supportedOps":     optimizer.SupportedOperations(),
		"hasReport":        hasLatest,
		"latestReport":     latest,
		"reportHistory":    history,
		"latestTelemetry":  latestSample,
		"telemetryHistory": samples,
		"busy":             busy,
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleTelemetryHistory(w http.ResponseWriter, r *http.Request) {
	var samples []telemetry.Sample
	if s.monitor != nil {
		samples = s.monitor.History(120)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"samples": samples,
	})
}

func (s *Server) handleTelemetrySSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	if s.monitor == nil {
		fmt.Fprintf(w, "data: {}\n\n")
		flusher.Flush()
		return
	}

	// Send immediate snapshot first
	initSample := s.monitor.Latest()
	if b, err := json.Marshal(initSample); err == nil {
		fmt.Fprintf(w, "data: %s\n\n", string(b))
		flusher.Flush()
	}

	ch, unsub := s.monitor.Subscribe()
	defer unsub()

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case sample, open := <-ch:
			if !open {
				return
			}
			b, err := json.Marshal(sample)
			if err != nil {
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", string(b))
			flusher.Flush()
		}
	}
}

type optimizeRequest struct {
	OpCategory string          `json:"opCategory"`
	BatchSize  int             `json:"batchSize"`
	HiddenDim  int             `json:"hiddenDim"`
	Iterations int             `json:"iterations"`
	Strategy   string          `json:"strategy"`
	Economics  *config.EconomicsConfig `json:"economics,omitempty"`
}

func (s *Server) handleOptimize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req optimizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	if s.runningOp {
		s.mu.Unlock()
		http.Error(w, "an optimization job is already running on the GPU", http.StatusConflict)
		return
	}
	s.runningOp = true
	if req.Economics != nil {
		s.cfg.Economics = *req.Economics
	}
	econ := s.cfg.Economics
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		s.runningOp = false
		s.mu.Unlock()
	}()

	strat := optimizer.StrategyBayesian
	if req.Strategy == string(optimizer.StrategyGrid) {
		strat = optimizer.StrategyGrid
	}

	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()

	report := s.pipeline.Execute(ctx, req.OpCategory, req.BatchSize, req.HiddenDim, req.Iterations, strat, econ)
	_ = s.store.Save(report)

	writeJSON(w, http.StatusOK, report)
}

func (s *Server) handleRecompute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var econ config.EconomicsConfig
	if err := json.NewDecoder(r.Body).Decode(&econ); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	s.cfg.Economics = econ
	s.mu.Unlock()

	latest, ok := s.store.Latest()
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"economics": econ})
		return
	}

	updated := profiler.RecomputeEconomics(latest, econ)
	_ = s.store.Save(updated)
	writeJSON(w, http.StatusOK, updated)
}

func (s *Server) handleWorkload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req workload.BenchmarkConfig
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}
	s.mu.RLock()
	cfgModelPath := s.cfg.DeepSeekModelPath
	cfgEBPFMode := s.cfg.EBPFMode
	cfgTracePID := s.cfg.TracePID
	s.mu.RUnlock()

	def := workload.DefaultBenchmarkConfig()
	if req.ServerURL == "" {
		req.ServerURL = def.ServerURL
	}
	if req.ModelName == "" {
		req.ModelName = def.ModelName
	}
	if req.ModelPath == "" {
		if cfgModelPath != "" {
			req.ModelPath = cfgModelPath
		} else {
			req.ModelPath = def.ModelPath
		}
	}
	if req.NumRequests <= 0 {
		req.NumRequests = def.NumRequests
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = def.MaxTokens
	}
	if req.EBPFMode == "" {
		req.EBPFMode = ebpftrace.ParseMode(cfgEBPFMode)
	} else {
		req.EBPFMode = ebpftrace.ParseMode(string(req.EBPFMode))
	}
	if req.TracePID == 0 {
		req.TracePID = cfgTracePID
	}

	s.mu.Lock()
	if s.runningOp {
		s.mu.Unlock()
		http.Error(w, "a GPU benchmark job is already running", http.StatusConflict)
		return
	}
	s.runningOp = true
	econ := s.cfg.Economics
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		s.runningOp = false
		s.mu.Unlock()
	}()

	ctx, cancel := context.WithTimeout(r.Context(), 180*time.Second)
	defer cancel()

	rep, err := workload.RunInferenceBenchmark(ctx, req, s.monitor, econ)
	if err != nil {
		http.Error(w, "workload benchmark failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, rep)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
