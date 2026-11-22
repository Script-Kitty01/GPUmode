package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/ebpftrace"
	"gpumode/internal/models"
	"gpumode/internal/optimizer"
	"gpumode/internal/physics"
	"gpumode/internal/profiler"
	"gpumode/internal/scheduler"
	"gpumode/internal/store"
	"gpumode/internal/telemetry"
	"gpumode/internal/workload"
)

// Server is the real-time Go daemon serving REST endpoints, live SSE telemetry streams, and the Web Dashboard.
type Server struct {
	mu                sync.RWMutex
	cfg               config.DaemonConfig
	hardware          physics.GPUHardwareSpec
	monitor           *telemetry.Monitor
	pipeline          *profiler.Pipeline
	store             *store.Store
	ebpfStatus        ebpftrace.Summary
	runningOp         bool
	customModelDir    string
	directModelPath   string
	baselineByModel   map[string]workload.WorkloadBenchmarkReport
	optimizedByModel  map[string]workload.WorkloadBenchmarkReport
	lastBaseline      *workload.WorkloadBenchmarkReport
	lastOptimized     *workload.WorkloadBenchmarkReport
}

func NewServer(
	cfg config.DaemonConfig,
	hw physics.GPUHardwareSpec,
	mon *telemetry.Monitor,
	pipe *profiler.Pipeline,
	st *store.Store,
) *Server {
	return &Server{
		cfg:              cfg,
		hardware:         hw,
		monitor:          mon,
		pipeline:         pipe,
		store:            st,
		ebpfStatus:       ebpftrace.ProbeAvailability(ebpftrace.ParseMode(cfg.EBPFMode)),
		baselineByModel:  make(map[string]workload.WorkloadBenchmarkReport),
		optimizedByModel: make(map[string]workload.WorkloadBenchmarkReport),
	}
}

// Handler returns the HTTP router for the daemon.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleDashboard)
	mux.HandleFunc("/metrics", s.handlePrometheusMetrics)
	mux.HandleFunc("/api/state", s.handleState)
	mux.HandleFunc("/api/models", s.handleModels)
	mux.HandleFunc("/api/telemetry/history", s.handleTelemetryHistory)
	mux.HandleFunc("/api/stream/telemetry", s.handleTelemetrySSE)
	mux.HandleFunc("/api/trace/perfetto", s.handlePerfettoTrace)
	mux.HandleFunc("/api/optimize", s.handleOptimize)
	mux.HandleFunc("/api/recompute", s.handleRecompute)
	mux.HandleFunc("/api/workload", s.handleWorkload)
	mux.HandleFunc("/api/schedule", s.handleSchedule)
	return mux
}

func (s *Server) handleSchedule(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	hw := s.hardware
	econ := s.cfg.Economics
	optWL := s.lastOptimized
	baseWL := s.lastBaseline
	s.mu.RUnlock()

	var prof scheduler.ModelDeploymentProfile
	if optWL != nil {
		prof = scheduler.ModelDeploymentProfile{
			ModelName:            optWL.ModelName,
			MeasuredPeakVRAMMiB:  optWL.MeasuredPeakVRAMMiB,
			MeasuredTPOTMS:       optWL.MedianTPOTMS,
			MeasuredTokensPerSec: optWL.ThroughputTokensPerS,
			MeasuredPowerWatts:   optWL.Energy.AvgPowerWatts,
			JoulesPer1KTokens:    optWL.JoulesPer1KTokens,
		}
	} else if baseWL != nil {
		prof = scheduler.ModelDeploymentProfile{
			ModelName:            baseWL.ModelName,
			MeasuredPeakVRAMMiB:  baseWL.MeasuredPeakVRAMMiB,
			MeasuredTPOTMS:       baseWL.MedianTPOTMS,
			MeasuredTokensPerSec: baseWL.ThroughputTokensPerS,
			MeasuredPowerWatts:   baseWL.Energy.AvgPowerWatts,
			JoulesPer1KTokens:    baseWL.JoulesPer1KTokens,
		}
	}
	if r.Method == http.MethodPost && r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&prof)
	}
	rep := scheduler.BuildFleetScheduleReport(hw, prof, econ)
	writeJSON(w, http.StatusOK, rep)
}

func (s *Server) handlePrometheusMetrics(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	hw := s.hardware
	ebpfSum := s.ebpfStatus
	s.mu.RUnlock()

	if latest, ok := s.store.Latest(); ok && latest.HostCUDATrace.Active {
		ebpfSum = latest.HostCUDATrace
	}

	var sample telemetry.Sample
	backend := "NVIDIA_SMI"
	if s.monitor != nil {
		backend = s.monitor.Backend()
		sample = s.monitor.Latest()
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	fmt.Fprintf(w, "# HELP gpumode_nvml_power_watts Current GPU board power draw in Watts.\n")
	fmt.Fprintf(w, "# TYPE gpumode_nvml_power_watts gauge\n")
	fmt.Fprintf(w, "gpumode_nvml_power_watts{gpu=%q,arch=%q,backend=%q} %.2f\n", hw.Name, hw.ArchFlag, backend, sample.PowerWatts)
	fmt.Fprintf(w, "# HELP gpumode_nvml_sm_clock_mhz Current SM clock frequency in MHz.\n")
	fmt.Fprintf(w, "# TYPE gpumode_nvml_sm_clock_mhz gauge\n")
	fmt.Fprintf(w, "gpumode_nvml_sm_clock_mhz{gpu=%q} %.0f\n", hw.Name, sample.SMClockMHz)
	fmt.Fprintf(w, "# HELP gpumode_nvml_mem_clock_mhz Current memory clock frequency in MHz.\n")
	fmt.Fprintf(w, "# TYPE gpumode_nvml_mem_clock_mhz gauge\n")
	fmt.Fprintf(w, "gpumode_nvml_mem_clock_mhz{gpu=%q} %.0f\n", hw.Name, sample.MemClockMHz)
	fmt.Fprintf(w, "# HELP gpumode_nvml_temperature_celsius Current GPU core temperature in Celsius.\n")
	fmt.Fprintf(w, "# TYPE gpumode_nvml_temperature_celsius gauge\n")
	fmt.Fprintf(w, "gpumode_nvml_temperature_celsius{gpu=%q} %.1f\n", hw.Name, sample.TemperatureC)
	fmt.Fprintf(w, "# HELP gpumode_nvml_vram_used_mib Current used GPU VRAM in MiB.\n")
	fmt.Fprintf(w, "# TYPE gpumode_nvml_vram_used_mib gauge\n")
	fmt.Fprintf(w, "gpumode_nvml_vram_used_mib{gpu=%q} %.1f\n", hw.Name, sample.VRAMUsedMiB)
	fmt.Fprintf(w, "# HELP gpumode_nvml_throttle_reasons_bitmask Active NVML clock throttle reason bitmask.\n")
	fmt.Fprintf(w, "# TYPE gpumode_nvml_throttle_reasons_bitmask gauge\n")
	fmt.Fprintf(w, "gpumode_nvml_throttle_reasons_bitmask{gpu=%q} %d\n", hw.Name, sample.ThrottleReasonsBitmask)
	fmt.Fprintf(w, "# HELP gpumode_hardware_empirical_dram_gbps Empirical sustained DRAM STREAM Triad bandwidth in GB/s.\n")
	fmt.Fprintf(w, "# TYPE gpumode_hardware_empirical_dram_gbps gauge\n")
	fmt.Fprintf(w, "gpumode_hardware_empirical_dram_gbps{gpu=%q} %.2f\n", hw.Name, hw.EmpiricalDRAMBandwidthGBps)
	fmt.Fprintf(w, "# HELP gpumode_hardware_empirical_l2_gbps Empirical sustained L2 cache bandwidth in GB/s.\n")
	fmt.Fprintf(w, "# TYPE gpumode_hardware_empirical_l2_gbps gauge\n")
	fmt.Fprintf(w, "gpumode_hardware_empirical_l2_gbps{gpu=%q} %.2f\n", hw.Name, hw.EmpiricalL2BandwidthGBps)
	fmt.Fprintf(w, "# HELP gpumode_ebpf_kernel_launches_total Total CUDA kernel launches captured in latest eBPF window.\n")
	fmt.Fprintf(w, "# TYPE gpumode_ebpf_kernel_launches_total gauge\n")
	fmt.Fprintf(w, "gpumode_ebpf_kernel_launches_total{provenance=%q} %d\n", ebpfSum.Provenance, ebpfSum.KernelLaunches)
	fmt.Fprintf(w, "# HELP gpumode_ebpf_launch_gap_avg_us Average inter-launch host dispatch gap in microseconds.\n")
	fmt.Fprintf(w, "# TYPE gpumode_ebpf_launch_gap_avg_us gauge\n")
	fmt.Fprintf(w, "gpumode_ebpf_launch_gap_avg_us %.2f\n", ebpfSum.AvgLaunchGapUS)
	fmt.Fprintf(w, "# HELP gpumode_ebpf_launch_gap_p95_us p95 inter-launch host dispatch gap in microseconds.\n")
	fmt.Fprintf(w, "# TYPE gpumode_ebpf_launch_gap_p95_us gauge\n")
	fmt.Fprintf(w, "gpumode_ebpf_launch_gap_p95_us %.2f\n", ebpfSum.P95LaunchGapUS)
	fmt.Fprintf(w, "# HELP gpumode_ebpf_sync_stall_ms_total Total host CUDA stream/ctx synchronize stall time in ms.\n")
	fmt.Fprintf(w, "# TYPE gpumode_ebpf_sync_stall_ms_total gauge\n")
	fmt.Fprintf(w, "gpumode_ebpf_sync_stall_ms_total %.2f\n", ebpfSum.SyncStallMS)
	fmt.Fprintf(w, "# HELP gpumode_ebpf_off_cpu_ms_total Total Linux scheduler off-CPU starvation time in ms.\n")
	fmt.Fprintf(w, "# TYPE gpumode_ebpf_off_cpu_ms_total gauge\n")
	fmt.Fprintf(w, "gpumode_ebpf_off_cpu_ms_total %.2f\n", ebpfSum.OffCPUMS)
}

func (s *Server) handlePerfettoTrace(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	ebpfSum := s.ebpfStatus
	if s.lastOptimized != nil && s.lastOptimized.HostCUDATrace.Active {
		ebpfSum = s.lastOptimized.HostCUDATrace
	} else if s.lastBaseline != nil && s.lastBaseline.HostCUDATrace.Active {
		ebpfSum = s.lastBaseline.HostCUDATrace
	}
	s.mu.RUnlock()

	if !ebpfSum.Active || ebpfSum.KernelLaunches == 0 {
		if latest, ok := s.store.Latest(); ok && latest.HostCUDATrace.KernelLaunches > 0 {
			ebpfSum = latest.HostCUDATrace
		}
	}

	traceBytes, err := ebpftrace.BuildPerfettoTrace(ebpfSum)
	if err != nil {
		http.Error(w, "failed to build Perfetto trace: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="gpumode_ebpf_trace.json"`)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(traceBytes)
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
	customDir := s.customModelDir
	directPath := s.directModelPath
	var baseCopy *workload.WorkloadBenchmarkReport
	if s.lastBaseline != nil {
		cp := *s.lastBaseline
		baseCopy = &cp
	}
	var optCopy *workload.WorkloadBenchmarkReport
	if s.lastOptimized != nil {
		cp := *s.lastOptimized
		optCopy = &cp
	}
	comparisons := make(map[string]map[string]workload.WorkloadBenchmarkReport)
	for k, v := range s.baselineByModel {
		if comparisons[k] == nil {
			comparisons[k] = make(map[string]workload.WorkloadBenchmarkReport)
		}
		comparisons[k]["baseline"] = v
	}
	for k, v := range s.optimizedByModel {
		if comparisons[k] == nil {
			comparisons[k] = make(map[string]workload.WorkloadBenchmarkReport)
		}
		comparisons[k]["optimized"] = v
	}
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

	discCtx, discCancel := context.WithTimeout(r.Context(), 1200*time.Millisecond)
	discovered := models.DiscoverLocalModels(discCtx, models.DiscoverOptions{
		CustomDir:    customDir,
		ProbeTimeout: 450 * time.Millisecond,
	})
	discCancel()

	resp := map[string]any{
		"hardware":           hw,
		"economics":          econ,
		"telemetryBackend":   telemetryBackend,
		"ebpfStatus":         ebpfStatus,
		"supportedOps":       optimizer.SupportedOperations(),
		"hasReport":          hasLatest,
		"latestReport":       latest,
		"reportHistory":      history,
		"latestTelemetry":    latestSample,
		"telemetryHistory":   samples,
		"busy":               busy,
		"customModelDir":     customDir,
		"directModelPath":    directPath,
		"discoveredModels":   discovered,
		"baselineWorkload":   baseCopy,
		"optimizedWorkload":  optCopy,
		"comparisonsByModel": comparisons,
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	q := r.URL.Query()
	var customDir string
	if q.Has("dir") {
		customDir = strings.TrimSpace(q.Get("dir"))
		s.mu.Lock()
		s.customModelDir = customDir
		s.mu.Unlock()
	} else {
		s.mu.RLock()
		customDir = s.customModelDir
		s.mu.RUnlock()
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	res := models.DiscoverLocalModels(ctx, models.DiscoverOptions{
		CustomDir:    customDir,
		OpenAIURL:    strings.TrimSpace(q.Get("llmUrl")),
		OllamaURL:    strings.TrimSpace(q.Get("ollamaUrl")),
		ProbeTimeout: 750 * time.Millisecond,
	})
	writeJSON(w, http.StatusOK, res)
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

type workloadRequest struct {
	workload.BenchmarkConfig
	DirectModelPath string `json:"directModelPath,omitempty"`
	CustomDir       string `json:"customDir,omitempty"`
	OpCategory      string `json:"opCategory,omitempty"`
	BatchSize       int    `json:"batchSize,omitempty"`
	HiddenDim       int    `json:"hiddenDim,omitempty"`
	Iterations      int    `json:"iterations,omitempty"`
	Strategy        string `json:"strategy,omitempty"`
}

func (s *Server) handleWorkload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req workloadRequest
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}
	s.mu.RLock()
	cfgModelPath := s.cfg.DeepSeekModelPath
	cfgEBPFMode := s.cfg.EBPFMode
	cfgTracePID := s.cfg.TracePID
	customDir := s.customModelDir
	s.mu.RUnlock()

	if strings.TrimSpace(req.CustomDir) != "" {
		customDir = strings.TrimSpace(req.CustomDir)
		s.mu.Lock()
		s.customModelDir = customDir
		s.mu.Unlock()
	}

	benchCfg := req.BenchmarkConfig
	def := workload.DefaultBenchmarkConfig()

	modelKey := strings.TrimSpace(benchCfg.ModelID)
	rawDirectPath := strings.TrimSpace(req.DirectModelPath)
	if modelKey == "__direct_path__" && rawDirectPath == "" {
		rawDirectPath = strings.TrimSpace(benchCfg.ModelPath)
	}

	if rawDirectPath != "" || modelKey == "__direct_path__" {
		if rawDirectPath == "" {
			http.Error(w, "direct model path is required when Direct Local Model Path is selected", http.StatusBadRequest)
			return
		}
		resolved, err := models.ResolveDirectModelPath(rawDirectPath)
		if err != nil {
			http.Error(w, "invalid direct model path: "+err.Error(), http.StatusBadRequest)
			return
		}
		s.mu.Lock()
		s.directModelPath = rawDirectPath
		s.mu.Unlock()

		modelKey = resolved.ID
		benchCfg.ModelID = resolved.ID
		benchCfg.ModelName = resolved.Name
		benchCfg.ModelSource = resolved.Source
		benchCfg.ModelPath = resolved.Path
		benchCfg.AutoServer = true
		benchCfg.ForceLocalSpawn = true
	} else if modelKey == "__custom_url__" {
		modelKey = ""
		benchCfg.ModelID = ""
		benchCfg.AutoServer = false
		benchCfg.ForceLocalSpawn = false
	} else if modelKey != "" {
		discCtx, discCancel := context.WithTimeout(r.Context(), 1500*time.Millisecond)
		disc := models.DiscoverLocalModels(discCtx, models.DiscoverOptions{
			CustomDir:    customDir,
			OpenAIURL:    benchCfg.ServerURL,
			ProbeTimeout: 600 * time.Millisecond,
		})
		discCancel()
		for _, m := range disc.Models {
			if m.ID == modelKey {
				benchCfg.ModelName = m.Name
				benchCfg.ModelSource = m.Source
				if m.Path != "" {
					benchCfg.ModelPath = m.Path
					benchCfg.AutoServer = true
					benchCfg.ForceLocalSpawn = true
				} else if m.EndpointURL != "" {
					benchCfg.ServerURL = m.EndpointURL
					benchCfg.AutoServer = false
					benchCfg.ForceLocalSpawn = false
				}
				break
			}
		}
	}

	if benchCfg.ServerURL == "" {
		benchCfg.ServerURL = def.ServerURL
	}
	if benchCfg.ModelName == "" {
		benchCfg.ModelName = def.ModelName
	}
	if benchCfg.ModelPath == "" && benchCfg.AutoServer {
		if cfgModelPath != "" {
			benchCfg.ModelPath = cfgModelPath
		} else {
			benchCfg.ModelPath = def.ModelPath
		}
	}
	if benchCfg.NumRequests <= 0 {
		benchCfg.NumRequests = def.NumRequests
	}
	if benchCfg.MaxTokens <= 0 {
		benchCfg.MaxTokens = def.MaxTokens
	}
	if benchCfg.EBPFMode == "" {
		benchCfg.EBPFMode = ebpftrace.ParseMode(cfgEBPFMode)
	} else {
		benchCfg.EBPFMode = ebpftrace.ParseMode(string(benchCfg.EBPFMode))
	}
	if benchCfg.TracePID == 0 {
		benchCfg.TracePID = cfgTracePID
	}
	if benchCfg.Mode == "" {
		benchCfg.Mode = "baseline"
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

	ctx, cancel := context.WithTimeout(r.Context(), 240*time.Second)
	defer cancel()

	rep, err := workload.RunInferenceBenchmark(ctx, benchCfg, s.monitor, econ)
	if err != nil {
		http.Error(w, "workload benchmark failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	if modelKey == "" {
		if benchCfg.ModelPath != "" {
			modelKey = benchCfg.ModelPath
		} else {
			modelKey = rep.ModelName
		}
	}

	var causalRep *profiler.CausalChainReport
	if rep.RunMode == "optimized" && s.pipeline != nil {
		opCat := req.OpCategory
		if opCat == "" {
			opCat = "rmsnorm_residual"
		}
		strat := optimizer.StrategyBayesian
		if req.Strategy == string(optimizer.StrategyGrid) {
			strat = optimizer.StrategyGrid
		}
		runPipe := *s.pipeline
		if benchCfg.ModelPath != "" {
			if _, lookErr := exec.LookPath("llama-cli"); lookErr == nil {
				dsProposer := optimizer.NewLocalDeepSeekProposerForHardware("llama-cli", benchCfg.ModelPath, s.hardware)
				runPipe.Autotuner = optimizer.NewAutotuner(dsProposer, s.pipeline.Runner)
			}
		}
		cr := runPipe.Execute(ctx, opCat, req.BatchSize, req.HiddenDim, req.Iterations, strat, econ)
		_ = s.store.Save(cr)
		causalRep = &cr
	}

	s.mu.Lock()
	if rep.RunMode == "optimized" {
		s.optimizedByModel[modelKey] = rep
		cp := rep
		s.lastOptimized = &cp
	} else {
		s.baselineByModel[modelKey] = rep
		cp := rep
		s.lastBaseline = &cp
	}
	var pairedBase *workload.WorkloadBenchmarkReport
	if b, ok := s.baselineByModel[modelKey]; ok {
		cp := b
		pairedBase = &cp
	}
	var pairedOpt *workload.WorkloadBenchmarkReport
	if o, ok := s.optimizedByModel[modelKey]; ok {
		cp := o
		pairedOpt = &cp
	}
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]any{
		"mode":         rep.RunMode,
		"modelKey":     modelKey,
		"workload":     rep,
		"baseline":     pairedBase,
		"optimized":    pairedOpt,
		"causalReport": causalRep,
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
