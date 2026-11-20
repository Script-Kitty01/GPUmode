package workload

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/ebpftrace"
	"gpumode/internal/optimizer"
	"gpumode/internal/physics"
	"gpumode/internal/telemetry"
)

// BenchmarkConfig configures a real HTTP inference benchmark against a local llama-server or Ollama endpoint.
type BenchmarkConfig struct {
	ServerURL   string         `json:"serverUrl"`   // e.g. http://127.0.0.1:8081
	ModelName   string         `json:"modelName"`   // e.g. deepseek-v4-9b or ollama model tag
	ModelPath   string         `json:"modelPath"`   // GGUF path when AutoStartServer is true
	AutoServer  bool           `json:"autoServer"`  // Launch temporary llama-server if none is running
	GPULayers   int            `json:"gpuLayers"`   // -ngl for llama-server
	CtxSize     int            `json:"ctxSize"`     // -c for llama-server
	NumRequests int            `json:"numRequests"` // Number of real streaming inference requests
	MaxTokens   int            `json:"maxTokens"`   // max_tokens per request
	Prompts     []string       `json:"prompts,omitempty"`
	EBPFMode    ebpftrace.Mode `json:"ebpfMode,omitempty"`
	TracePID    int            `json:"tracePid,omitempty"`
}

// RequestMeasurement holds real streaming timings for a single LLM inference call.
type RequestMeasurement struct {
	RequestIndex   int     `json:"requestIndex"`
	TTFTMS         float64 `json:"ttftMs"`
	TPOTMS         float64 `json:"tpotMs"`
	TotalLatencyMS float64 `json:"totalLatencyMs"`
	OutputTokens   int     `json:"outputTokens"`
	TokensPerSec   float64 `json:"tokensPerSec"`
	Success        bool    `json:"success"`
	Error          string  `json:"error,omitempty"`
}

// WorkloadBenchmarkReport captures end-to-end measured inference, energy, VRAM, and production economics.
type WorkloadBenchmarkReport struct {
	Timestamp              time.Time                   `json:"timestamp"`
	Adapter                string                      `json:"adapter"`
	ServerURL              string                      `json:"serverUrl"`
	ModelName              string                      `json:"modelName"`
	Provenance             string                      `json:"provenance"` // "MEASURED"
	TelemetryBackend       string                      `json:"telemetryBackend,omitempty"`
	HostCUDATrace          ebpftrace.Summary           `json:"hostCudaTrace"`
	LaunchesPerOutputToken float64                     `json:"launchesPerOutputToken,omitempty"`
	CompletedRequests      int                         `json:"completedRequests"`
	TotalOutputTokens      int                         `json:"totalOutputTokens"`
	TotalDurationSec       float64                     `json:"totalDurationSec"`
	MedianTTFTMS           float64                     `json:"medianTtftMs"`
	P95TTFTMS              float64                     `json:"p95TtftMs"`
	MedianTPOTMS           float64                     `json:"medianTpotMs"`
	MedianLatencyMS        float64                     `json:"medianLatencyMs"`
	P95LatencyMS           float64                     `json:"p95LatencyMs"`
	ThroughputTokensPerS   float64                     `json:"throughputTokensPerS"`
	ThroughputRPS          float64                     `json:"throughputRps"`
	MeasuredPeakVRAMMiB    float64                     `json:"measuredPeakVramMib"`
	Energy                 telemetry.EnergyMeasurement `json:"energy"`
	JoulesPer1KTokens      float64                     `json:"joulesPer1kTokens"`
	Economics              physics.BusinessKPIs        `json:"economics"`
	Requests               []RequestMeasurement        `json:"requests"`
}

func DefaultBenchmarkConfig() BenchmarkConfig {
	ngl := 24
	if v := strings.TrimSpace(os.Getenv("GPUMODE_NGL")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			ngl = n
		}
	}
	return BenchmarkConfig{
		ServerURL:   "http://127.0.0.1:8081",
		ModelName:   strings.TrimSpace(os.Getenv("GPUMODE_MODEL_NAME")),
		ModelPath:   optimizer.ResolveDefaultModelPath(),
		AutoServer:  true,
		GPULayers:   ngl,
		CtxSize:     1024,
		NumRequests: 4,
		MaxTokens:   48,
		EBPFMode:    ebpftrace.ModeAuto,
		Prompts: []string{
			"Explain why fused RMSNorm + Residual Add reduces DRAM bandwidth traffic in CUDA in two concise sentences.",
			"Write a short 3-line explanation of warp shuffle reductions in CUDA.",
			"Summarize how Roofline arithmetic intensity distinguishes memory-bound from compute-bound GPU kernels.",
			"List two ways float4 vectorized global memory loads improve GPU occupancy.",
		},
	}
}

// RunInferenceBenchmark executes real streaming chat-completion requests against a local OpenAI-compatible
// inference server (llama-server or Ollama), recording TTFT, TPOT, throughput, live GPU power/energy,
// measured VRAM footprint, and production unit economics.
func RunInferenceBenchmark(
	ctx context.Context,
	cfg BenchmarkConfig,
	mon *telemetry.Monitor,
	econ config.EconomicsConfig,
) (WorkloadBenchmarkReport, error) {
	if cfg.ServerURL == "" {
		cfg.ServerURL = "http://127.0.0.1:8081"
	}
	cfg.ServerURL = strings.TrimRight(cfg.ServerURL, "/")
	if cfg.NumRequests <= 0 {
		cfg.NumRequests = 4
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 48
	}
	if len(cfg.Prompts) == 0 {
		cfg.Prompts = DefaultBenchmarkConfig().Prompts
	}

	var idlePowerW float64
	var vramTotalMiB float64
	if mon != nil {
		idlePowerW = mon.MeasureIdleBaseline(ctx, 2)
		vramTotalMiB = mon.Latest().VRAMTotalMiB
	}

	// Auto-launch llama-server if requested and endpoint is not already reachable
	var serverCmd *exec.Cmd
	if cfg.AutoServer && !isEndpointHealthy(ctx, cfg.ServerURL) {
		if cfg.ModelPath == "" {
			cfg.ModelPath = optimizer.ResolveDefaultModelPath()
		}
		if cfg.ModelPath == "" {
			return WorkloadBenchmarkReport{}, fmt.Errorf(
				"no running inference server at %s and no local GGUF model found; set GPUMODE_MODEL_PATH or GPUMODE_MODEL_DIR, or pass -deepseek-model",
				cfg.ServerURL,
			)
		}
		if _, err := os.Stat(cfg.ModelPath); err != nil {
			return WorkloadBenchmarkReport{}, fmt.Errorf("auto-server requested but GGUF model not found at %s: %w", cfg.ModelPath, err)
		}
		port := extractPort(cfg.ServerURL, "8081")
		ngl := cfg.GPULayers
		if ngl <= 0 {
			if vramTotalMiB >= 12000 {
				ngl = 99
			} else if vramTotalMiB > 0 {
				ngl = int(vramTotalMiB / 256.0)
				if ngl < 8 {
					ngl = 8
				}
			} else {
				ngl = 24
			}
		}
		ctxSz := cfg.CtxSize
		if ctxSz <= 0 {
			ctxSz = 1024
		}
		serverCmd = exec.CommandContext(
			ctx,
			"llama-server",
			"-m", cfg.ModelPath,
			"-ngl", fmt.Sprintf("%d", ngl),
			"-c", fmt.Sprintf("%d", ctxSz),
			"--reasoning", "off",
			"--host", "127.0.0.1",
			"--port", port,
		)
		if err := serverCmd.Start(); err != nil {
			return WorkloadBenchmarkReport{}, fmt.Errorf("failed to start llama-server: %w", err)
		}
		defer func() {
			if serverCmd.Process != nil {
				_ = serverCmd.Process.Kill()
				_, _ = serverCmd.Process.Wait()
			}
		}()

		if err := waitForHealthy(ctx, cfg.ServerURL, 45*time.Second); err != nil {
			return WorkloadBenchmarkReport{}, fmt.Errorf("llama-server did not become healthy: %w", err)
		}
		if cfg.ModelName == "" {
			cfg.ModelName = strings.TrimSuffix(filepath.Base(cfg.ModelPath), filepath.Ext(cfg.ModelPath))
		}
	}

	if cfg.ModelName == "" {
		cfg.ModelName = discoverServedModel(ctx, cfg.ServerURL)
	}
	if cfg.ModelName == "" {
		cfg.ModelName = "local-model"
	}

	// Warmup request (excluded from timed stats)
	_, _ = executeStreamingChat(ctx, cfg.ServerURL, cfg.ModelName, "Say OK.", 8)

	ebpfMode := ebpftrace.ParseMode(string(cfg.EBPFMode))
	targetPIDs := resolveBenchmarkTargetPIDs(cfg, serverCmd, mon)
	sess, ebpfErr := ebpftrace.StartSession(ebpfMode, targetPIDs)
	if ebpfErr != nil {
		return WorkloadBenchmarkReport{}, ebpfErr
	}
	defer func() {
		if sess != nil {
			_ = sess.Stop()
		}
	}()

	burstInterval := 75 * time.Millisecond
	if mon != nil {
		if bi := mon.BurstInterval(); bi > 0 {
			burstInterval = bi
		}
	}

	// Start active burst telemetry polling during real inference load
	var burstWG sync.WaitGroup
	burstCtx, burstCancel := context.WithCancel(ctx)
	if mon != nil {
		_, _ = mon.PollOnce(ctx)
		burstWG.Add(1)
		go func(interval time.Duration) {
			defer burstWG.Done()
			tk := time.NewTicker(interval)
			defer tk.Stop()
			for {
				select {
				case <-burstCtx.Done():
					return
				case <-tk.C:
					_, _ = mon.PollOnce(burstCtx)
				}
			}
		}(burstInterval)
	}

	winStart := time.Now()
	results := make([]RequestMeasurement, 0, cfg.NumRequests)
	var ttfts, tpots, totals []float64
	totalTokens := 0

	for i := 0; i < cfg.NumRequests; i++ {
		prompt := cfg.Prompts[i%len(cfg.Prompts)]
		rm, err := executeStreamingChat(ctx, cfg.ServerURL, cfg.ModelName, prompt, cfg.MaxTokens)
		rm.RequestIndex = i + 1
		if err != nil {
			rm.Success = false
			rm.Error = err.Error()
		} else {
			rm.Success = true
			ttfts = append(ttfts, rm.TTFTMS)
			if rm.TPOTMS > 0 {
				tpots = append(tpots, rm.TPOTMS)
			}
			totals = append(totals, rm.TotalLatencyMS)
			totalTokens += rm.OutputTokens
		}
		if i == 0 && mon != nil {
			for _, pid := range mon.ActiveComputePIDs() {
				sess.AddTargetPID(pid)
			}
		}
		results = append(results, rm)
	}
	winEnd := time.Now()
	if mon != nil {
		for _, pid := range mon.ActiveComputePIDs() {
			sess.AddTargetPID(pid)
		}
	}
	ebpfSum := sess.Stop()
	burstCancel()
	burstWG.Wait()

	if mon != nil {
		_, _ = mon.PollOnce(ctx)
	}

	if len(totals) == 0 {
		return WorkloadBenchmarkReport{}, fmt.Errorf("all %d inference requests failed (last error: %s)", cfg.NumRequests, results[len(results)-1].Error)
	}

	totalDurSec := math.Max(1e-3, winEnd.Sub(winStart).Seconds())
	medLat := percentileFloat(totals, 0.50)
	p95Lat := percentileFloat(totals, 0.95)
	medTTFT := percentileFloat(ttfts, 0.50)
	p95TTFT := percentileFloat(ttfts, 0.95)
	medTPOT := percentileFloat(tpots, 0.50)
	tokPerSec := float64(totalTokens) / totalDurSec
	rps := float64(len(totals)) / totalDurSec

	var em telemetry.EnergyMeasurement
	if mon != nil {
		winEM := mon.MeasureWindow(winStart, winEnd, len(totals), medLat)
		em = telemetry.AggregateWindowMeasurements([]telemetry.EnergyMeasurement{winEM}, idlePowerW, totalDurSec*1000.0, burstInterval)
	} else {
		em = telemetry.EnergyMeasurement{
			DurationSec:           totalDurSec,
			TotalKernelTimeMS:     totalDurSec * 1000.0,
			MeasurementConfidence: "LOW",
			Provenance:            "ESTIMATED",
		}
	}

	joulesPer1KTokens := 0.0
	if totalTokens > 0 && em.TotalEnergyJoules > 0 {
		joulesPer1KTokens = (em.TotalEnergyJoules / float64(totalTokens)) * 1000.0
	}

	telemetryBackend := ""
	if mon != nil {
		telemetryBackend = mon.Backend()
	}

	launchesPerOutputToken := 0.0
	if ebpfSum.Active && totalTokens > 0 {
		launchesPerOutputToken = float64(ebpfSum.KernelLaunches) / float64(totalTokens)
	}

	wm := physics.WorkloadMetrics{
		LatencyMS:          medLat,
		P95LatencyMS:       p95Lat,
		ThroughputRPS:      rps,
		AvgPowerWatts:      em.AvgPowerWatts,
		EnergyJoulesPerReq: em.JoulesPerRequest,
		PeakVRAMMiB:        em.PeakVRAMMiB,
	}
	kpi := physics.ComputeBusinessKPIs(wm, econ)

	return WorkloadBenchmarkReport{
		Timestamp:              time.Now(),
		Adapter:                "OpenAI-Compatible Local Inference Adapter (llama-server / Ollama)",
		ServerURL:              cfg.ServerURL,
		ModelName:              cfg.ModelName,
		Provenance:             "MEASURED",
		TelemetryBackend:       telemetryBackend,
		HostCUDATrace:          ebpfSum,
		LaunchesPerOutputToken: launchesPerOutputToken,
		CompletedRequests:      len(totals),
		TotalOutputTokens:      totalTokens,
		TotalDurationSec:       totalDurSec,
		MedianTTFTMS:           medTTFT,
		P95TTFTMS:              p95TTFT,
		MedianTPOTMS:           medTPOT,
		MedianLatencyMS:        medLat,
		P95LatencyMS:           p95Lat,
		ThroughputTokensPerS:   tokPerSec,
		ThroughputRPS:          rps,
		MeasuredPeakVRAMMiB:    em.PeakVRAMMiB,
		Energy:                 em,
		JoulesPer1KTokens:      joulesPer1KTokens,
		Economics:              kpi,
		Requests:               results,
	}, nil
}

func resolveBenchmarkTargetPIDs(cfg BenchmarkConfig, serverCmd *exec.Cmd, mon *telemetry.Monitor) []int {
	seen := make(map[int]struct{})
	add := func(pid int) {
		if pid > 0 {
			seen[pid] = struct{}{}
		}
	}

	add(cfg.TracePID)
	if serverCmd != nil && serverCmd.Process != nil {
		add(serverCmd.Process.Pid)
	}

	portStr := extractPort(cfg.ServerURL, "")
	if portStr == "" {
		if strings.HasPrefix(strings.ToLower(cfg.ServerURL), "https://") {
			portStr = "443"
		} else if strings.HasPrefix(strings.ToLower(cfg.ServerURL), "http://") {
			portStr = "80"
		}
	}
	if portNum, err := strconv.Atoi(portStr); err == nil && portNum > 0 {
		for _, pid := range ebpftrace.FindPIDsByTCPPort(portNum) {
			add(pid)
		}
	}

	if mon != nil {
		for _, pid := range mon.ActiveComputePIDs() {
			add(pid)
		}
	}

	if len(seen) == 0 {
		return nil
	}
	pids := make([]int, 0, len(seen))
	for pid := range seen {
		pids = append(pids, pid)
	}
	sort.Ints(pids)
	return pids
}

func executeStreamingChat(ctx context.Context, baseURL, modelName, prompt string, maxTokens int) (RequestMeasurement, error) {
	bodyMap := map[string]any{
		"model":       modelName,
		"stream":      true,
		"max_tokens":  maxTokens,
		"temperature": 0.2,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	payload, _ := json.Marshal(bodyMap)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return RequestMeasurement{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	tStart := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return RequestMeasurement{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return RequestMeasurement{}, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	scanner := bufio.NewScanner(resp.Body)
	var firstTokenTime time.Time
	tokenCount := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err == nil && len(chunk.Choices) > 0 {
			if chunk.Choices[0].Delta.Content != "" {
				if firstTokenTime.IsZero() {
					firstTokenTime = time.Now()
				}
				tokenCount++
			}
		}
	}

	tEnd := time.Now()
	if err := scanner.Err(); err != nil {
		return RequestMeasurement{}, err
	}

	totalMS := float64(tEnd.Sub(tStart).Microseconds()) / 1000.0
	ttftMS := totalMS
	if !firstTokenTime.IsZero() {
		ttftMS = float64(firstTokenTime.Sub(tStart).Microseconds()) / 1000.0
	}
	tpotMS := 0.0
	if tokenCount > 1 && !firstTokenTime.IsZero() {
		decodeMS := float64(tEnd.Sub(firstTokenTime).Microseconds()) / 1000.0
		tpotMS = decodeMS / float64(tokenCount-1)
	}
	tps := 0.0
	if totalMS > 0 {
		tps = float64(tokenCount) / (totalMS / 1000.0)
	}

	return RequestMeasurement{
		TTFTMS:         ttftMS,
		TPOTMS:         tpotMS,
		TotalLatencyMS: totalMS,
		OutputTokens:   tokenCount,
		TokensPerSec:   tps,
	}, nil
}

func isEndpointHealthy(ctx context.Context, baseURL string) bool {
	c := &http.Client{Timeout: 1 * time.Second}
	for _, path := range []string{"/health", "/v1/models", "/api/tags"} {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+path, nil)
		if err != nil {
			continue
		}
		if resp, err := c.Do(req); err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return true
			}
		}
	}
	return false
}

func discoverServedModel(ctx context.Context, baseURL string) string {
	c := &http.Client{Timeout: 2 * time.Second}
	if req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/models", nil); err == nil {
		if resp, err := c.Do(req); err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				var parsed struct {
					Data []struct {
						ID string `json:"id"`
					} `json:"data"`
				}
				if json.NewDecoder(resp.Body).Decode(&parsed) == nil && len(parsed.Data) > 0 && parsed.Data[0].ID != "" {
					return parsed.Data[0].ID
				}
			}
		}
	}
	if req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/tags", nil); err == nil {
		if resp, err := c.Do(req); err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				var parsed struct {
					Models []struct {
						Name string `json:"name"`
					} `json:"models"`
				}
				if json.NewDecoder(resp.Body).Decode(&parsed) == nil && len(parsed.Models) > 0 && parsed.Models[0].Name != "" {
					return parsed.Models[0].Name
				}
			}
		}
	}
	return ""
}

func waitForHealthy(ctx context.Context, baseURL string, maxWait time.Duration) error {
	deadline := time.Now().Add(maxWait)
	for time.Now().Before(deadline) {
		if isEndpointHealthy(ctx, baseURL) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return fmt.Errorf("timed out waiting %s for %s", maxWait, baseURL)
}

func extractPort(rawURL, fallback string) string {
	trimmed := strings.TrimPrefix(strings.TrimPrefix(rawURL, "http://"), "https://")
	hostPort := strings.Split(trimmed, "/")[0]
	if _, port, err := net.SplitHostPort(hostPort); err == nil && port != "" {
		return port
	}
	return fallback
}

func percentileFloat(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := make([]float64, len(vals))
	copy(cp, vals)
	sort.Float64s(cp)
	idx := int(math.Ceil(p*float64(len(cp)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(cp) {
		idx = len(cp) - 1
	}
	return cp[idx]
}
