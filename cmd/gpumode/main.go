package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/harness"
	"gpumode/internal/optimizer"
	"gpumode/internal/physics"
	"gpumode/internal/profiler"
	"gpumode/internal/server"
	"gpumode/internal/store"
	"gpumode/internal/telemetry"
	"gpumode/internal/workload"
)

func main() {
	cfg := config.DefaultDaemonConfig()

	cmd := "dashboard"
	if len(os.Args) > 1 && os.Args[1][0] != '-' {
		cmd = os.Args[1]
		os.Args = append([]string{os.Args[0]}, os.Args[2:]...)
	}

	fs := flag.NewFlagSet("gpumode", flag.ExitOnError)
	addr := fs.String("addr", cfg.ListenAddr, "HTTP listen address for the local Web Dashboard")
	gpuIndex := fs.Int("gpu", cfg.GPUIndex, "NVIDIA GPU device index to probe, benchmark, and monitor")
	nvccPath := fs.String("nvcc", cfg.NVCCPath, "Path to NVIDIA nvcc compiler")
	sanitizerPath := fs.String("sanitizer", cfg.SanitizerPath, "Path to NVIDIA compute-sanitizer")
	nvidiaSMIPath := fs.String("nvidia-smi", cfg.NvidiaSMIPath, "Path to nvidia-smi")
	presetsPath := fs.String("presets", cfg.PresetsPath, "Path to JSON file of Digital Twin reference GPU presets")
	op := fs.String("op", "rmsnorm_residual", "Target operation (rmsnorm_residual, rmsnorm, layernorm, softmax, swiglu)")
	strategy := fs.String("strategy", string(optimizer.StrategyBayesian), "Autotuning strategy (BAYESIAN_UCB or GRID_SEARCH)")
	batch := fs.Int("batch", 256, "Batch x sequence token count")
	hidden := fs.Int("hidden", 4096, "Hidden dimension")
	iters := fs.Int("iters", 50, "Timed benchmark iterations")
	gpuRate := fs.Float64("gpu-rate", cfg.Economics.GPUHourlyRateUSD, "Custom GPU hourly rate ($/GPU-hr)")
	kwhRate := fs.Float64("kwh-rate", cfg.Economics.ElectricityRateKWh, "Electricity tariff ($/kWh)")
	pue := fs.Float64("pue", cfg.Economics.DatacenterPUE, "Datacenter PUE multiplier")
	qps := fs.Float64("qps", cfg.Economics.TargetQPS, "Target workload traffic (requests/sec)")
	sla := fs.Float64("sla", cfg.Economics.LatencySLAMS, "Latency p95 SLA in milliseconds")
	pythonPath := fs.String("python", cfg.PythonPath, "Python interpreter for PyTorch profiler bridge")
	proposerFlag := fs.String("proposer", "deterministic", "Kernel proposer engine (deterministic or deepseek)")
	deepseekModel := fs.String("deepseek-model", cfg.DeepSeekModelPath, "Path to local GGUF model for -proposer=deepseek or auto-server")
	llmURL := fs.String("llm-url", "http://127.0.0.1:8081", "OpenAI-compatible local inference server URL (llama-server or Ollama) for workload command")
	llmModel := fs.String("llm-model", "", "Model name for workload benchmark requests (auto-discovered if empty)")
	llmReqs := fs.Int("llm-reqs", 4, "Number of streaming inference requests for workload benchmark")
	llmMaxTok := fs.Int("llm-tokens", 48, "Max tokens per request for workload benchmark")
	autoServer := fs.Bool("auto-server", true, "Automatically launch temporary llama-server with -deepseek-model if -llm-url is not running")
	jsonOut := fs.Bool("json", false, "Emit full CausalChainReport as JSON")
	_ = fs.Parse(os.Args[1:])

	cfg.ListenAddr = *addr
	cfg.GPUIndex = *gpuIndex
	cfg.NVCCPath = *nvccPath
	cfg.SanitizerPath = *sanitizerPath
	cfg.NvidiaSMIPath = *nvidiaSMIPath
	cfg.PresetsPath = *presetsPath
	cfg.DeepSeekModelPath = *deepseekModel
	cfg.PythonPath = *pythonPath
	cfg.Economics.GPUHourlyRateUSD = *gpuRate
	cfg.Economics.ElectricityRateKWh = *kwhRate
	cfg.Economics.DatacenterPUE = *pue
	cfg.Economics.TargetQPS = *qps
	cfg.Economics.LatencySLAMS = *sla

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	hw, err := physics.DetectHardwareForDevice(ctx, cfg.GPUIndex, cfg.NVCCPath, cfg.NvidiaSMIPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[gpumode] CUDA hardware detection failed: %v\n", err)
		fmt.Fprintf(os.Stderr, "[gpumode] An NVIDIA GPU, nvidia-smi, and the CUDA toolkit (nvcc) are required.\n")
		os.Exit(1)
	}
	if _, nvccErr := exec.LookPath(cfg.NVCCPath); nvccErr != nil {
		fmt.Fprintf(os.Stderr, "[gpumode] nvcc not executable at %q (%v); pass -nvcc or set GPUMODE_NVCC\n", cfg.NVCCPath, nvccErr)
		os.Exit(1)
	}

	mon := telemetry.NewMonitorForGPU(cfg.NvidiaSMIPath, cfg.GPUIndex, cfg.TelemetryInterval, 300)
	mon.SetHardwareDefaults(hw.MaxSMClockMHz, hw.TDPWatts)
	mon.Start(ctx)

	runner := harness.NewRunner(cfg.NVCCPath, cfg.SanitizerPath, hw.ArchFlag, cfg.CompileWorkers, cfg.KernelTimeout, mon)
	runner.GPUIndex = cfg.GPUIndex
	pipe := profiler.NewPipeline(hw, mon, runner, cfg.PythonPath, "scripts/profile_pytorch.py")
	pipe.PresetsPath = cfg.PresetsPath
	if *proposerFlag == "deepseek" || cmd == "deepseek" {
		dsProposer := optimizer.NewLocalDeepSeekProposerForHardware("llama-cli", *deepseekModel, hw)
		pipe.Autotuner = optimizer.NewAutotuner(dsProposer, runner)
	}
	st, err := store.NewStore(cfg.DataDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[gpumode] failed to initialize data store: %v\n", err)
		os.Exit(1)
	}

	switch cmd {
	case "optimize", "analyze", "deepseek":
		strat := optimizer.StrategyBayesian
		if *strategy == string(optimizer.StrategyGrid) {
			strat = optimizer.StrategyGrid
		}
		rep := pipe.Execute(ctx, *op, *batch, *hidden, *iters, strat, cfg.Economics)
		_ = st.Save(rep)
		if *jsonOut {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			_ = enc.Encode(rep)
			return
		}
		printCausalReportCLI(rep)

	case "monitor":
		s, _ := mon.PollOnce(ctx)
		fmt.Printf("GPU: %s (%s) | Power: %.2f W | Clocks: %.0f / %.0f MHz | Temp: %.1f C | Util: %.0f%% SM / %.0f%% Mem | VRAM: %.0f / %.0f MiB\n",
			hw.Name, hw.ArchFlag, s.PowerWatts, s.SMClockMHz, s.MemClockMHz, s.TemperatureC, s.GPUUtilPct, s.MemUtilPct, s.VRAMUsedMiB, s.VRAMTotalMiB)

	case "workload":
		wCfg := workload.DefaultBenchmarkConfig()
		wCfg.ServerURL = *llmURL
		wCfg.ModelName = *llmModel
		wCfg.ModelPath = *deepseekModel
		wCfg.AutoServer = *autoServer
		wCfg.NumRequests = *llmReqs
		wCfg.MaxTokens = *llmMaxTok

		wCtx, wCancel := context.WithTimeout(ctx, 180*time.Second)
		defer wCancel()
		wRep, wErr := workload.RunInferenceBenchmark(wCtx, wCfg, mon, cfg.Economics)
		if wErr != nil {
			fmt.Fprintf(os.Stderr, "[gpumode] workload benchmark failed: %v\n", wErr)
			os.Exit(1)
		}
		if *jsonOut {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			_ = enc.Encode(wRep)
			return
		}
		printWorkloadReportCLI(wRep)

	case "dashboard", "serve":
		// Pre-warm an initial verified causal report if the store has none yet
		if _, ok := st.Latest(); !ok {
			fmt.Printf("[gpumode] Running initial CUDA-to-Economics verification on %s (%s)...\n", hw.Name, hw.ArchFlag)
			rep := pipe.Execute(ctx, *op, *batch, *hidden, *iters, optimizer.StrategyBayesian, cfg.Economics)
			_ = st.Save(rep)
		}
		srv := server.NewServer(cfg, hw, mon, pipe, st)
		httpSrv := &http.Server{
			Addr:    cfg.ListenAddr,
			Handler: srv.Handler(),
		}
		go func() {
			<-ctx.Done()
			shutCtx, c := context.WithTimeout(context.Background(), 3*time.Second)
			defer c()
			_ = httpSrv.Shutdown(shutCtx)
		}()
		fmt.Printf("[gpumode] Real-Time Go Daemon & Web Dashboard listening at http://%s\n", cfg.ListenAddr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "[gpumode] server error: %v\n", err)
			os.Exit(1)
		}

	default:
		fmt.Fprintf(os.Stderr, "Usage: gpumode [dashboard|optimize|deepseek|workload|monitor] [flags]\n")
		fs.PrintDefaults()
		os.Exit(1)
	}
}

func printWorkloadReportCLI(rep workload.WorkloadBenchmarkReport) {
	fmt.Printf("================================================================================\n")
	fmt.Printf(" gpuMode — Real Local LLM Workload Benchmark [%s]\n", rep.Provenance)
	fmt.Printf("================================================================================\n")
	fmt.Printf("  • Adapter             : %s\n", rep.Adapter)
	fmt.Printf("  • Model / Endpoint    : %s @ %s\n", rep.ModelName, rep.ServerURL)
	fmt.Printf("  • Requests / Tokens   : %d requests | %d output tokens in %.2fs\n",
		rep.CompletedRequests, rep.TotalOutputTokens, rep.TotalDurationSec)
	fmt.Printf("  • TTFT (p50 / p95)    : %.1f ms / %.1f ms\n", rep.MedianTTFTMS, rep.P95TTFTMS)
	fmt.Printf("  • TPOT / Request p50  : %.2f ms/tok | %.1f ms/req\n", rep.MedianTPOTMS, rep.MedianLatencyMS)
	fmt.Printf("  • Sustained Throughput: %.2f tokens/s (%.2f req/s)\n", rep.ThroughputTokensPerS, rep.ThroughputRPS)
	fmt.Printf("  • Measured Peak VRAM  : %.0f MiB\n", rep.MeasuredPeakVRAMMiB)
	fmt.Printf("  • Power & Energy      : %.1f W total (%.1f W active above %.1f W idle) | %.2f J/1K tokens [%s, %s conf, ±%.1f%%]\n",
		rep.Energy.AvgPowerWatts, rep.Energy.ActivePowerWatts, rep.Energy.IdlePowerWatts,
		rep.JoulesPer1KTokens, rep.Energy.Provenance, rep.Energy.MeasurementConfidence, rep.Energy.EnergyUncertaintyPct)
	fmt.Printf("  • Production Cost     : $%.4f / 1M output tokens | $%.4f / 1M requests\n",
		rep.CostPer1MTokensUSD, rep.Economics.CostPer1MInferencesUSD)
	fmt.Printf("================================================================================\n")
}

func printCausalReportCLI(rep profiler.CausalChainReport) {
	ec := rep.EconomicsComparison
	sa := rep.SevenAnswers
	fmt.Printf("================================================================================\n")
	fmt.Printf(" gpuMode — 7-Step CUDA-to-Economics Verification Report (%s / %s)\n", rep.Hardware.Name, rep.Hardware.ArchFlag)
	fmt.Printf("================================================================================\n")
	fmt.Printf("Headline Business Metrics:\n")
	fmt.Printf("  • $ / 1M Inferences      : $%.4f -> $%.4f (-%.1f%%)\n",
		ec.Baseline.CostPer1MInferencesUSD, ec.Optimized.CostPer1MInferencesUSD, ec.CostPer1MReductionPct)
	fmt.Printf("  • Joules / 1K Inferences : %.2f J -> %.2f J (-%.1f%% GPU energy)\n",
		ec.Baseline.JoulesPer1KInferences, ec.Optimized.JoulesPer1KInferences, ec.EnergySavedPct)
	fmt.Printf("  • Requests / GPU-Hour    : %.0f -> %.0f (%.2fx speedup)\n",
		ec.Baseline.RequestsPerGPUHour, ec.Optimized.RequestsPerGPUHour, ec.SpeedupFactor)
	fmt.Printf("  • Monthly Savings        : $%.2f/mo ($%.2f/yr) | %.1f kWh/mo | %.1f kg CO2e/mo\n\n",
		ec.MonthlySavingsUSD, ec.AnnualSavingsUSD, ec.MonthlyKWhSaved, ec.MonthlyCarbonSavedKgCO2e)

	fmt.Printf("1. Why is the GPU underutilized?\n   %s\n\n", sa.Q1WhyUnderutilized)
	fmt.Printf("2. Which operations are responsible?\n   %s\n\n", sa.Q2WhichOpsResponsible)
	fmt.Printf("3. What optimization helps?\n   %s\n\n", sa.Q3WhatOptimizationHelps)
	fmt.Printf("4. Is that optimization numerically safe?\n   %s\n\n", sa.Q4IsItNumericallySafe)
	fmt.Printf("5. How much energy does it save?\n   %s\n\n", sa.Q5HowMuchEnergySaved)
	fmt.Printf("6. Can fewer/cheaper GPUs serve the workload?\n   %s\n\n", sa.Q6CanCheaperGPUServe)
	fmt.Printf("7. How much money does the business save monthly?\n   %s\n", sa.Q7MonthlyMoneySaved)
	fmt.Printf("================================================================================\n")
}
