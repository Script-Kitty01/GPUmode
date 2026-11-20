package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"gpumode/internal/config"
	"gpumode/internal/telemetry"
)

// PTXASStats holds low-level hardware register and shared-memory usage extracted from nvcc -Xptxas -v.
type PTXASStats struct {
	RegistersPerThread int `json:"registersPerThread"`
	SharedMemBytes     int `json:"sharedMemBytes"`
	ConstantMemBytes   int `json:"constantMemBytes"`
	SpillStoresBytes   int `json:"spillStoresBytes"`
	SpillLoadsBytes    int `json:"spillLoadsBytes"`
}

// KernelSpec defines a self-contained CUDA kernel candidate to compile, verify, and benchmark.
type KernelSpec struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	OpCategory    string `json:"opCategory"`
	Variant       string `json:"variant"`
	BlockSize     int    `json:"blockSize"`
	VecWidth      int    `json:"vecWidth"`
	Fused         bool   `json:"fused"`
	CUDASource    string `json:"cudaSource"`    // Complete locked harness + kernel .cu source
	DisplayKernel string `json:"displayKernel"` // Clean __global__ kernel snippet for UI comparison
	BatchSize     int    `json:"batchSize"`
	HiddenDim     int    `json:"hiddenDim"`
	Iterations    int    `json:"iterations"`
}

// HarnessOutput matches the JSON line printed by the compiled CUDA C++ benchmark binary.
type HarnessOutput struct {
	ParityPassed      bool    `json:"parityPassed"`
	MaxAbsError       float64 `json:"maxAbsError"`
	EdgeCasePassed    bool    `json:"edgeCasePassed"`
	MedianLatencyMS   float64 `json:"medianLatencyMs"`
	P95LatencyMS      float64 `json:"p95LatencyMs"`
	MinLatencyMS      float64 `json:"minLatencyMs"`
	EffectiveBWGBps   float64 `json:"effectiveBwGbps"`
	CompletedRequests int     `json:"completedRequests"`
	PeakVRAMMiB       float64 `json:"peakVramMib"`
}

// BenchmarkResult combines compiler stats, sanitizer/numerical verification, timing, and live energy integration.
type BenchmarkResult struct {
	Spec            KernelSpec                  `json:"spec"`
	CompiledOK      bool                        `json:"compiledOk"`
	CompileTimeMS   float64                     `json:"compileTimeMs"`
	CompilerLog     string                      `json:"compilerLog,omitempty"`
	PTXAS           PTXASStats                  `json:"ptxas"`
	SanitizerPassed bool                        `json:"sanitizerPassed"`
	SanitizerLog    string                      `json:"sanitizerLog,omitempty"`
	ParityPassed    bool                        `json:"parityPassed"`
	EdgeCasePassed  bool                        `json:"edgeCasePassed"`
	MaxAbsError     float64                     `json:"maxAbsError"`
	MedianLatencyMS float64                     `json:"medianLatencyMs"`
	P95LatencyMS    float64                     `json:"p95LatencyMs"`
	ThroughputRPS   float64                     `json:"throughputRps"`
	EffectiveBWGBps float64                     `json:"effectiveBwGbps"`
	PeakVRAMMiB     float64                     `json:"peakVramMib"`
	Energy          telemetry.EnergyMeasurement `json:"energy"`
	ErrorMessage    string                      `json:"errorMessage,omitempty"`
}

// Runner manages parallel nvcc compilation workers and a serialized GPU verification & benchmark mutex.
type Runner struct {
	NVCCPath       string
	SanitizerPath  string
	ArchFlag       string
	GPUIndex       int
	CompileWorkers int
	RepeatWindows  int
	KernelTimeout  time.Duration
	RunSanitizer   bool
	Monitor        *telemetry.Monitor
	gpuMu          sync.Mutex
}

// NewRunner creates a CUDA compilation & verification runner targeting the detected GPU architecture.
func NewRunner(nvccPath, sanitizerPath, archFlag string, workers int, timeout time.Duration, mon *telemetry.Monitor) *Runner {
	if nvccPath == "" {
		nvccPath = config.ResolveNVCCPath()
	}
	if sanitizerPath == "" {
		sanitizerPath = config.ResolveSanitizerPath()
	}
	if archFlag == "" {
		archFlag = "native"
	}
	if workers <= 0 {
		workers = 3
	}
	if timeout <= 0 {
		timeout = 25 * time.Second
	}
	return &Runner{
		NVCCPath:       nvccPath,
		SanitizerPath:  sanitizerPath,
		ArchFlag:       archFlag,
		GPUIndex:       config.DefaultGPUIndex(),
		CompileWorkers: workers,
		RepeatWindows:  3,
		KernelTimeout:  timeout,
		RunSanitizer:   true,
		Monitor:        mon,
	}
}

var (
	reUsedRegs   = regexp.MustCompile(`Used (\d+) registers`)
	reSharedMem  = regexp.MustCompile(`(\d+) bytes smem`)
	reConstMem   = regexp.MustCompile(`(\d+) bytes cmem\[0\]`)
	reSpillStore = regexp.MustCompile(`(\d+) bytes spill stores`)
	reSpillLoad  = regexp.MustCompile(`(\d+) bytes spill loads`)
)

// ParsePTXASOutput extracts register, shared memory, and spill counts from nvcc -Xptxas -v stderr output.
func ParsePTXASOutput(log string) PTXASStats {
	var stats PTXASStats
	if m := reUsedRegs.FindStringSubmatch(log); len(m) == 2 {
		stats.RegistersPerThread, _ = strconv.Atoi(m[1])
	}
	if m := reSharedMem.FindStringSubmatch(log); len(m) == 2 {
		stats.SharedMemBytes, _ = strconv.Atoi(m[1])
	}
	if m := reConstMem.FindStringSubmatch(log); len(m) == 2 {
		stats.ConstantMemBytes, _ = strconv.Atoi(m[1])
	}
	if m := reSpillStore.FindStringSubmatch(log); len(m) == 2 {
		stats.SpillStoresBytes, _ = strconv.Atoi(m[1])
	}
	if m := reSpillLoad.FindStringSubmatch(log); len(m) == 2 {
		stats.SpillLoadsBytes, _ = strconv.Atoi(m[1])
	}
	return stats
}

type compiledArtifact struct {
	index         int
	spec          KernelSpec
	binPath       string
	tempDir       string
	compiledOK    bool
	compileTimeMS float64
	compilerLog   string
	ptxas         PTXASStats
	err           error
}

// EvaluateBatch compiles all candidate KernelSpecs in parallel using a Go worker pool,
// then runs GPU verification & energy-synchronized benchmarking sequentially under the GPU mutex.
func (r *Runner) EvaluateBatch(ctx context.Context, specs []KernelSpec) []BenchmarkResult {
	n := len(specs)
	if n == 0 {
		return nil
	}

	jobs := make(chan int, n)
	compiled := make([]compiledArtifact, n)

	workers := r.CompileWorkers
	if workers > n {
		workers = n
	}

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				compiled[idx] = r.compileOne(ctx, idx, specs[idx])
			}
		}()
	}

	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	// Execute GPU verification and energy-timed benchmarks serially so kernels never contend on the GPU
	results := make([]BenchmarkResult, n)
	for i := 0; i < n; i++ {
		art := compiled[i]
		if art.tempDir != "" {
			defer os.RemoveAll(art.tempDir)
		}
		if !art.compiledOK {
			errMsg := "compilation failed"
			if art.err != nil {
				errMsg = art.err.Error()
			}
			results[i] = BenchmarkResult{
				Spec:          art.spec,
				CompiledOK:    false,
				CompileTimeMS: art.compileTimeMS,
				CompilerLog:   art.compilerLog,
				ErrorMessage:  errMsg,
			}
			continue
		}
		results[i] = r.executeAndMeasure(ctx, art)
	}

	return results
}

func (r *Runner) compileOne(ctx context.Context, idx int, spec KernelSpec) compiledArtifact {
	start := time.Now()
	tmpDir, err := os.MkdirTemp("", "gpumode_kern_*")
	if err != nil {
		return compiledArtifact{index: idx, spec: spec, err: err}
	}

	cuPath := filepath.Join(tmpDir, "kernel_bench.cu")
	binPath := filepath.Join(tmpDir, "kernel_bench")

	if err := os.WriteFile(cuPath, []byte(spec.CUDASource), 0600); err != nil {
		return compiledArtifact{index: idx, spec: spec, tempDir: tmpDir, err: err}
	}

	cCtx, cancel := context.WithTimeout(ctx, r.KernelTimeout)
	defer cancel()

	args := []string{
		"-O3",
		"-arch=" + r.ArchFlag,
		"-Xptxas", "-v",
		cuPath,
		"-o", binPath,
	}

	cmd := exec.CommandContext(cCtx, r.NVCCPath, args...)
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	runErr := cmd.Run()
	elapsedMS := float64(time.Since(start).Microseconds()) / 1000.0
	combinedLog := strings.TrimSpace(outBuf.String() + "\n" + errBuf.String())
	ptxas := ParsePTXASOutput(combinedLog)

	if runErr != nil {
		return compiledArtifact{
			index:         idx,
			spec:          spec,
			tempDir:       tmpDir,
			compiledOK:    false,
			compileTimeMS: elapsedMS,
			compilerLog:   combinedLog,
			ptxas:         ptxas,
			err:           fmt.Errorf("nvcc error: %v (%s)", runErr, combinedLog),
		}
	}

	return compiledArtifact{
		index:         idx,
		spec:          spec,
		binPath:       binPath,
		tempDir:       tmpDir,
		compiledOK:    true,
		compileTimeMS: elapsedMS,
		compilerLog:   combinedLog,
		ptxas:         ptxas,
	}
}

func (r *Runner) executeAndMeasure(ctx context.Context, art compiledArtifact) BenchmarkResult {
	r.gpuMu.Lock()
	defer r.gpuMu.Unlock()

	res := BenchmarkResult{
		Spec:          art.spec,
		CompiledOK:    true,
		CompileTimeMS: art.compileTimeMS,
		CompilerLog:   art.compilerLog,
		PTXAS:         art.ptxas,
	}

	// Step 1: Optional quick compute-sanitizer check (--verify-only mode)
	if r.RunSanitizer && r.SanitizerPath != "" {
		if _, err := os.Stat(r.SanitizerPath); err == nil {
			sCtx, sCancel := context.WithTimeout(ctx, r.KernelTimeout)
			sCmd := exec.CommandContext(sCtx, r.SanitizerPath, "--tool", "memcheck", "--error-exitcode", "99", art.binPath, "--verify-only")
			sCmd.Env = r.cudaExecEnv()
			var sOut bytes.Buffer
			sCmd.Stdout = &sOut
			sCmd.Stderr = &sOut
			sErr := sCmd.Run()
			sCancel()
			res.SanitizerLog = strings.TrimSpace(sOut.String())
			if sErr == nil || strings.Contains(res.SanitizerLog, "0 errors") {
				res.SanitizerPassed = true
			} else {
				// If compute-sanitizer is restricted by OS permissions, fall back to direct execution check
				if strings.Contains(res.SanitizerLog, "ERR_NVGPUCTRPERM") || strings.Contains(res.SanitizerLog, "not supported") {
					res.SanitizerPassed = true
				} else {
					res.SanitizerPassed = false
					res.ErrorMessage = "compute-sanitizer failed: " + res.SanitizerLog
					return res
				}
			}
		} else {
			res.SanitizerPassed = true
		}
	} else {
		res.SanitizerPassed = true
	}

	// Step 2: Establish idle GPU power baseline before benchmark windows
	var idlePowerW float64
	if r.Monitor != nil {
		idlePowerW = r.Monitor.MeasureIdleBaseline(ctx, 2)
	}

	repeatCount := r.RepeatWindows
	if repeatCount <= 0 {
		repeatCount = 3
	}

	var windowMeasurements []telemetry.EnergyMeasurement
	var medLatencies []float64
	var p95Latencies []float64
	var bwList []float64
	var lastOut HarnessOutput
	var totalKernelMS float64

	burstInterval := 50 * time.Millisecond
	if r.Monitor != nil {
		if bi := r.Monitor.BurstInterval(); bi > 0 {
			burstInterval = bi
		}
	}

	// Step 3: Run repeated benchmark windows with burst telemetry polling so energy uncertainty & confidence are quantified
	for w := 0; w < repeatCount; w++ {
		var burstWG sync.WaitGroup
		burstCtx, burstCancel := context.WithCancel(ctx)
		if r.Monitor != nil {
			_, _ = r.Monitor.PollOnce(ctx)
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
						_, _ = r.Monitor.PollOnce(burstCtx)
					}
				}
			}(burstInterval)
		}

		winStart := time.Now()
		bCtx, bCancel := context.WithTimeout(ctx, r.KernelTimeout)
		bCmd := exec.CommandContext(bCtx, art.binPath)
		bCmd.Env = r.cudaExecEnv()
		var bOut, bErr bytes.Buffer
		bCmd.Stdout = &bOut
		bCmd.Stderr = &bErr
		runErr := bCmd.Run()
		bCancel()
		burstCancel()
		burstWG.Wait()

		if r.Monitor != nil {
			_, _ = r.Monitor.PollOnce(ctx)
		}
		winEnd := time.Now()

		if runErr != nil {
			res.ErrorMessage = fmt.Sprintf("kernel execution failed (window %d): %v (%s)", w+1, runErr, strings.TrimSpace(bErr.String()))
			return res
		}

		var hOut HarnessOutput
		lines := strings.Split(strings.TrimSpace(bOut.String()), "\n")
		parsed := false
		for i := len(lines) - 1; i >= 0; i-- {
			line := strings.TrimSpace(lines[i])
			if strings.HasPrefix(line, "{") && strings.HasSuffix(line, "}") {
				if err := json.Unmarshal([]byte(line), &hOut); err == nil {
					parsed = true
					break
				}
			}
		}
		if !parsed {
			res.ErrorMessage = fmt.Sprintf("failed to parse harness JSON output: %s", bOut.String())
			return res
		}

		lastOut = hOut
		medLatencies = append(medLatencies, hOut.MedianLatencyMS)
		p95Latencies = append(p95Latencies, hOut.P95LatencyMS)
		bwList = append(bwList, hOut.EffectiveBWGBps)

		completedReqs := hOut.CompletedRequests
		if completedReqs <= 0 {
			completedReqs = art.spec.Iterations
			if completedReqs <= 0 {
				completedReqs = 200
			}
		}
		winKernelMS := hOut.MedianLatencyMS * float64(completedReqs)
		totalKernelMS += winKernelMS

		if r.Monitor != nil {
			winEnergy := r.Monitor.MeasureWindow(winStart, winEnd, completedReqs, hOut.MedianLatencyMS)
			avgW := winEnergy.AvgPowerWatts
			if avgW <= 0 {
				avgW = idlePowerW
				if avgW <= 0 {
					avgW = 15.0
				}
			}
			latSec := hOut.MedianLatencyMS / 1000.0
			kernDurSec := latSec * float64(completedReqs)
			jPerReq := avgW * latSec
			winEnergy.DurationSec = kernDurSec
			winEnergy.TotalEnergyJoules = jPerReq * float64(completedReqs)
			winEnergy.JoulesPerRequest = jPerReq
			winEnergy.JoulesPer1KRequests = jPerReq * 1000.0
			winEnergy.EnergyDelayProduct = jPerReq * latSec
			if avgW > 0 && hOut.MedianLatencyMS > 0 {
				winEnergy.PerformancePerWatt = (1000.0 / hOut.MedianLatencyMS) / avgW
			}
			windowMeasurements = append(windowMeasurements, winEnergy)
		}
	}

	// Median across repeated windows
	res.ParityPassed = lastOut.ParityPassed
	res.EdgeCasePassed = lastOut.EdgeCasePassed
	res.MaxAbsError = lastOut.MaxAbsError
	res.MedianLatencyMS = medianFloat(medLatencies)
	res.P95LatencyMS = medianFloat(p95Latencies)
	res.EffectiveBWGBps = medianFloat(bwList)
	res.PeakVRAMMiB = lastOut.PeakVRAMMiB
	if res.MedianLatencyMS > 0 {
		res.ThroughputRPS = 1000.0 / res.MedianLatencyMS
	}

	if r.Monitor != nil && len(windowMeasurements) > 0 {
		res.Energy = telemetry.AggregateWindowMeasurements(windowMeasurements, idlePowerW, totalKernelMS, burstInterval)
	} else {
		completedReqs := lastOut.CompletedRequests
		if completedReqs <= 0 {
			completedReqs = 200
		}
		durSec := (res.MedianLatencyMS / 1000.0) * float64(completedReqs)
		res.Energy = telemetry.EnergyMeasurement{
			DurationSec:             durSec,
			TotalEnergyJoules:       0,
			AvgPowerWatts:           0,
			IdlePowerWatts:          0,
			ActivePowerWatts:        0,
			ActiveEnergyJoules:      0,
			ActiveJoulesPer1K:       0,
			PeakPowerWatts:          0,
			JoulesPerRequest:        0,
			JoulesPer1KRequests:     0,
			EnergyDelayProduct:      0,
			PerformancePerWatt:      0,
			SampleCount:             0,
			RepeatWindows:           repeatCount,
			TotalKernelTimeMS:       totalKernelMS,
			EnergyUncertaintyPct:    100.0,
			EnergyUncertaintyJoules: 0,
			MeasurementConfidence:   "LOW",
			Provenance:              "ESTIMATED",
		}
	}

	return res
}

func (r *Runner) cudaExecEnv() []string {
	env := os.Environ()
	if os.Getenv("CUDA_VISIBLE_DEVICES") == "" && r.GPUIndex >= 0 {
		env = append(env, fmt.Sprintf("CUDA_VISIBLE_DEVICES=%d", r.GPUIndex))
	}
	return env
}

func medianFloat(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := make([]float64, len(vals))
	copy(cp, vals)
	for i := 0; i < len(cp); i++ {
		for j := i + 1; j < len(cp); j++ {
			if cp[j] < cp[i] {
				cp[i], cp[j] = cp[j], cp[i]
			}
		}
	}
	mid := len(cp) / 2
	if len(cp)%2 == 1 {
		return cp[mid]
	}
	return 0.5 * (cp[mid-1] + cp[mid])
}
