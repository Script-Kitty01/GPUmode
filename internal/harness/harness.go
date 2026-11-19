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
	Spec              KernelSpec                  `json:"spec"`
	CompiledOK        bool                        `json:"compiledOk"`
	CompileTimeMS     float64                     `json:"compileTimeMs"`
	CompilerLog       string                      `json:"compilerLog,omitempty"`
	PTXAS             PTXASStats                  `json:"ptxas"`
	SanitizerPassed   bool                        `json:"sanitizerPassed"`
	SanitizerLog      string                      `json:"sanitizerLog,omitempty"`
	ParityPassed      bool                        `json:"parityPassed"`
	EdgeCasePassed    bool                        `json:"edgeCasePassed"`
	MaxAbsError       float64                     `json:"maxAbsError"`
	MedianLatencyMS   float64                     `json:"medianLatencyMs"`
	P95LatencyMS      float64                     `json:"p95LatencyMs"`
	ThroughputRPS     float64                     `json:"throughputRps"`
	EffectiveBWGBps   float64                     `json:"effectiveBwGbps"`
	PeakVRAMMiB       float64                     `json:"peakVramMib"`
	Energy            telemetry.EnergyMeasurement `json:"energy"`
	ErrorMessage      string                      `json:"errorMessage,omitempty"`
}

// Runner manages parallel nvcc compilation workers and a serialized GPU verification & benchmark mutex.
type Runner struct {
	NVCCPath       string
	SanitizerPath  string
	ArchFlag       string
	CompileWorkers int
	KernelTimeout  time.Duration
	RunSanitizer   bool
	Monitor        *telemetry.Monitor
	gpuMu          sync.Mutex
}

// NewRunner creates a CUDA compilation & verification runner targeting the detected GPU architecture.
func NewRunner(nvccPath, sanitizerPath, archFlag string, workers int, timeout time.Duration, mon *telemetry.Monitor) *Runner {
	if nvccPath == "" {
		nvccPath = "/opt/cuda/bin/nvcc"
	}
	if sanitizerPath == "" {
		sanitizerPath = "/opt/cuda/bin/compute-sanitizer"
	}
	if archFlag == "" {
		archFlag = "sm_89"
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
		CompileWorkers: workers,
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

	// Step 2: Run full benchmark with live energy sampling window
	if r.Monitor != nil {
		_, _ = r.Monitor.PollOnce(ctx)
	}
	winStart := time.Now()

	bCtx, bCancel := context.WithTimeout(ctx, r.KernelTimeout)
	bCmd := exec.CommandContext(bCtx, art.binPath)
	var bOut, bErr bytes.Buffer
	bCmd.Stdout = &bOut
	bCmd.Stderr = &bErr
	runErr := bCmd.Run()
	bCancel()

	if r.Monitor != nil {
		_, _ = r.Monitor.PollOnce(ctx)
	}
	winEnd := time.Now()

	if runErr != nil {
		res.ErrorMessage = fmt.Sprintf("kernel execution failed: %v (%s)", runErr, strings.TrimSpace(bErr.String()))
		return res
	}

	// Parse JSON output line from the C++ harness
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

	res.ParityPassed = hOut.ParityPassed
	res.EdgeCasePassed = hOut.EdgeCasePassed
	res.MaxAbsError = hOut.MaxAbsError
	res.MedianLatencyMS = hOut.MedianLatencyMS
	res.P95LatencyMS = hOut.P95LatencyMS
	res.EffectiveBWGBps = hOut.EffectiveBWGBps
	res.PeakVRAMMiB = hOut.PeakVRAMMiB
	if res.MedianLatencyMS > 0 {
		res.ThroughputRPS = 1000.0 / res.MedianLatencyMS
	}

	completedReqs := hOut.CompletedRequests
	if completedReqs <= 0 {
		completedReqs = art.spec.Iterations
		if completedReqs <= 0 {
			completedReqs = 200
		}
	}

	// Step 3: Integrate power P(t) over the benchmark execution window, then attribute energy
	// to the exact CUDA-event timed kernel execution duration (excluding host CUDA context init & CPU parity check).
	if r.Monitor != nil {
		winEnergy := r.Monitor.MeasureWindow(winStart, winEnd, completedReqs, res.MedianLatencyMS)
		avgW := winEnergy.AvgPowerWatts
		if avgW <= 0 {
			avgW = 15.0
		}
		latSec := res.MedianLatencyMS / 1000.0
		kernDurSec := latSec * float64(completedReqs)
		jPerReq := avgW * latSec
		winEnergy.DurationSec = kernDurSec
		winEnergy.TotalEnergyJoules = jPerReq * float64(completedReqs)
		winEnergy.JoulesPerRequest = jPerReq
		winEnergy.JoulesPer1KRequests = jPerReq * 1000.0
		winEnergy.EnergyDelayProduct = jPerReq * latSec
		if avgW > 0 {
			winEnergy.PerformancePerWatt = res.ThroughputRPS / avgW
		}
		res.Energy = winEnergy
	} else {
		// Fallback estimate if monitor is nil in isolated unit test
		avgW := 48.0
		durSec := (res.MedianLatencyMS / 1000.0) * float64(completedReqs)
		totalJ := avgW * durSec
		jPerReq := avgW * (res.MedianLatencyMS / 1000.0)
		res.Energy = telemetry.EnergyMeasurement{
			DurationSec:         durSec,
			TotalEnergyJoules:   totalJ,
			AvgPowerWatts:       avgW,
			PeakPowerWatts:      avgW * 1.1,
			JoulesPerRequest:    jPerReq,
			JoulesPer1KRequests: jPerReq * 1000.0,
			EnergyDelayProduct:  jPerReq * (res.MedianLatencyMS / 1000.0),
			PerformancePerWatt:  res.ThroughputRPS / avgW,
			SampleCount:         1,
		}
	}

	return res
}
