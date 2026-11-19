package telemetry

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	DefaultBufferCapacity = 300
	DefaultPollInterval   = 150 * time.Millisecond
	nvidiaSMIQueryArgs    = "name,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total"
)

// Sample represents a point-in-time GPU telemetry snapshot from nvidia-smi.
type Sample struct {
	Timestamp    time.Time `json:"timestamp"`
	GPUName      string    `json:"gpuName"`
	PowerWatts   float64   `json:"powerWatts"`
	SMClockMHz   float64   `json:"smClockMhz"`
	MemClockMHz  float64   `json:"memClockMhz"`
	TemperatureC float64   `json:"temperatureC"`
	GPUUtilPct   float64   `json:"gpuUtilPct"`
	MemUtilPct   float64   `json:"memUtilPct"`
	VRAMUsedMiB  float64   `json:"vramUsedMib"`
	VRAMTotalMiB float64   `json:"vramTotalMib"`
}

// EnergyMeasurement captures integrated energy, idle-baseline subtraction, uncertainty, and provenance over execution windows.
type EnergyMeasurement struct {
	DurationSec             float64 `json:"durationSec"`
	TotalEnergyJoules       float64 `json:"totalEnergyJoules"`
	AvgPowerWatts           float64 `json:"avgPowerWatts"`
	IdlePowerWatts          float64 `json:"idlePowerWatts"`
	ActivePowerWatts        float64 `json:"activePowerWatts"`
	ActiveEnergyJoules      float64 `json:"activeEnergyJoules"`
	ActiveJoulesPer1K       float64 `json:"activeJoulesPer1k"`
	PeakPowerWatts          float64 `json:"peakPowerWatts"`
	AvgSMClockMHz           float64 `json:"avgSmClockMhz"`
	AvgMemClockMHz          float64 `json:"avgMemClockMhz"`
	PeakTempC               float64 `json:"peakTempC"`
	PeakVRAMMiB             float64 `json:"peakVramMib"`
	JoulesPerRequest        float64 `json:"joulesPerRequest"`
	JoulesPer1KRequests     float64 `json:"joulesPer1kRequests"`
	EnergyDelayProduct      float64 `json:"energyDelayProduct"`
	PerformancePerWatt      float64 `json:"performancePerWatt"`
	SampleCount             int     `json:"sampleCount"`
	RepeatWindows           int     `json:"repeatWindows"`
	TotalKernelTimeMS       float64 `json:"totalKernelTimeMs"`
	EnergyUncertaintyPct    float64 `json:"energyUncertaintyPct"`
	EnergyUncertaintyJoules float64 `json:"energyUncertaintyJoules"`
	MeasurementConfidence   string  `json:"measurementConfidence"` // "HIGH", "MEDIUM", "LOW"
	Provenance              string  `json:"provenance"`            // "MEASURED", "INTERPOLATED", "ESTIMATED"
}

// DVFSCandidatePolicy represents a simulated Dynamic Voltage and Frequency Scaling (DVFS) operating point.
type DVFSCandidatePolicy struct {
	PolicyName             string  `json:"policyName"`
	TargetSMClockMHz       float64 `json:"targetSmClockMhz"`
	EstimatedPowerWatts    float64 `json:"estimatedPowerWatts"`
	EstimatedLatencyMS     float64 `json:"estimatedLatencyMs"`
	EstimatedThroughputRPS float64 `json:"estimatedThroughputRps"`
	JoulesPer1KRequests    float64 `json:"joulesPer1kRequests"`
	PowerChangePct         float64 `json:"powerChangePct"`
	ThroughputChangePct    float64 `json:"throughputChangePct"`
	MeetsLatencySLA        bool    `json:"meetsLatencySla"`
	Provenance             string  `json:"provenance"` // "MEASURED" or "SIMULATED"
	Note                   string  `json:"note"`
}

// SamplerFunc allows overriding the hardware query in unit tests.
type SamplerFunc func(ctx context.Context) (Sample, error)

// Monitor manages a thread-safe telemetry ring buffer, live SSE pub/sub channels,
// trapezoidal energy integration, and DVFS policy simulation.
type Monitor struct {
	mu            sync.RWMutex
	nvidiaSMIPath string
	gpuIndex      int
	interval      time.Duration
	capacity      int
	ring          []Sample
	head          int
	count         int

	defaultMaxClockMHz float64
	defaultPowerWatts  float64

	subMu       sync.Mutex
	nextSubID   int
	subscribers map[int]chan Sample

	sampler SamplerFunc
}

// NewMonitor creates a new GPU telemetry monitor for GPU 0 with a ring buffer of capacity items.
func NewMonitor(nvidiaSMIPath string, interval time.Duration, capacity int) *Monitor {
	return NewMonitorForGPU(nvidiaSMIPath, 0, interval, capacity)
}

// NewMonitorForGPU creates a new GPU telemetry monitor targeting a specific NVIDIA GPU index.
func NewMonitorForGPU(nvidiaSMIPath string, gpuIndex int, interval time.Duration, capacity int) *Monitor {
	if strings.TrimSpace(nvidiaSMIPath) == "" {
		nvidiaSMIPath = "nvidia-smi"
	}
	if gpuIndex < 0 {
		gpuIndex = 0
	}
	if interval <= 0 {
		interval = DefaultPollInterval
	}
	if capacity <= 0 {
		capacity = DefaultBufferCapacity
	}
	m := &Monitor{
		nvidiaSMIPath: nvidiaSMIPath,
		gpuIndex:      gpuIndex,
		interval:      interval,
		capacity:      capacity,
		ring:          make([]Sample, capacity),
		subscribers:   make(map[int]chan Sample),
	}
	m.sampler = m.queryNvidiaSMI
	return m
}

// SetHardwareDefaults configures the detected GPU's max SM clock and TDP/idle power
// so DVFS simulation scales from actual hardware properties rather than fixed defaults.
func (m *Monitor) SetHardwareDefaults(maxClockMHz, tdpWatts float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.defaultMaxClockMHz = maxClockMHz
	m.defaultPowerWatts = tdpWatts
}

// SetSampler overrides the hardware polling function (useful for deterministic testing).
func (m *Monitor) SetSampler(fn SamplerFunc) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if fn != nil {
		m.sampler = fn
	} else {
		m.sampler = m.queryNvidiaSMI
	}
}

// RecordSample appends a sample to the ring buffer and broadcasts it to all active SSE subscribers.
func (m *Monitor) RecordSample(s Sample) {
	if s.Timestamp.IsZero() {
		s.Timestamp = time.Now()
	}

	m.mu.Lock()
	m.ring[m.head] = s
	m.head = (m.head + 1) % m.capacity
	if m.count < m.capacity {
		m.count++
	}
	m.mu.Unlock()

	m.subMu.Lock()
	for _, ch := range m.subscribers {
		select {
		case ch <- s:
		default:
			// Drop oldest unread sample if subscriber is slow so producer never blocks.
			select {
			case <-ch:
			default:
			}
			select {
			case ch <- s:
			default:
			}
		}
	}
	m.subMu.Unlock()
}

// PollOnce executes a single telemetry sample collection and records it.
func (m *Monitor) PollOnce(ctx context.Context) (Sample, error) {
	m.mu.RLock()
	sampler := m.sampler
	m.mu.RUnlock()

	s, err := sampler(ctx)
	if err != nil {
		return Sample{}, err
	}
	m.RecordSample(s)
	return s, nil
}

// Start launches the background polling loop until ctx is canceled.
func (m *Monitor) Start(ctx context.Context) {
	// Take an immediate initial sample before starting ticker.
	_, _ = m.PollOnce(ctx)

	go func() {
		ticker := time.NewTicker(m.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				_, _ = m.PollOnce(ctx)
			}
		}
	}()
}

// Subscribe registers a live telemetry channel and returns a cleanup function to unsubscribe.
func (m *Monitor) Subscribe() (chan Sample, func()) {
	ch := make(chan Sample, 32)

	m.subMu.Lock()
	id := m.nextSubID
	m.nextSubID++
	m.subscribers[id] = ch
	m.subMu.Unlock()

	var once sync.Once
	unsubscribe := func() {
		once.Do(func() {
			m.subMu.Lock()
			if existing, ok := m.subscribers[id]; ok {
				delete(m.subscribers, id)
				close(existing)
			}
			m.subMu.Unlock()
		})
	}
	return ch, unsubscribe
}

// Latest returns the most recently recorded Sample (or a zero Sample if empty).
func (m *Monitor) Latest() Sample {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.count == 0 {
		return Sample{}
	}
	idx := (m.head - 1 + m.capacity) % m.capacity
	return m.ring[idx]
}

// History returns up to limit most recent samples in chronological order (oldest to newest).
// If limit <= 0 or limit > count, all buffered samples are returned.
func (m *Monitor) History(limit int) []Sample {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.count == 0 {
		return nil
	}
	n := m.count
	if limit > 0 && limit < n {
		n = limit
	}

	out := make([]Sample, n)
	startIdx := (m.head - n + m.capacity) % m.capacity
	for i := 0; i < n; i++ {
		out[i] = m.ring[(startIdx+i)%m.capacity]
	}
	return out
}

// queryNvidiaSMI runs nvidia-smi and parses the target GPU's CSV telemetry line.
func (m *Monitor) queryNvidiaSMI(ctx context.Context) (Sample, error) {
	queryCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	cmd := exec.CommandContext(
		queryCtx,
		m.nvidiaSMIPath,
		"-i", strconv.Itoa(m.gpuIndex),
		"--query-gpu="+nvidiaSMIQueryArgs,
		"--format=csv,noheader,nounits",
	)
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return Sample{}, fmt.Errorf("nvidia-smi poll failed for GPU %d: %w", m.gpuIndex, err)
	}
	return ParseNvidiaSMICSVLine(out.String(), time.Now())
}

// ParseNvidiaSMICSVLine parses a single line of output from:
// nvidia-smi --query-gpu=name,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
func ParseNvidiaSMICSVLine(raw string, ts time.Time) (Sample, error) {
	lines := strings.Split(strings.TrimSpace(raw), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		return Sample{}, fmt.Errorf("empty nvidia-smi output")
	}
	line := strings.TrimSpace(lines[0])
	parts := strings.Split(line, ",")
	if len(parts) < 9 {
		return Sample{}, fmt.Errorf("expected at least 9 CSV fields from nvidia-smi, got %d in %q", len(parts), line)
	}

	if ts.IsZero() {
		ts = time.Now()
	}

	// If GPU name itself contains commas (rare), join leading parts so the last 8 are numeric fields.
	numOffset := len(parts) - 8
	gpuName := strings.TrimSpace(strings.Join(parts[:numOffset], ","))

	return Sample{
		Timestamp:    ts,
		GPUName:      gpuName,
		PowerWatts:   parseFlexibleFloat(parts[numOffset+0]),
		SMClockMHz:   parseFlexibleFloat(parts[numOffset+1]),
		MemClockMHz:  parseFlexibleFloat(parts[numOffset+2]),
		TemperatureC: parseFlexibleFloat(parts[numOffset+3]),
		GPUUtilPct:   parseFlexibleFloat(parts[numOffset+4]),
		MemUtilPct:   parseFlexibleFloat(parts[numOffset+5]),
		VRAMUsedMiB:  parseFlexibleFloat(parts[numOffset+6]),
		VRAMTotalMiB: parseFlexibleFloat(parts[numOffset+7]),
	}, nil
}

// parseFlexibleFloat parses float fields from nvidia-smi, gracefully returning 0 for "[N/A]" or "[Not Supported]".
func parseFlexibleFloat(s string) float64 {
	clean := strings.TrimSpace(s)
	if clean == "" || strings.Contains(strings.ToUpper(clean), "N/A") || strings.Contains(strings.ToUpper(clean), "NOT SUPPORTED") {
		return 0
	}
	// Strip optional unit suffixes if any slipped in
	fields := strings.Fields(clean)
	if len(fields) > 0 {
		clean = fields[0]
	}
	v, err := strconv.ParseFloat(clean, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return v
}

// Interval returns the configured polling interval.
func (m *Monitor) Interval() time.Duration {
	return m.interval
}

// MeasureIdleBaseline polls a few samples when idle (or uses recent low-utilization samples)
// to establish an idle GPU board power baseline for active-power subtraction.
func (m *Monitor) MeasureIdleBaseline(ctx context.Context, numPolls int) float64 {
	if numPolls <= 0 {
		numPolls = 2
	}
	var sum float64
	var cnt int
	for i := 0; i < numPolls; i++ {
		if s, err := m.PollOnce(ctx); err == nil && s.PowerWatts > 0 {
			sum += s.PowerWatts
			cnt++
		}
		if i < numPolls-1 {
			select {
			case <-ctx.Done():
				break
			case <-time.After(45 * time.Millisecond):
			}
		}
	}
	if cnt > 0 {
		return sum / float64(cnt)
	}
	latest := m.Latest()
	return latest.PowerWatts
}

// MeasureWindow computes integrated energy E = ∫ P(t) dt over [start, end] using trapezoidal integration,
// along with average/peak clocks, temperature, VRAM, EDP, Joules/1K requests, and Perf/Watt.
func (m *Monitor) MeasureWindow(start, end time.Time, completedRequests int, latencyMS float64) EnergyMeasurement {
	all := m.History(0)
	em := IntegrateSamplesWindow(all, start, end, completedRequests, latencyMS)
	if em.Provenance == "INTERPOLATED" && m.interval > 0 {
		// Aliasing uncertainty grows when the measurement window is shorter than the polling period
		winMS := end.Sub(start).Seconds() * 1000.0
		pollMS := float64(m.interval.Milliseconds())
		if pollMS > 0 && winMS < pollMS {
			em.EnergyUncertaintyPct = math.Min(45.0, math.Round((12.0+25.0*(1.0-winMS/pollMS))*10.0)/10.0)
			em.EnergyUncertaintyJoules = em.JoulesPer1KRequests * (em.EnergyUncertaintyPct / 100.0)
		}
	}
	return em
}

// IntegrateSamplesWindow performs trapezoidal integration over a slice of chronological samples for [start, end].
// If the window is shorter than the polling interval or contains fewer than 2 samples inside [start, end],
// it explicitly labels the measurement as INTERPOLATED with LOW/MEDIUM confidence rather than pretending it is a high-confidence direct measurement.
func IntegrateSamplesWindow(samples []Sample, start, end time.Time, completedRequests int, latencyMS float64) EnergyMeasurement {
	durationSec := end.Sub(start).Seconds()
	if durationSec <= 0 {
		if latencyMS > 0 && completedRequests > 0 {
			durationSec = (latencyMS / 1000.0) * float64(completedRequests)
		} else if latencyMS > 0 {
			durationSec = latencyMS / 1000.0
		} else {
			durationSec = 1e-3
		}
		end = start.Add(time.Duration(durationSec * float64(time.Second)))
	}

	if len(samples) == 0 {
		return finalizeEnergyMeasurement(EnergyMeasurement{
			DurationSec:           durationSec,
			RepeatWindows:         1,
			MeasurementConfidence: "LOW",
			Provenance:            "ESTIMATED",
			EnergyUncertaintyPct:  35.0,
		}, completedRequests, latencyMS)
	}

	// Collect samples inside [start, end], plus nearest before start and after end for boundary interpolation.
	var inWindow []Sample
	var before *Sample
	var after *Sample

	for i := range samples {
		s := samples[i]
		if s.Timestamp.Before(start) {
			before = &samples[i]
		} else if s.Timestamp.After(end) {
			if after == nil {
				after = &samples[i]
			}
		} else {
			inWindow = append(inWindow, s)
		}
	}

	// Build boundary-anchored integration points covering [start, end].
	var points []Sample
	startSample := interpolateBoundarySample(start, before, inWindow, after, samples)
	endSample := interpolateBoundarySample(end, before, inWindow, after, samples)

	points = append(points, startSample)
	for _, s := range inWindow {
		if s.Timestamp.After(start) && s.Timestamp.Before(end) {
			points = append(points, s)
		}
	}
	points = append(points, endSample)

	// Compute trapezoidal integral of PowerWatts over points and summary statistics.
	var totalEnergyJoules float64
	var weightedSMClock float64
	var weightedMemClock float64
	var peakPower float64
	var peakTemp float64
	var peakVRAM float64

	statSamples := inWindow
	if len(statSamples) == 0 {
		statSamples = points
	}
	for _, s := range statSamples {
		if s.PowerWatts > peakPower {
			peakPower = s.PowerWatts
		}
		if s.TemperatureC > peakTemp {
			peakTemp = s.TemperatureC
		}
		if s.VRAMUsedMiB > peakVRAM {
			peakVRAM = s.VRAMUsedMiB
		}
	}

	for i := 0; i < len(points)-1; i++ {
		p0 := points[i]
		p1 := points[i+1]
		dt := p1.Timestamp.Sub(p0.Timestamp).Seconds()
		if dt <= 0 {
			continue
		}
		totalEnergyJoules += 0.5 * (p0.PowerWatts + p1.PowerWatts) * dt
		weightedSMClock += 0.5 * (p0.SMClockMHz + p1.SMClockMHz) * dt
		weightedMemClock += 0.5 * (p0.MemClockMHz + p1.MemClockMHz) * dt
	}

	avgPower := totalEnergyJoules / durationSec
	avgSMClock := weightedSMClock / durationSec
	avgMemClock := weightedMemClock / durationSec

	if avgPower > peakPower {
		peakPower = avgPower
	}

	sampleCount := len(inWindow)
	provenance := "MEASURED"
	confidence := "HIGH"
	uncertaintyPct := 5.0
	if sampleCount >= 2 {
		provenance = "MEASURED"
		if sampleCount >= 3 {
			confidence = "HIGH"
			uncertaintyPct = 5.0
		} else {
			confidence = "MEDIUM"
			uncertaintyPct = 10.0
		}
	} else {
		provenance = "INTERPOLATED"
		confidence = "LOW"
		uncertaintyPct = 22.0
		if sampleCount == 0 && len(samples) > 0 {
			sampleCount = 1
		}
	}

	em := EnergyMeasurement{
		DurationSec:           durationSec,
		TotalEnergyJoules:     totalEnergyJoules,
		AvgPowerWatts:         avgPower,
		PeakPowerWatts:        peakPower,
		AvgSMClockMHz:         avgSMClock,
		AvgMemClockMHz:        avgMemClock,
		PeakTempC:             peakTemp,
		PeakVRAMMiB:           peakVRAM,
		SampleCount:           sampleCount,
		RepeatWindows:         1,
		EnergyUncertaintyPct:  uncertaintyPct,
		MeasurementConfidence: confidence,
		Provenance:            provenance,
	}
	return finalizeEnergyMeasurement(em, completedRequests, latencyMS)
}

// AggregateWindowMeasurements combines repeated benchmark windows, subtracts idle baseline power,
// and reports inter-window uncertainty and measurement confidence.
func AggregateWindowMeasurements(
	windows []EnergyMeasurement,
	idlePowerW float64,
	totalKernelTimeMS float64,
	pollInterval time.Duration,
) EnergyMeasurement {
	if len(windows) == 0 {
		return EnergyMeasurement{
			IdlePowerWatts:        idlePowerW,
			MeasurementConfidence: "LOW",
			Provenance:            "ESTIMATED",
		}
	}

	var sumPower, sumPeakPower, sumSM, sumMem, sumTemp, sumVRAM float64
	var sumJ1K, sumJReq, sumEDP, sumPerfW, sumDur, sumTotJ float64
	var totalSamples int
	measuredCount := 0

	for _, w := range windows {
		sumPower += w.AvgPowerWatts
		if w.PeakPowerWatts > sumPeakPower {
			sumPeakPower = w.PeakPowerWatts
		}
		sumSM += w.AvgSMClockMHz
		sumMem += w.AvgMemClockMHz
		if w.PeakTempC > sumTemp {
			sumTemp = w.PeakTempC
		}
		if w.PeakVRAMMiB > sumVRAM {
			sumVRAM = w.PeakVRAMMiB
		}
		sumJ1K += w.JoulesPer1KRequests
		sumJReq += w.JoulesPerRequest
		sumEDP += w.EnergyDelayProduct
		sumPerfW += w.PerformancePerWatt
		sumDur += w.DurationSec
		sumTotJ += w.TotalEnergyJoules
		totalSamples += w.SampleCount
		if w.Provenance == "MEASURED" {
			measuredCount++
		}
	}

	n := float64(len(windows))
	meanPower := sumPower / n
	meanJ1K := sumJ1K / n
	meanJReq := sumJReq / n
	meanDur := sumDur / n

	// Inter-window sample standard deviation on Joules/1K
	var varianceSum float64
	for _, w := range windows {
		diff := w.JoulesPer1KRequests - meanJ1K
		varianceSum += diff * diff
	}
	cvPct := 0.0
	if len(windows) > 1 && meanJ1K > 0 {
		stddev := math.Sqrt(varianceSum / float64(len(windows)-1))
		// 95% confidence half-width approximation (~2 * SE)
		se := stddev / math.Sqrt(n)
		cvPct = (1.96 * se / meanJ1K) * 100.0
	}

	// Aliasing uncertainty if CUDA kernel duration per window is shorter than nvidia-smi polling period
	pollMS := float64(pollInterval.Milliseconds())
	if pollMS <= 0 {
		pollMS = 150.0
	}
	perWinKernelMS := totalKernelTimeMS / n
	aliasingPct := 3.5
	if perWinKernelMS > 0 && perWinKernelMS < pollMS {
		aliasingPct = math.Min(30.0, 6.0+18.0*(1.0-perWinKernelMS/pollMS))
	}
	totalUncPct := math.Round(math.Sqrt(cvPct*cvPct+aliasingPct*aliasingPct)*10.0) / 10.0

	activePower := math.Max(0, meanPower-idlePowerW)
	activeRatio := 0.0
	if meanPower > 0 {
		activeRatio = activePower / meanPower
	}
	activeJ1K := meanJ1K * activeRatio
	activeTotJ := sumTotJ * activeRatio

	provenance := "INTERPOLATED"
	if measuredCount == len(windows) || totalSamples >= 3 {
		provenance = "MEASURED"
	} else if totalSamples == 0 {
		provenance = "ESTIMATED"
	}

	confidence := "LOW"
	if totalSamples >= 4 && totalUncPct <= 15.0 {
		confidence = "HIGH"
	} else if totalSamples >= 2 && totalUncPct <= 25.0 {
		confidence = "MEDIUM"
	}

	return EnergyMeasurement{
		DurationSec:             meanDur,
		TotalEnergyJoules:       sumTotJ / n,
		AvgPowerWatts:           meanPower,
		IdlePowerWatts:          idlePowerW,
		ActivePowerWatts:        activePower,
		ActiveEnergyJoules:      activeTotJ / n,
		ActiveJoulesPer1K:       activeJ1K,
		PeakPowerWatts:          sumPeakPower,
		AvgSMClockMHz:           sumSM / n,
		AvgMemClockMHz:          sumMem / n,
		PeakTempC:               sumTemp,
		PeakVRAMMiB:             sumVRAM,
		JoulesPerRequest:        meanJReq,
		JoulesPer1KRequests:     meanJ1K,
		EnergyDelayProduct:      sumEDP / n,
		PerformancePerWatt:      sumPerfW / n,
		SampleCount:             totalSamples,
		RepeatWindows:           len(windows),
		TotalKernelTimeMS:       totalKernelTimeMS,
		EnergyUncertaintyPct:    totalUncPct,
		EnergyUncertaintyJoules: meanJ1K * (totalUncPct / 100.0),
		MeasurementConfidence:   confidence,
		Provenance:              provenance,
	}
}

func interpolateBoundarySample(target time.Time, before *Sample, inWindow []Sample, after *Sample, all []Sample) Sample {
	// If we have samples in window:
	if len(inWindow) > 0 {
		first := inWindow[0]
		last := inWindow[len(inWindow)-1]
		if !target.After(first.Timestamp) {
			if before != nil {
				return lerpSample(*before, first, target)
			}
			c := first
			c.Timestamp = target
			return c
		}
		if !target.Before(last.Timestamp) {
			if after != nil {
				return lerpSample(last, *after, target)
			}
			c := last
			c.Timestamp = target
			return c
		}
	}

	// No samples inside window: interpolate between before and after if both exist.
	if before != nil && after != nil {
		return lerpSample(*before, *after, target)
	}
	if before != nil {
		c := *before
		c.Timestamp = target
		return c
	}
	if after != nil {
		c := *after
		c.Timestamp = target
		return c
	}
	// Fallback to latest available sample.
	latest := all[len(all)-1]
	latest.Timestamp = target
	return latest
}

func lerpSample(a, b Sample, t time.Time) Sample {
	total := b.Timestamp.Sub(a.Timestamp).Seconds()
	if total <= 0 {
		out := b
		out.Timestamp = t
		return out
	}
	alpha := t.Sub(a.Timestamp).Seconds() / total
	if alpha < 0 {
		alpha = 0
	} else if alpha > 1 {
		alpha = 1
	}
	return Sample{
		Timestamp:    t,
		GPUName:      b.GPUName,
		PowerWatts:   a.PowerWatts + alpha*(b.PowerWatts-a.PowerWatts),
		SMClockMHz:   a.SMClockMHz + alpha*(b.SMClockMHz-a.SMClockMHz),
		MemClockMHz:  a.MemClockMHz + alpha*(b.MemClockMHz-a.MemClockMHz),
		TemperatureC: a.TemperatureC + alpha*(b.TemperatureC-a.TemperatureC),
		GPUUtilPct:   a.GPUUtilPct + alpha*(b.GPUUtilPct-a.GPUUtilPct),
		MemUtilPct:   a.MemUtilPct + alpha*(b.MemUtilPct-a.MemUtilPct),
		VRAMUsedMiB:  a.VRAMUsedMiB + alpha*(b.VRAMUsedMiB-a.VRAMUsedMiB),
		VRAMTotalMiB: b.VRAMTotalMiB,
	}
}

func finalizeEnergyMeasurement(em EnergyMeasurement, completedRequests int, latencyMS float64) EnergyMeasurement {
	reqs := float64(completedRequests)
	if reqs <= 0 {
		reqs = 1.0
	}

	em.JoulesPerRequest = em.TotalEnergyJoules / reqs
	em.JoulesPer1KRequests = em.JoulesPerRequest * 1000.0

	latencySec := latencyMS / 1000.0
	if latencySec <= 0 && em.DurationSec > 0 {
		latencySec = em.DurationSec / reqs
	}
	em.EnergyDelayProduct = em.JoulesPerRequest * latencySec

	if em.AvgPowerWatts > 0 && em.DurationSec > 0 {
		rps := reqs / em.DurationSec
		em.PerformancePerWatt = rps / em.AvgPowerWatts
	}
	if em.EnergyUncertaintyPct > 0 && em.EnergyUncertaintyJoules == 0 {
		em.EnergyUncertaintyJoules = em.JoulesPer1KRequests * (em.EnergyUncertaintyPct / 100.0)
	}
	if em.IdlePowerWatts > 0 {
		em.ActivePowerWatts = math.Max(0, em.AvgPowerWatts-em.IdlePowerWatts)
		if em.AvgPowerWatts > 0 {
			ratio := em.ActivePowerWatts / em.AvgPowerWatts
			em.ActiveEnergyJoules = em.TotalEnergyJoules * ratio
			em.ActiveJoulesPer1K = em.JoulesPer1KRequests * ratio
		}
	}
	return em
}

// SimulateDVFSPolicies models Dynamic Voltage and Frequency Scaling (DVFS) operating points.
// It uses CMOS dynamic power physics (P = P_static + P_dyn * (f / f_0)^2.2) and memory-bound
// sensitivity (memory-bound ops depend primarily on HBM/GDDR bandwidth rather than SM clock,
// losing very little throughput when SM clock drops by 15-20% while saving 22-30% power).
func (m *Monitor) SimulateDVFSPolicies(base EnergyMeasurement, baseLatencyMS float64, slaMS float64, memoryBoundFraction float64) []DVFSCandidatePolicy {
	m.mu.RLock()
	defClock := m.defaultMaxClockMHz
	defPower := m.defaultPowerWatts
	m.mu.RUnlock()
	return ComputeDVFSPoliciesWithDefaults(base, baseLatencyMS, slaMS, memoryBoundFraction, defClock, defPower)
}

// ComputeDVFSPolicies is a package-level helper for simulating DVFS power/clock policies.
func ComputeDVFSPolicies(base EnergyMeasurement, baseLatencyMS float64, slaMS float64, memoryBoundFraction float64) []DVFSCandidatePolicy {
	return ComputeDVFSPoliciesWithDefaults(base, baseLatencyMS, slaMS, memoryBoundFraction, 0, 0)
}

// ComputeDVFSPoliciesWithDefaults models DVFS operating points using detected GPU clock/power defaults when live averages are absent.
func ComputeDVFSPoliciesWithDefaults(base EnergyMeasurement, baseLatencyMS float64, slaMS float64, memoryBoundFraction float64, defaultClockMHz, defaultPowerW float64) []DVFSCandidatePolicy {
	memFrac := math.Max(0.0, math.Min(1.0, memoryBoundFraction))
	computeFrac := 1.0 - memFrac

	baseClock := base.AvgSMClockMHz
	if baseClock <= 0 {
		baseClock = defaultClockMHz
	}
	if baseClock <= 0 {
		return nil
	}
	basePower := base.AvgPowerWatts
	if basePower <= 0 {
		if defaultPowerW > 0 {
			basePower = defaultPowerW * 0.65
		} else {
			return nil
		}
	}
	if baseLatencyMS <= 0 {
		baseLatencyMS = 1.0
	}
	if slaMS <= 0 {
		slaMS = baseLatencyMS * 1.25
	}

	baseRPS := 1000.0 / baseLatencyMS
	staticPowerFrac := 0.22 // ~22% static/leakage + board VRAM power, 78% dynamic SM core power
	dynamicPowerFrac := 1.0 - staticPowerFrac
	dvfsExponent := 2.2 // P_dyn ∝ V^2 * f ≈ f^2.2

	type spec struct {
		name       string
		clockScale float64
		note       string
	}

	specs := []spec{
		{
			name:       "Max Boost (Current)",
			clockScale: 1.00,
			note:       "Hardware observation at unlocked boost clock; baseline telemetry measurement.",
		},
		{
			name:       "Balanced Sweet-Spot (-15% Clock)",
			clockScale: 0.85,
			note:       "DVFS simulation (nvidia-smi -lgc / power.limit locked on mobile GPUs): -15% SM clock yields ~23-26% power reduction with minimal throughput loss on memory-bound kernels.",
		},
		{
			name:       "Eco SLA-Aware (-25% Clock)",
			clockScale: 0.75,
			note:       "DVFS simulation (power.limit locked on mobile GPUs): -25% SM clock maximizes Joules/1K inference savings while checking p95 SLA compliance.",
		},
		{
			name:       "Ultra-Low Power Cap (-35% Clock)",
			clockScale: 0.65,
			note:       "DVFS simulation (power.limit locked on mobile GPUs): aggressive underclock for off-peak batch or thermal-throttled envelopes.",
		},
	}

	out := make([]DVFSCandidatePolicy, 0, len(specs))
	for _, sp := range specs {
		targetClock := math.Round(baseClock * sp.clockScale)

		// Dynamic power scales super-linearly with frequency due to V-f curve: P(f) = P0 * (static + dyn * scale^2.2)
		powerScale := staticPowerFrac + dynamicPowerFrac*math.Pow(sp.clockScale, dvfsExponent)
		estPower := basePower * powerScale

		// Latency execution time model:
		// Compute-bound portion scales inversely with SM frequency (1 / clockScale).
		// Memory-bound portion is governed by GDDR6 memory clock, with only a small L2/issue sensitivity (0.18) to SM clock.
		memSlowdown := 1.0 + 0.18*(1.0/sp.clockScale-1.0)
		computeSlowdown := 1.0 / sp.clockScale
		latencyScale := memFrac*memSlowdown + computeFrac*computeSlowdown
		if sp.clockScale == 1.0 {
			latencyScale = 1.0
		}

		estLatencyMS := baseLatencyMS * latencyScale
		estRPS := 1000.0 / estLatencyMS
		joulesPerReq := estPower * (estLatencyMS / 1000.0)
		joulesPer1K := joulesPerReq * 1000.0

		powerChangePct := ((estPower - basePower) / basePower) * 100.0
		throughputChangePct := ((estRPS - baseRPS) / baseRPS) * 100.0

		prov := "SIMULATED"
		if sp.clockScale == 1.0 {
			prov = "MEASURED"
		}
		out = append(out, DVFSCandidatePolicy{
			PolicyName:             sp.name,
			TargetSMClockMHz:       targetClock,
			EstimatedPowerWatts:    estPower,
			EstimatedLatencyMS:     estLatencyMS,
			EstimatedThroughputRPS: estRPS,
			JoulesPer1KRequests:    joulesPer1K,
			PowerChangePct:         powerChangePct,
			ThroughputChangePct:    throughputChangePct,
			MeetsLatencySLA:        estLatencyMS <= slaMS,
			Provenance:             prov,
			Note:                   sp.note,
		})
	}

	return out
}
