package ebpftrace

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

type Mode string

const (
	ModeAuto Mode = "auto"
	ModeOn   Mode = "on"
	ModeOff  Mode = "off"

	ProvenanceMeasuredEBPF            = "MEASURED_EBPF"
	ProvenanceUnavailableUnprivileged = "UNAVAILABLE_UNPRIVILEGED"
	ProvenanceUnavailableUnsupported  = "UNAVAILABLE_UNSUPPORTED"
	ProvenanceDisabled                = "DISABLED"
)

// ParseMode normalizes raw user configuration into a valid eBPF tracing Mode.
// Empty or unrecognized values default safely to ModeAuto.
func ParseMode(raw string) Mode {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "on":
		return ModeOn
	case "off":
		return ModeOff
	case "auto", "":
		return ModeAuto
	default:
		return ModeAuto
	}
}

type PIDStats struct {
	PID               uint32     `json:"pid"`
	Comm              string     `json:"comm,omitempty"`
	KernelLaunches    uint64     `json:"kernelLaunches"`
	GraphLaunches     uint64     `json:"graphLaunches,omitempty"`
	AvgBlockThreads   float64    `json:"avgBlockThreads,omitempty"`
	AvgGridBlocks     float64    `json:"avgGridBlocks,omitempty"`
	AvgSharedMemBytes float64    `json:"avgSharedMemBytes,omitempty"`
	LastGridX         uint32     `json:"lastGridX,omitempty"`
	LastBlockX        uint32     `json:"lastBlockX,omitempty"`
	AvgLaunchGapUS    float64    `json:"avgLaunchGapUs"`
	P50LaunchGapUS    float64    `json:"p50LaunchGapUs,omitempty"`
	P95LaunchGapUS    float64    `json:"p95LaunchGapUs,omitempty"`
	P99LaunchGapUS    float64    `json:"p99LaunchGapUs,omitempty"`
	MaxLaunchGapUS    float64    `json:"maxLaunchGapUs"`
	MemcpyH2DCalls    uint64     `json:"memcpyH2dCalls"`
	MemcpyH2DBytes    uint64     `json:"memcpyH2dBytes"`
	MemcpyD2HCalls    uint64     `json:"memcpyD2hCalls"`
	MemcpyD2HBytes    uint64     `json:"memcpyD2hBytes"`
	MemAllocCalls     uint64     `json:"memAllocCalls"`
	MemAllocBytes     uint64     `json:"memAllocBytes"`
	SyncCalls         uint64     `json:"syncCalls"`
	SyncStallMS       float64    `json:"syncStallMs"`
	P95SyncStallMS    float64    `json:"p95SyncStallMs,omitempty"`
	MaxSyncStallMS    float64    `json:"maxSyncStallMs"`
	SchedSwitches     uint64     `json:"schedSwitches"`
	OffCPUMS          float64    `json:"offCpuMs"`
	LaunchGapHist     [16]uint64 `json:"launchGapHist,omitempty"`
	SyncStallHist     [16]uint64 `json:"syncStallHist,omitempty"`
}

type Summary struct {
	Enabled           bool       `json:"enabled"`
	Active            bool       `json:"active"`
	Mode              Mode       `json:"mode"`
	Provenance        string     `json:"provenance"`
	StatusDetail      string     `json:"statusDetail,omitempty"`
	AttachedLibCUDA   string     `json:"attachedLibCuda,omitempty"`
	AttachedLibCUDART string     `json:"attachedLibCudart,omitempty"`
	HookedSymbols     []string   `json:"hookedSymbols,omitempty"`
	TargetPIDs        []uint32   `json:"targetPids,omitempty"`
	DurationMS        float64    `json:"durationMs"`
	KernelLaunches    uint64     `json:"kernelLaunches"`
	GraphLaunches     uint64     `json:"graphLaunches,omitempty"`
	LaunchesPerSec    float64    `json:"launchesPerSec"`
	AvgBlockThreads   float64    `json:"avgBlockThreads,omitempty"`
	AvgGridBlocks     float64    `json:"avgGridBlocks,omitempty"`
	AvgSharedMemBytes float64    `json:"avgSharedMemBytes,omitempty"`
	LastGridX         uint32     `json:"lastGridX,omitempty"`
	LastBlockX        uint32     `json:"lastBlockX,omitempty"`
	AvgLaunchGapUS    float64    `json:"avgLaunchGapUs"`
	P50LaunchGapUS    float64    `json:"p50LaunchGapUs,omitempty"`
	P95LaunchGapUS    float64    `json:"p95LaunchGapUs,omitempty"`
	P99LaunchGapUS    float64    `json:"p99LaunchGapUs,omitempty"`
	MaxLaunchGapUS    float64    `json:"maxLaunchGapUs"`
	MemcpyH2DCalls    uint64     `json:"memcpyH2dCalls"`
	MemcpyH2DMiB      float64    `json:"memcpyH2dMib"`
	MemcpyD2HCalls    uint64     `json:"memcpyD2hCalls"`
	MemcpyD2HMiB      float64    `json:"memcpyD2hMib"`
	MemAllocCalls     uint64     `json:"memAllocCalls"`
	MemAllocMiB       float64    `json:"memAllocMib"`
	SyncCalls         uint64     `json:"syncCalls"`
	SyncStallMS       float64    `json:"syncStallMs"`
	SyncStallPct      float64    `json:"syncStallPct"`
	P95SyncStallMS    float64    `json:"p95SyncStallMs,omitempty"`
	MaxSyncStallMS    float64    `json:"maxSyncStallMs"`
	SchedSwitches     uint64     `json:"schedSwitches"`
	OffCPUMS          float64    `json:"offCpuMs"`
	HostStarvationPct float64    `json:"hostStarvationPct"`
	LaunchGapHist     [16]uint64 `json:"launchGapHist,omitempty"`
	SyncStallHist     [16]uint64 `json:"syncStallHist,omitempty"`
	Processes         []PIDStats `json:"processes,omitempty"`
}

// rawProcessEntry is an internal architecture-neutral mirror of BPF tgid_stats.
type rawProcessEntry struct {
	PID               uint32
	Comm              string
	KernelLaunches    uint64
	GraphLaunches     uint64
	LaunchGapSumNs    uint64
	LaunchGapCount    uint64
	MaxLaunchGapNs    uint64
	BlockThreadsSum   uint64
	GridBlocksSum     uint64
	SharedMemBytesSum uint64
	LastGridX         uint32
	LastBlockX        uint32
	MemcpyH2DCalls    uint64
	MemcpyH2DBytes    uint64
	MemcpyD2HCalls    uint64
	MemcpyD2HBytes    uint64
	MemAllocCalls     uint64
	MemAllocBytes     uint64
	SyncCalls         uint64
	SyncStallSumNs    uint64
	MaxSyncStallNs    uint64
	SchedSwitches     uint64
	OffCPUSumNs       uint64
	LaunchGapHist     [16]uint64
	SyncStallHist     [16]uint64
}

func (r rawProcessEntry) hasCUDAActivity() bool {
	return r.KernelLaunches > 0 ||
		r.MemcpyH2DCalls > 0 ||
		r.MemcpyD2HCalls > 0 ||
		r.MemAllocCalls > 0 ||
		r.SyncCalls > 0
}

func filterAndAggregateEntries(
	entries []rawProcessEntry,
	targetSet map[uint32]struct{},
	selfPID uint32,
	ppidLookup func(uint32) uint32,
	durationMS float64,
	sum *Summary,
) {
	var selected []rawProcessEntry

	if len(targetSet) == 0 {
		for _, e := range entries {
			if e.PID == selfPID {
				continue
			}
			selected = append(selected, e)
		}
	} else {
		isLineageMatch := func(pid uint32) bool {
			if _, ok := targetSet[pid]; ok {
				return true
			}
			if ppidLookup == nil {
				return false
			}
			cur := pid
			for depth := 0; depth < 12; depth++ {
				parent := ppidLookup(cur)
				if parent <= 1 || parent == cur {
					return false
				}
				if _, ok := targetSet[parent]; ok {
					return true
				}
				cur = parent
			}
			return false
		}

		hasDirectCUDA := false
		for _, e := range entries {
			if isLineageMatch(e.PID) {
				selected = append(selected, e)
				if e.hasCUDAActivity() {
					hasDirectCUDA = true
				}
			}
		}

		// Fallback if target wrapper spawned short-lived worker children whose /proc
		// entries exited before Stop() could inspect PPID lineage.
		if !hasDirectCUDA {
			var cudaFallback []rawProcessEntry
			for _, e := range entries {
				if e.PID != selfPID && e.hasCUDAActivity() {
					cudaFallback = append(cudaFallback, e)
				}
			}
			if len(cudaFallback) > 0 {
				seen := make(map[uint32]struct{}, len(selected)+len(cudaFallback))
				for _, e := range selected {
					seen[e.PID] = struct{}{}
				}
				for _, e := range cudaFallback {
					if _, exists := seen[e.PID]; !exists {
						selected = append(selected, e)
						seen[e.PID] = struct{}{}
					}
				}
			}
		}
	}

	sort.Slice(selected, func(i, j int) bool {
		if selected[i].KernelLaunches != selected[j].KernelLaunches {
			return selected[i].KernelLaunches > selected[j].KernelLaunches
		}
		return selected[i].PID < selected[j].PID
	})

	var (
		totalLaunches      uint64
		totalGraphLaunches uint64
		totalGapSumNs      uint64
		totalGapCount      uint64
		maxLaunchGapNs     uint64
		totalBlockThreads  uint64
		totalGridBlocks    uint64
		totalSharedMem     uint64
		lastGridX          uint32
		lastBlockX         uint32
		totalH2DCalls      uint64
		totalH2DBytes      uint64
		totalD2HCalls      uint64
		totalD2HBytes      uint64
		totalAllocCalls    uint64
		totalAllocBytes    uint64
		totalSyncCalls     uint64
		totalSyncStallNs   uint64
		maxSyncStallNs     uint64
		totalSchedSwitch   uint64
		totalOffCPUNs      uint64
		totalLaunchHist    [16]uint64
		totalSyncHist      [16]uint64
	)

	procs := make([]PIDStats, 0, len(selected))
	for _, e := range selected {
		var avgGapUS float64
		if e.LaunchGapCount > 0 {
			avgGapUS = float64(e.LaunchGapSumNs) / float64(e.LaunchGapCount) / 1e3
		}
		maxGapUS := float64(e.MaxLaunchGapNs) / 1e3
		syncMS := float64(e.SyncStallSumNs) / 1e6
		maxSyncMS := float64(e.MaxSyncStallNs) / 1e6
		offCPUMS := float64(e.OffCPUSumNs) / 1e6

		var avgBlockThreads, avgGridBlocks, avgSharedMem float64
		if e.KernelLaunches > 0 {
			avgBlockThreads = float64(e.BlockThreadsSum) / float64(e.KernelLaunches)
			avgGridBlocks = float64(e.GridBlocksSum) / float64(e.KernelLaunches)
			avgSharedMem = float64(e.SharedMemBytesSum) / float64(e.KernelLaunches)
		}

		p50Gap := Log2HistQuantileUS(e.LaunchGapHist, 0.50, avgGapUS, maxGapUS)
		p95Gap := Log2HistQuantileUS(e.LaunchGapHist, 0.95, avgGapUS, maxGapUS)
		p99Gap := Log2HistQuantileUS(e.LaunchGapHist, 0.99, avgGapUS, maxGapUS)
		avgSyncUS := 0.0
		if e.SyncCalls > 0 {
			avgSyncUS = (syncMS * 1000.0) / float64(e.SyncCalls)
		}
		p95SyncMS := Log2HistQuantileUS(e.SyncStallHist, 0.95, avgSyncUS, maxSyncMS*1000.0) / 1000.0

		procs = append(procs, PIDStats{
			PID:               e.PID,
			Comm:              e.Comm,
			KernelLaunches:    e.KernelLaunches,
			GraphLaunches:     e.GraphLaunches,
			AvgBlockThreads:   avgBlockThreads,
			AvgGridBlocks:     avgGridBlocks,
			AvgSharedMemBytes: avgSharedMem,
			LastGridX:         e.LastGridX,
			LastBlockX:        e.LastBlockX,
			AvgLaunchGapUS:    avgGapUS,
			P50LaunchGapUS:    p50Gap,
			P95LaunchGapUS:    p95Gap,
			P99LaunchGapUS:    p99Gap,
			MaxLaunchGapUS:    maxGapUS,
			MemcpyH2DCalls:    e.MemcpyH2DCalls,
			MemcpyH2DBytes:    e.MemcpyH2DBytes,
			MemcpyD2HCalls:    e.MemcpyD2HCalls,
			MemcpyD2HBytes:    e.MemcpyD2HBytes,
			MemAllocCalls:     e.MemAllocCalls,
			MemAllocBytes:     e.MemAllocBytes,
			SyncCalls:         e.SyncCalls,
			SyncStallMS:       syncMS,
			P95SyncStallMS:    p95SyncMS,
			MaxSyncStallMS:    maxSyncMS,
			SchedSwitches:     e.SchedSwitches,
			OffCPUMS:          offCPUMS,
			LaunchGapHist:     e.LaunchGapHist,
			SyncStallHist:     e.SyncStallHist,
		})

		totalLaunches += e.KernelLaunches
		totalGraphLaunches += e.GraphLaunches
		totalGapSumNs += e.LaunchGapSumNs
		totalGapCount += e.LaunchGapCount
		if e.MaxLaunchGapNs > maxLaunchGapNs {
			maxLaunchGapNs = e.MaxLaunchGapNs
		}
		totalBlockThreads += e.BlockThreadsSum
		totalGridBlocks += e.GridBlocksSum
		totalSharedMem += e.SharedMemBytesSum
		if e.LastGridX > 0 {
			lastGridX = e.LastGridX
		}
		if e.LastBlockX > 0 {
			lastBlockX = e.LastBlockX
		}
		totalH2DCalls += e.MemcpyH2DCalls
		totalH2DBytes += e.MemcpyH2DBytes
		totalD2HCalls += e.MemcpyD2HCalls
		totalD2HBytes += e.MemcpyD2HBytes
		totalAllocCalls += e.MemAllocCalls
		totalAllocBytes += e.MemAllocBytes
		totalSyncCalls += e.SyncCalls
		totalSyncStallNs += e.SyncStallSumNs
		if e.MaxSyncStallNs > maxSyncStallNs {
			maxSyncStallNs = e.MaxSyncStallNs
		}
		totalSchedSwitch += e.SchedSwitches
		totalOffCPUNs += e.OffCPUSumNs
		for b := 0; b < 16; b++ {
			totalLaunchHist[b] += e.LaunchGapHist[b]
			totalSyncHist[b] += e.SyncStallHist[b]
		}
	}

	const bytesPerMiB = 1024.0 * 1024.0

	sum.DurationMS = math.Max(0, durationMS)
	sum.KernelLaunches = totalLaunches
	sum.GraphLaunches = totalGraphLaunches
	if sum.DurationMS > 0 {
		sum.LaunchesPerSec = float64(totalLaunches) / (sum.DurationMS / 1000.0)
	}
	if totalLaunches > 0 {
		sum.AvgBlockThreads = float64(totalBlockThreads) / float64(totalLaunches)
		sum.AvgGridBlocks = float64(totalGridBlocks) / float64(totalLaunches)
		sum.AvgSharedMemBytes = float64(totalSharedMem) / float64(totalLaunches)
	}
	sum.LastGridX = lastGridX
	sum.LastBlockX = lastBlockX
	if totalGapCount > 0 {
		sum.AvgLaunchGapUS = float64(totalGapSumNs) / float64(totalGapCount) / 1e3
	}
	sum.MaxLaunchGapUS = float64(maxLaunchGapNs) / 1e3
	sum.P50LaunchGapUS = Log2HistQuantileUS(totalLaunchHist, 0.50, sum.AvgLaunchGapUS, sum.MaxLaunchGapUS)
	sum.P95LaunchGapUS = Log2HistQuantileUS(totalLaunchHist, 0.95, sum.AvgLaunchGapUS, sum.MaxLaunchGapUS)
	sum.P99LaunchGapUS = Log2HistQuantileUS(totalLaunchHist, 0.99, sum.AvgLaunchGapUS, sum.MaxLaunchGapUS)
	sum.LaunchGapHist = totalLaunchHist
	sum.SyncStallHist = totalSyncHist
	sum.MemcpyH2DCalls = totalH2DCalls
	sum.MemcpyH2DMiB = float64(totalH2DBytes) / bytesPerMiB
	sum.MemcpyD2HCalls = totalD2HCalls
	sum.MemcpyD2HMiB = float64(totalD2HBytes) / bytesPerMiB
	sum.MemAllocCalls = totalAllocCalls
	sum.MemAllocMiB = float64(totalAllocBytes) / bytesPerMiB
	sum.SyncCalls = totalSyncCalls
	sum.SyncStallMS = float64(totalSyncStallNs) / 1e6
	sum.MaxSyncStallMS = float64(maxSyncStallNs) / 1e6
	avgSyncUS := 0.0
	if totalSyncCalls > 0 {
		avgSyncUS = (sum.SyncStallMS * 1000.0) / float64(totalSyncCalls)
	}
	sum.P95SyncStallMS = Log2HistQuantileUS(totalSyncHist, 0.95, avgSyncUS, sum.MaxSyncStallMS*1000.0) / 1000.0
	sum.SchedSwitches = totalSchedSwitch
	sum.OffCPUMS = float64(totalOffCPUNs) / 1e6
	if sum.DurationMS > 0 {
		sum.SyncStallPct = clampPct((sum.SyncStallMS / sum.DurationMS) * 100.0)
		sum.HostStarvationPct = clampPct((sum.OffCPUMS / sum.DurationMS) * 100.0)
	}
	if len(procs) > 0 {
		sum.Processes = procs
	}
}

// Log2HistQuantileUS estimates the q-quantile (0..1) in microseconds from a 16-bucket log2 histogram.
func Log2HistQuantileUS(hist [16]uint64, q float64, fallbackAvgUS, maxUS float64) float64 {
	var total uint64
	for _, c := range hist {
		total += c
	}
	if total == 0 {
		if fallbackAvgUS <= 0 {
			return 0
		}
		if q >= 0.95 && maxUS > fallbackAvgUS {
			return math.Min(maxUS, fallbackAvgUS+(maxUS-fallbackAvgUS)*q)
		}
		return fallbackAvgUS
	}
	if q < 0 {
		q = 0
	}
	if q > 1 {
		q = 1
	}
	target := float64(total) * q
	var cum uint64
	for b, c := range hist {
		if c == 0 {
			continue
		}
		prevCum := cum
		cum += c
		if float64(cum) >= target {
			low := float64(uint64(1) << uint(b))
			if b == 0 {
				low = 0
			}
			high := float64(uint64(1) << uint(b+1))
			frac := (target - float64(prevCum)) / float64(c)
			est := low + frac*(high-low)
			if maxUS > 0 && est > maxUS {
				return maxUS
			}
			return math.Round(est*100) / 100
		}
	}
	return maxUS
}

// PerfettoEvent represents a single event in the Chrome Trace Event / Perfetto JSON format.
type PerfettoEvent struct {
	Name string         `json:"name"`
	Cat  string         `json:"cat"`
	Ph   string         `json:"ph"`
	Ts   float64        `json:"ts"`
	Dur  float64        `json:"dur,omitempty"`
	PID  uint32         `json:"pid"`
	TID  uint32         `json:"tid"`
	Args map[string]any `json:"args,omitempty"`
}

// BuildPerfettoTrace generates a Chrome Trace Event / Perfetto JSON document from an eBPF Summary.
func BuildPerfettoTrace(sum Summary) ([]byte, error) {
	events := make([]PerfettoEvent, 0, 64)
	procs := sum.Processes
	if len(procs) == 0 && sum.KernelLaunches > 0 {
		procs = []PIDStats{{
			PID:               1,
			Comm:              "cuda-workload",
			KernelLaunches:    sum.KernelLaunches,
			GraphLaunches:     sum.GraphLaunches,
			AvgBlockThreads:   sum.AvgBlockThreads,
			AvgGridBlocks:     sum.AvgGridBlocks,
			AvgSharedMemBytes: sum.AvgSharedMemBytes,
			AvgLaunchGapUS:    sum.AvgLaunchGapUS,
			P95LaunchGapUS:    sum.P95LaunchGapUS,
			MaxLaunchGapUS:    sum.MaxLaunchGapUS,
			SyncCalls:         sum.SyncCalls,
			SyncStallMS:       sum.SyncStallMS,
			OffCPUMS:          sum.OffCPUMS,
			SchedSwitches:     sum.SchedSwitches,
		}}
	}

	var cursorUS float64
	for _, p := range procs {
		events = append(events, PerfettoEvent{
			Name: "process_name",
			Cat:  "__metadata",
			Ph:   "M",
			PID:  p.PID,
			TID:  p.PID,
			Args: map[string]any{"name": fmt.Sprintf("%s (pid=%d)", p.Comm, p.PID)},
		})

		sampleLaunches := int(p.KernelLaunches)
		if sampleLaunches > 32 {
			sampleLaunches = 32
		}
		gapUS := p.AvgLaunchGapUS
		if gapUS <= 0 {
			gapUS = 15.0
		}
		for i := 0; i < sampleLaunches; i++ {
			events = append(events, PerfettoEvent{
				Name: "cuLaunchKernel",
				Cat:  "cuda.driver",
				Ph:   "X",
				Ts:   cursorUS,
				Dur:  math.Max(4.0, gapUS*0.65),
				PID:  p.PID,
				TID:  p.PID,
				Args: map[string]any{
					"avgGridBlocks":     p.AvgGridBlocks,
					"avgBlockThreads":   p.AvgBlockThreads,
					"avgSharedMemBytes": p.AvgSharedMemBytes,
					"lastGridX":         p.LastGridX,
					"lastBlockX":        p.LastBlockX,
				},
			})
			cursorUS += gapUS
		}
		if p.SyncStallMS > 0 {
			events = append(events, PerfettoEvent{
				Name: "cuStreamSynchronize",
				Cat:  "cuda.sync",
				Ph:   "X",
				Ts:   cursorUS,
				Dur:  p.SyncStallMS * 1000.0,
				PID:  p.PID,
				TID:  p.PID,
				Args: map[string]any{
					"syncCalls":      p.SyncCalls,
					"p95SyncStallMs": p.P95SyncStallMS,
					"maxSyncStallMs": p.MaxSyncStallMS,
				},
			})
			cursorUS += p.SyncStallMS * 1000.0
		}
		if p.OffCPUMS > 0 {
			events = append(events, PerfettoEvent{
				Name: "sched_switch (off-CPU)",
				Cat:  "linux.sched",
				Ph:   "X",
				Ts:   cursorUS,
				Dur:  p.OffCPUMS * 1000.0,
				PID:  p.PID,
				TID:  p.PID + 1,
				Args: map[string]any{
					"schedSwitches": p.SchedSwitches,
					"offCpuMs":      p.OffCPUMS,
				},
			})
			cursorUS += p.OffCPUMS * 1000.0
		}
	}

	doc := map[string]any{
		"displayTimeUnit": "ms",
		"metadata": map[string]any{
			"generator":      "gpuMode eBPF Host-to-CUDA Tracer",
			"provenance":     sum.Provenance,
			"durationMs":     sum.DurationMS,
			"kernelLaunches": sum.KernelLaunches,
			"graphLaunches":  sum.GraphLaunches,
			"p50LaunchGapUs": sum.P50LaunchGapUS,
			"p95LaunchGapUs": sum.P95LaunchGapUS,
			"p99LaunchGapUs": sum.P99LaunchGapUS,
		},
		"traceEvents": events,
	}
	return json.MarshalIndent(doc, "", "  ")
}

func clampPct(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 100 {
		return 100
	}
	return v
}

func int8CommToString(raw [16]int8) string {
	buf := make([]byte, 0, len(raw))
	for _, c := range raw {
		if c == 0 {
			break
		}
		buf = append(buf, byte(c))
	}
	return strings.TrimSpace(string(buf))
}

func sortedUint32Keys(m map[uint32]struct{}) []uint32 {
	if len(m) == 0 {
		return nil
	}
	out := make([]uint32, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
