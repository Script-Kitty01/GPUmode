package ebpftrace

import (
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
	PID            uint32  `json:"pid"`
	Comm           string  `json:"comm,omitempty"`
	KernelLaunches uint64  `json:"kernelLaunches"`
	AvgLaunchGapUS float64 `json:"avgLaunchGapUs"`
	MaxLaunchGapUS float64 `json:"maxLaunchGapUs"`
	MemcpyH2DCalls uint64  `json:"memcpyH2dCalls"`
	MemcpyH2DBytes uint64  `json:"memcpyH2dBytes"`
	MemcpyD2HCalls uint64  `json:"memcpyD2hCalls"`
	MemcpyD2HBytes uint64  `json:"memcpyD2hBytes"`
	MemAllocCalls  uint64  `json:"memAllocCalls"`
	MemAllocBytes  uint64  `json:"memAllocBytes"`
	SyncCalls      uint64  `json:"syncCalls"`
	SyncStallMS    float64 `json:"syncStallMs"`
	MaxSyncStallMS float64 `json:"maxSyncStallMs"`
	SchedSwitches  uint64  `json:"schedSwitches"`
	OffCPUMS       float64 `json:"offCpuMs"`
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
	LaunchesPerSec    float64    `json:"launchesPerSec"`
	AvgLaunchGapUS    float64    `json:"avgLaunchGapUs"`
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
	MaxSyncStallMS    float64    `json:"maxSyncStallMs"`
	SchedSwitches     uint64     `json:"schedSwitches"`
	OffCPUMS          float64    `json:"offCpuMs"`
	HostStarvationPct float64    `json:"hostStarvationPct"`
	Processes         []PIDStats `json:"processes,omitempty"`
}

// rawProcessEntry is an internal architecture-neutral mirror of BPF tgid_stats.
type rawProcessEntry struct {
	PID            uint32
	Comm           string
	KernelLaunches uint64
	LaunchGapSumNs uint64
	LaunchGapCount uint64
	MaxLaunchGapNs uint64
	MemcpyH2DCalls uint64
	MemcpyH2DBytes uint64
	MemcpyD2HCalls uint64
	MemcpyD2HBytes uint64
	MemAllocCalls  uint64
	MemAllocBytes  uint64
	SyncCalls      uint64
	SyncStallSumNs uint64
	MaxSyncStallNs uint64
	SchedSwitches  uint64
	OffCPUSumNs    uint64
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
		totalLaunches    uint64
		totalGapSumNs    uint64
		totalGapCount    uint64
		maxLaunchGapNs   uint64
		totalH2DCalls    uint64
		totalH2DBytes    uint64
		totalD2HCalls    uint64
		totalD2HBytes    uint64
		totalAllocCalls  uint64
		totalAllocBytes  uint64
		totalSyncCalls   uint64
		totalSyncStallNs uint64
		maxSyncStallNs   uint64
		totalSchedSwitch uint64
		totalOffCPUNs    uint64
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

		procs = append(procs, PIDStats{
			PID:            e.PID,
			Comm:           e.Comm,
			KernelLaunches: e.KernelLaunches,
			AvgLaunchGapUS: avgGapUS,
			MaxLaunchGapUS: maxGapUS,
			MemcpyH2DCalls: e.MemcpyH2DCalls,
			MemcpyH2DBytes: e.MemcpyH2DBytes,
			MemcpyD2HCalls: e.MemcpyD2HCalls,
			MemcpyD2HBytes: e.MemcpyD2HBytes,
			MemAllocCalls:  e.MemAllocCalls,
			MemAllocBytes:  e.MemAllocBytes,
			SyncCalls:      e.SyncCalls,
			SyncStallMS:    syncMS,
			MaxSyncStallMS: maxSyncMS,
			SchedSwitches:  e.SchedSwitches,
			OffCPUMS:       offCPUMS,
		})

		totalLaunches += e.KernelLaunches
		totalGapSumNs += e.LaunchGapSumNs
		totalGapCount += e.LaunchGapCount
		if e.MaxLaunchGapNs > maxLaunchGapNs {
			maxLaunchGapNs = e.MaxLaunchGapNs
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
	}

	const bytesPerMiB = 1024.0 * 1024.0

	sum.DurationMS = math.Max(0, durationMS)
	sum.KernelLaunches = totalLaunches
	if sum.DurationMS > 0 {
		sum.LaunchesPerSec = float64(totalLaunches) / (sum.DurationMS / 1000.0)
	}
	if totalGapCount > 0 {
		sum.AvgLaunchGapUS = float64(totalGapSumNs) / float64(totalGapCount) / 1e3
	}
	sum.MaxLaunchGapUS = float64(maxLaunchGapNs) / 1e3
	sum.MemcpyH2DCalls = totalH2DCalls
	sum.MemcpyH2DMiB = float64(totalH2DBytes) / bytesPerMiB
	sum.MemcpyD2HCalls = totalD2HCalls
	sum.MemcpyD2HMiB = float64(totalD2HBytes) / bytesPerMiB
	sum.MemAllocCalls = totalAllocCalls
	sum.MemAllocMiB = float64(totalAllocBytes) / bytesPerMiB
	sum.SyncCalls = totalSyncCalls
	sum.SyncStallMS = float64(totalSyncStallNs) / 1e6
	sum.MaxSyncStallMS = float64(maxSyncStallNs) / 1e6
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
