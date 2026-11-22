//go:build linux && amd64

package ebpftrace

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/rlimit"
	"golang.org/x/sys/unix"
)

const unprivilegedStatusMsg = "eBPF requires CAP_BPF and CAP_PERFMON (run './run.sh setcap' or execute with sudo)"

// Session manages an active or gracefully-degraded eBPF tracing window.
type Session struct {
	mu            sync.Mutex
	stopped       bool
	cachedSummary Summary

	mode          Mode
	enabled       bool
	active        bool
	provenance    string
	statusDetail  string
	libCudaPath   string
	libCudartPath string
	hookedSymbols []string
	targetPIDSet  map[uint32]struct{}
	startedAt     time.Time

	objs  *cudaTraceObjects
	links []link.Link
}

// StartSession starts an eBPF CUDA + scheduler tracing session according to mode.
func StartSession(mode Mode, targetPIDs []int) (*Session, error) {
	normMode := ParseMode(string(mode))
	return startSessionInternal(normMode, targetPIDs, normMode == ModeOn)
}

// ProbeAvailability checks whether eBPF CUDA tracing can attach on the current host
// and returns an immediate Summary describing capabilities, libraries, and hooked symbols.
func ProbeAvailability(mode Mode) Summary {
	normMode := ParseMode(string(mode))
	sess, _ := startSessionInternal(normMode, nil, false)
	if sess == nil {
		return Summary{
			Enabled:    normMode != ModeOff,
			Active:     false,
			Mode:       normMode,
			Provenance: ProvenanceUnavailableUnsupported,
		}
	}
	return sess.Stop()
}

func startSessionInternal(mode Mode, targetPIDs []int, failHard bool) (*Session, error) {
	targetSet := make(map[uint32]struct{}, len(targetPIDs))
	for _, pid := range targetPIDs {
		if pid > 0 {
			targetSet[uint32(pid)] = struct{}{}
		}
	}

	now := time.Now()
	if mode == ModeOff {
		return &Session{
			mode:         ModeOff,
			enabled:      false,
			active:       false,
			provenance:   ProvenanceDisabled,
			statusDetail: "eBPF tracing disabled",
			targetPIDSet: targetSet,
			startedAt:    now,
		}, nil
	}

	libCudaPath, libCudartPath := DiscoverCUDALibraries(targetPIDs...)
	if libCudaPath == "" && libCudartPath == "" {
		detail := "no CUDA driver (libcuda.so.1) or runtime (libcudart.so) library found on host"
		if failHard {
			return nil, fmt.Errorf("ebpftrace: %s", detail)
		}
		return &Session{
			mode:         mode,
			enabled:      true,
			active:       false,
			provenance:   ProvenanceUnavailableUnsupported,
			statusDetail: detail,
			targetPIDSet: targetSet,
			startedAt:    now,
		}, nil
	}

	// Best-effort memlock removal for kernels < 5.11; ignore errors when unprivileged.
	_ = rlimit.RemoveMemlock()

	// Capability-elevated binaries (setcap) start with PR_SET_DUMPABLE=0, which blocks
	// cilium/ebpf's vDSO /proc/self/mem kernel-version probe with EACCES.
	// Temporarily enable dumpable across object load (and populate KernelVersion from uname
	// as a fallback) so rootless setcap execution attaches without CAP_SYS_PTRACE.
	restoreDumpable := enableTempDumpable()
	var objs cudaTraceObjects
	loadErr := loadCudaTraceObjectsWithFallback(&objs)
	restoreDumpable()
	if loadErr != nil {
		return handleSessionStartFailure(mode, failHard, loadErr, libCudaPath, libCudartPath, targetSet, now)
	}

	// Pre-seed target TGIDs so sched_switch tracks target processes even before first CUDA call.
	for tgid := range targetSet {
		k := tgid
		v := uint8(1)
		_ = objs.TargetTgids.Put(&k, &v)
	}

	var (
		links         []link.Link
		hookedSymbols []string
		attachErr     error
	)

	// Prefer attaching directly to libcuda.so.1 to capture all Driver + Runtime + PyTorch/GGML
	// GPU work without 2x double-counting libcudart -> libcuda wrapper calls.
	if libCudaPath != "" {
		cudaLinks, cudaSyms, err := attachLibCudaProbes(libCudaPath, &objs)
		if err == nil {
			links = append(links, cudaLinks...)
			hookedSymbols = append(hookedSymbols, cudaSyms...)
		} else {
			attachErr = err
		}
	}

	// Only fall back to libcudart.so uprobes if libcuda.so.1 was absent or could not attach
	// (and the failure was not an EPERM/EACCES capability denial).
	if len(hookedSymbols) == 0 && libCudartPath != "" && !isPermissionError(attachErr) {
		rtLinks, rtSyms, err := attachLibCudartFallbackProbes(libCudartPath, &objs)
		if err == nil {
			links = append(links, rtLinks...)
			hookedSymbols = append(hookedSymbols, rtSyms...)
			attachErr = nil
		} else if attachErr == nil {
			attachErr = err
		}
	}

	if len(hookedSymbols) == 0 {
		closeLinks(links)
		_ = objs.Close()
		if attachErr == nil {
			attachErr = errors.New("failed to attach required CUDA launch uprobes")
		}
		return handleSessionStartFailure(mode, failHard, attachErr, libCudaPath, libCudartPath, targetSet, now)
	}

	// Best-effort attach to kernel scheduler tracepoint for off-CPU / host starvation tracking.
	if tpLink, err := link.AttachRawTracepoint(link.RawTracepointOptions{
		Name:    "sched_switch",
		Program: objs.HandleSchedSwitch,
	}); err == nil {
		links = append(links, tpLink)
		hookedSymbols = append(hookedSymbols, "sched/sched_switch")
	}

	primaryLib := libCudaPath
	if primaryLib == "" {
		primaryLib = libCudartPath
	}

	return &Session{
		mode:          mode,
		enabled:       true,
		active:        true,
		provenance:    ProvenanceMeasuredEBPF,
		statusDetail:  fmt.Sprintf("attached %d eBPF probes (%s)", len(hookedSymbols), filepath.Base(primaryLib)),
		libCudaPath:   libCudaPath,
		libCudartPath: libCudartPath,
		hookedSymbols: hookedSymbols,
		targetPIDSet:  targetSet,
		startedAt:     time.Now(),
		objs:          &objs,
		links:         links,
	}, nil
}

func handleSessionStartFailure(
	mode Mode,
	failHard bool,
	err error,
	libCudaPath string,
	libCudartPath string,
	targetSet map[uint32]struct{},
	startedAt time.Time,
) (*Session, error) {
	if isPermissionError(err) {
		if failHard {
			return nil, fmt.Errorf("ebpftrace: %s: %w", unprivilegedStatusMsg, err)
		}
		return &Session{
			mode:          mode,
			enabled:       true,
			active:        false,
			provenance:    ProvenanceUnavailableUnprivileged,
			statusDetail:  unprivilegedStatusMsg,
			libCudaPath:   libCudaPath,
			libCudartPath: libCudartPath,
			targetPIDSet:  targetSet,
			startedAt:     startedAt,
		}, nil
	}

	detail := fmt.Sprintf("eBPF unsupported or attach failed: %v", err)
	if failHard {
		return nil, fmt.Errorf("ebpftrace: %s", detail)
	}
	return &Session{
		mode:          mode,
		enabled:       true,
		active:        false,
		provenance:    ProvenanceUnavailableUnsupported,
		statusDetail:  detail,
		libCudaPath:   libCudaPath,
		libCudartPath: libCudartPath,
		targetPIDSet:  targetSet,
		startedAt:     startedAt,
	}, nil
}

type uprobeSpec struct {
	symbol string
	prog   *ebpf.Program
}

func attachLibCudaProbes(libPath string, objs *cudaTraceObjects) ([]link.Link, []string, error) {
	ex, err := link.OpenExecutable(libPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open %s: %w", libPath, err)
	}

	// Primary launch probe: cuLaunchKernel must attach for libcuda tracing to be valid.
	primaryLink, err := ex.Uprobe("cuLaunchKernel", objs.HandleCudaLaunch, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("attach uprobe cuLaunchKernel on %s: %w", libPath, err)
	}

	links := []link.Link{primaryLink}
	hooked := []string{"cuLaunchKernel"}

	optionalUprobes := []uprobeSpec{
		// Extended & per-thread-default-stream launch entrypoints (independent driver slots)
		{symbol: "cuLaunchKernelEx", prog: objs.HandleCudaLaunch},
		{symbol: "cuLaunchKernel_ptsz", prog: objs.HandleCudaLaunch},
		{symbol: "cuLaunchKernelEx_ptsz", prog: objs.HandleCudaLaunch},
		// CUDA Graph launch entrypoints
		{symbol: "cuGraphLaunch", prog: objs.HandleGraphLaunch},
		{symbol: "cuGraphLaunch_ptsz", prog: objs.HandleGraphLaunch},
		// Host -> Device transfers
		{symbol: "cuMemcpyHtoD_v2", prog: objs.HandleMemcpyH2d},
		{symbol: "cuMemcpyHtoDAsync_v2", prog: objs.HandleMemcpyH2d},
		{symbol: "cuMemcpyHtoD_v2_ptds", prog: objs.HandleMemcpyH2d},
		{symbol: "cuMemcpyHtoDAsync_v2_ptsz", prog: objs.HandleMemcpyH2d},
		// Device -> Host transfers
		{symbol: "cuMemcpyDtoH_v2", prog: objs.HandleMemcpyD2h},
		{symbol: "cuMemcpyDtoHAsync_v2", prog: objs.HandleMemcpyD2h},
		{symbol: "cuMemcpyDtoH_v2_ptds", prog: objs.HandleMemcpyD2h},
		{symbol: "cuMemcpyDtoHAsync_v2_ptsz", prog: objs.HandleMemcpyD2h},
		// Device memory allocations
		{symbol: "cuMemAlloc_v2", prog: objs.HandleMemAlloc},
		{symbol: "cuMemAllocAsync", prog: objs.HandleMemAlloc},
		{symbol: "cuMemAllocAsync_ptsz", prog: objs.HandleMemAlloc},
	}

	for _, spec := range optionalUprobes {
		l, uErr := ex.Uprobe(spec.symbol, spec.prog, nil)
		if uErr == nil {
			links = append(links, l)
			hooked = append(hooked, spec.symbol)
		}
	}

	syncSymbols := []string{
		"cuStreamSynchronize",
		"cuStreamSynchronize_ptsz",
		"cuCtxSynchronize",
		"cuCtxSynchronize_v2",
	}
	for _, sym := range syncSymbols {
		up, uErr := ex.Uprobe(sym, objs.HandleSyncEnter, nil)
		if uErr != nil {
			continue
		}
		uret, rErr := ex.Uretprobe(sym, objs.HandleSyncExit, nil)
		if rErr != nil {
			_ = up.Close()
			continue
		}
		links = append(links, up, uret)
		hooked = append(hooked, sym)
	}

	return links, hooked, nil
}

func attachLibCudartFallbackProbes(libPath string, objs *cudaTraceObjects) ([]link.Link, []string, error) {
	ex, err := link.OpenExecutable(libPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open %s: %w", libPath, err)
	}

	primaryLink, err := ex.Uprobe("cudaLaunchKernel", objs.HandleCudaLaunch, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("attach uprobe cudaLaunchKernel on %s: %w", libPath, err)
	}

	links := []link.Link{primaryLink}
	hooked := []string{"cudaLaunchKernel"}

	optionalUprobes := []uprobeSpec{
		{symbol: "cudaLaunchKernelExC", prog: objs.HandleCudaLaunch},
		{symbol: "cudaLaunchKernel_ptsz", prog: objs.HandleCudaLaunch},
		{symbol: "cudaLaunchKernelExC_ptsz", prog: objs.HandleCudaLaunch},
		{symbol: "cudaGraphLaunch", prog: objs.HandleGraphLaunch},
		{symbol: "cudaGraphLaunch_ptsz", prog: objs.HandleGraphLaunch},
		{symbol: "cudaMemcpy", prog: objs.HandleCudartMemcpy},
		{symbol: "cudaMemcpy_ptds", prog: objs.HandleCudartMemcpy},
		{symbol: "cudaMalloc", prog: objs.HandleMemAlloc},
	}
	for _, spec := range optionalUprobes {
		l, uErr := ex.Uprobe(spec.symbol, spec.prog, nil)
		if uErr == nil {
			links = append(links, l)
			hooked = append(hooked, spec.symbol)
		}
	}

	for _, sym := range []string{"cudaStreamSynchronize", "cudaStreamSynchronize_ptsz", "cudaDeviceSynchronize"} {
		up, uErr := ex.Uprobe(sym, objs.HandleSyncEnter, nil)
		if uErr != nil {
			continue
		}
		uret, rErr := ex.Uretprobe(sym, objs.HandleSyncExit, nil)
		if rErr != nil {
			_ = up.Close()
			continue
		}
		links = append(links, up, uret)
		hooked = append(hooked, sym)
	}

	return links, hooked, nil
}

// AddTargetPID registers a process ID with the live tracing session so scheduler
// off-CPU events and final Summary filtering include this process and its children.
func (s *Session) AddTargetPID(pid int) {
	if s == nil || pid <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return
	}
	if s.targetPIDSet == nil {
		s.targetPIDSet = make(map[uint32]struct{})
	}
	tgid := uint32(pid)
	s.targetPIDSet[tgid] = struct{}{}
	if s.active && s.objs != nil && s.objs.TargetTgids != nil {
		val := uint8(1)
		_ = s.objs.TargetTgids.Put(&tgid, &val)
	}
}

// Stop detaches all eBPF probes, reads aggregated kernel map counters, closes BPF
// resources, and returns the finalized Summary. Safe to call multiple times.
func (s *Session) Stop() Summary {
	if s == nil {
		return Summary{
			Enabled:    false,
			Active:     false,
			Mode:       ModeOff,
			Provenance: ProvenanceDisabled,
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return s.cachedSummary
	}
	s.stopped = true

	targetPIDs := sortedUint32Keys(s.targetPIDSet)
	sum := Summary{
		Enabled:           s.enabled,
		Active:            s.active,
		Mode:              s.mode,
		Provenance:        s.provenance,
		StatusDetail:      s.statusDetail,
		AttachedLibCUDA:   s.libCudaPath,
		AttachedLibCUDART: s.libCudartPath,
		TargetPIDs:        targetPIDs,
	}
	if len(s.hookedSymbols) > 0 {
		sum.HookedSymbols = append([]string(nil), s.hookedSymbols...)
	}

	if !s.active || s.objs == nil {
		s.cachedSummary = sum
		return sum
	}

	durationMS := float64(time.Since(s.startedAt).Nanoseconds()) / 1e6

	// Detach links before reading maps so metrics represent a clean snapshot.
	closeLinks(s.links)
	s.links = nil

	var entries []rawProcessEntry
	if s.objs.TgidStats != nil {
		var (
			tgid uint32
			val  cudaTraceCudaTgidStats
		)
		iter := s.objs.TgidStats.Iterate()
		for iter.Next(&tgid, &val) {
			comm := int8CommToString(val.Comm)
			if comm == "" {
				comm = readProcComm(tgid)
			}
			entries = append(entries, rawProcessEntry{
				PID:               tgid,
				Comm:              comm,
				KernelLaunches:    val.KernelLaunches,
				GraphLaunches:     val.GraphLaunches,
				LaunchGapSumNs:    val.LaunchGapSumNs,
				LaunchGapCount:    val.LaunchGapCount,
				MaxLaunchGapNs:    val.MaxLaunchGapNs,
				BlockThreadsSum:   val.BlockThreadsSum,
				GridBlocksSum:     val.GridBlocksSum,
				SharedMemBytesSum: val.SharedMemBytesSum,
				LastGridX:         val.LastGridX,
				LastBlockX:        val.LastBlockX,
				MemcpyH2DCalls:    val.MemcpyH2dCalls,
				MemcpyH2DBytes:    val.MemcpyH2dBytes,
				MemcpyD2HCalls:    val.MemcpyD2hCalls,
				MemcpyD2HBytes:    val.MemcpyD2hBytes,
				MemAllocCalls:     val.MemAllocCalls,
				MemAllocBytes:     val.MemAllocBytes,
				SyncCalls:         val.SyncCalls,
				SyncStallSumNs:    val.SyncStallSumNs,
				MaxSyncStallNs:    val.MaxSyncStallNs,
				SchedSwitches:     val.SchedSwitches,
				OffCPUSumNs:       val.OffCpuSumNs,
				LaunchGapHist:     val.LaunchGapHist,
				SyncStallHist:     val.SyncStallHist,
			})
		}
	}

	_ = s.objs.Close()
	s.objs = nil

	filterAndAggregateEntries(
		entries,
		s.targetPIDSet,
		uint32(os.Getpid()),
		readProcParentPID,
		durationMS,
		&sum,
	)

	s.cachedSummary = sum
	return sum
}

func closeLinks(links []link.Link) {
	for _, l := range links {
		if l != nil {
			_ = l.Close()
		}
	}
}

func isPermissionError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, unix.EPERM) || errors.Is(err, unix.EACCES) || errors.Is(err, os.ErrPermission) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "operation not permitted") ||
		strings.Contains(msg, "permission denied")
}

func enableTempDumpable() func() {
	dumpable, err := unix.PrctlRetInt(unix.PR_GET_DUMPABLE, 0, 0, 0, 0)
	if err != nil || dumpable != 0 {
		return func() {}
	}
	if err := unix.Prctl(unix.PR_SET_DUMPABLE, 1, 0, 0, 0); err != nil {
		return func() {}
	}
	return func() {
		_ = unix.Prctl(unix.PR_SET_DUMPABLE, 0, 0, 0, 0)
	}
}

func loadCudaTraceObjectsWithFallback(objs *cudaTraceObjects) error {
	spec, err := loadCudaTrace()
	if err != nil {
		return err
	}
	if kv := unameKernelVersionCode(); kv != 0 {
		for _, prog := range spec.Programs {
			if prog.KernelVersion == 0 {
				prog.KernelVersion = kv
			}
		}
	}
	return spec.LoadAndAssign(objs, nil)
}

func unameKernelVersionCode() uint32 {
	var uts unix.Utsname
	if err := unix.Uname(&uts); err != nil {
		return 0
	}
	rel := unix.ByteSliceToString(uts.Release[:])
	parts := strings.SplitN(rel, "-", 2)[0]
	nums := strings.Split(parts, ".")
	if len(nums) < 2 {
		return 0
	}
	major, err1 := strconv.Atoi(nums[0])
	minor, err2 := strconv.Atoi(nums[1])
	if err1 != nil || err2 != nil {
		return 0
	}
	patch := 0
	if len(nums) >= 3 {
		if p, err := strconv.Atoi(nums[2]); err == nil {
			if p > 255 {
				p = 255
			}
			patch = p
		}
	}
	return uint32((major << 16) + (minor << 8) + patch)
}
