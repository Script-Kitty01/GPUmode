package ebpftrace

import (
	"math"
	"net"
	"os"
	"runtime"
	"slices"
	"strconv"
	"testing"
)

func TestParseMode(t *testing.T) {
	tests := []struct {
		raw  string
		want Mode
	}{
		{"auto", ModeAuto},
		{"AUTO", ModeAuto},
		{"  Auto \t", ModeAuto},
		{"", ModeAuto},
		{"   ", ModeAuto},
		{"on", ModeOn},
		{"ON", ModeOn},
		{" On ", ModeOn},
		{"off", ModeOff},
		{"OFF", ModeOff},
		{" oFf ", ModeOff},
		{"unknown", ModeAuto},
		{"bogus-mode", ModeAuto},
	}

	for _, tc := range tests {
		got := ParseMode(tc.raw)
		if got != tc.want {
			t.Errorf("ParseMode(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

func TestModeOffSession(t *testing.T) {
	sess, err := StartSession(ModeOff, []int{1234})
	if err != nil {
		t.Fatalf("StartSession(ModeOff) unexpected error: %v", err)
	}
	sess.AddTargetPID(5678)

	sum := sess.Stop()
	if sum.Enabled {
		t.Errorf("expected Enabled=false for ModeOff, got true")
	}
	if sum.Active {
		t.Errorf("expected Active=false for ModeOff, got true")
	}
	if sum.Mode != ModeOff {
		t.Errorf("expected Mode=%q, got %q", ModeOff, sum.Mode)
	}
	if sum.Provenance != ProvenanceDisabled {
		t.Errorf("expected Provenance=%q, got %q", ProvenanceDisabled, sum.Provenance)
	}
	if !slices.Equal(sum.TargetPIDs, []uint32{1234, 5678}) {
		t.Errorf("expected TargetPIDs=[1234 5678], got %v", sum.TargetPIDs)
	}

	// Verify Stop() idempotency and ProbeAvailability(ModeOff).
	sum2 := sess.Stop()
	if sum2.Provenance != ProvenanceDisabled {
		t.Errorf("second Stop() Provenance=%q, want %q", sum2.Provenance, ProvenanceDisabled)
	}
	probe := ProbeAvailability(ModeOff)
	if probe.Provenance != ProvenanceDisabled || probe.Enabled || probe.Active {
		t.Errorf("unexpected ProbeAvailability(ModeOff): %+v", probe)
	}
}

func TestDiscoverCUDALibraries(t *testing.T) {
	libCuda, libCudart := DiscoverCUDALibraries(os.Getpid())
	if libCuda == "" {
		t.Skip("libcuda.so.1 not present on this host; skipping ELF symbol verification")
	}

	info, err := os.Stat(libCuda)
	if err != nil {
		t.Fatalf("stat discovered libCuda %q: %v", libCuda, err)
	}
	if info.IsDir() || info.Size() == 0 {
		t.Fatalf("discovered libCuda %q is not a non-empty regular file", libCuda)
	}

	requiredSyms := []string{
		"cuLaunchKernel",
		"cuMemcpyHtoD_v2",
		"cuMemcpyDtoH_v2",
		"cuMemAlloc_v2",
		"cuStreamSynchronize",
		"cuCtxSynchronize",
	}
	found, err := inspectELFSymbols(libCuda, requiredSyms...)
	if err != nil {
		t.Fatalf("inspectELFSymbols(%q): %v", libCuda, err)
	}
	for _, sym := range requiredSyms {
		if !found[sym] {
			t.Errorf("expected symbol %q in %q, but not found", sym, libCuda)
		}
	}

	if libCudart != "" {
		rtSyms, err := inspectELFSymbols(libCudart, "cudaLaunchKernel", "cudaStreamSynchronize")
		if err != nil {
			t.Fatalf("inspectELFSymbols(%q): %v", libCudart, err)
		}
		if !rtSyms["cudaLaunchKernel"] {
			t.Errorf("expected cudaLaunchKernel in discovered libcudart %q", libCudart)
		}
	}
}

func TestFindPIDsByTCPPort(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("FindPIDsByTCPPort requires Linux /proc/net/tcp")
	}
	if _, err := os.ReadDir("/proc/" + strconv.Itoa(os.Getpid()) + "/fd"); err != nil {
		t.Skipf("/proc/<pid>/fd not readable in this environment: %v", err)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	defer ln.Close()

	tcpAddr, ok := ln.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("unexpected listener addr type %T", ln.Addr())
	}

	pids := FindPIDsByTCPPort(tcpAddr.Port)
	if !slices.Contains(pids, os.Getpid()) {
		t.Fatalf("FindPIDsByTCPPort(%d) = %v, expected to include self PID %d", tcpAddr.Port, pids, os.Getpid())
	}

	if got := FindPIDsByTCPPort(0); got != nil {
		t.Errorf("FindPIDsByTCPPort(0) = %v, want nil", got)
	}
	if got := FindPIDsByTCPPort(70000); got != nil {
		t.Errorf("FindPIDsByTCPPort(70000) = %v, want nil", got)
	}
}

func TestModeAutoGracefulOrLive(t *testing.T) {
	sess, err := StartSession(ModeAuto, nil)
	if err != nil {
		t.Fatalf("StartSession(ModeAuto, nil) must never fail with error, got: %v", err)
	}
	sess.AddTargetPID(os.Getpid())
	sum := sess.Stop()

	if !sum.Enabled {
		t.Errorf("expected Enabled=true for ModeAuto, got false")
	}
	if sum.Mode != ModeAuto {
		t.Errorf("expected Mode=%q, got %q", ModeAuto, sum.Mode)
	}

	if sum.Active {
		if sum.Provenance != ProvenanceMeasuredEBPF {
			t.Errorf("Active=true session had Provenance=%q, want %q", sum.Provenance, ProvenanceMeasuredEBPF)
		}
		if len(sum.HookedSymbols) == 0 {
			t.Errorf("Active=true session had empty HookedSymbols")
		}
	} else {
		libCuda, libCudart := DiscoverCUDALibraries()
		if libCuda != "" || libCudart != "" {
			if sum.Provenance != ProvenanceUnavailableUnprivileged && sum.Provenance != ProvenanceUnavailableUnsupported {
				t.Errorf("unexpected inactive Provenance=%q", sum.Provenance)
			}
			// When unprivileged on a host with CUDA libraries, verify exact provenance & actionable message.
			if os.Geteuid() != 0 && sum.Provenance == ProvenanceUnavailableUnprivileged {
				if sum.StatusDetail == "" {
					t.Errorf("expected actionable StatusDetail when unprivileged")
				}
				// Also verify ModeOn returns an explicit error when unprivileged.
				if onSess, onErr := StartSession(ModeOn, nil); onErr == nil {
					_ = onSess.Stop()
					t.Errorf("expected ModeOn to return error when ModeAuto reported UNAVAILABLE_UNPRIVILEGED")
				}
			}
		}
	}
}

func TestFilterAndAggregateSummary(t *testing.T) {
	const mib = 1024 * 1024
	entries := []rawProcessEntry{
		{
			// Self process should be excluded unless explicitly targeted.
			PID:            100,
			Comm:           "gpumode",
			KernelLaunches: 999,
		},
		{
			// Parent wrapper process (targeted, no direct CUDA calls, only sched switches).
			PID:           200,
			Comm:          "bash",
			SchedSwitches: 10,
			OffCPUSumNs:   5_000_000, // 5ms
		},
		{
			// Child worker spawned by PID 200 (matched via PPID lineage).
			PID:            201,
			Comm:           "python3",
			KernelLaunches: 50,
			LaunchGapSumNs: 1_000_000, // 1000us total across 25 gaps => 40us avg
			LaunchGapCount: 25,
			MaxLaunchGapNs: 120_000, // 120us
			MemcpyH2DCalls: 4,
			MemcpyH2DBytes: 8 * mib,
			MemcpyD2HCalls: 2,
			MemcpyD2HBytes: 2 * mib,
			MemAllocCalls:  1,
			MemAllocBytes:  64 * mib,
			SyncCalls:      5,
			SyncStallSumNs: 25_000_000, // 25ms
			MaxSyncStallNs: 10_000_000, // 10ms
			SchedSwitches:  30,
			OffCPUSumNs:    15_000_000, // 15ms
		},
		{
			// Unrelated process.
			PID:            999,
			Comm:           "other",
			KernelLaunches: 777,
		},
	}

	ppidMap := map[uint32]uint32{
		201: 200,
		200: 1,
	}
	targetSet := map[uint32]struct{}{200: {}}

	var sum Summary
	filterAndAggregateEntries(entries, targetSet, 100, func(pid uint32) uint32 {
		return ppidMap[pid]
	}, 100.0, &sum) // 100ms window

	if sum.KernelLaunches != 50 {
		t.Fatalf("KernelLaunches = %d, want 50", sum.KernelLaunches)
	}
	if math.Abs(sum.LaunchesPerSec-500.0) > 1e-6 {
		t.Errorf("LaunchesPerSec = %v, want 500", sum.LaunchesPerSec)
	}
	if math.Abs(sum.AvgLaunchGapUS-40.0) > 1e-6 {
		t.Errorf("AvgLaunchGapUS = %v, want 40", sum.AvgLaunchGapUS)
	}
	if math.Abs(sum.MaxLaunchGapUS-120.0) > 1e-6 {
		t.Errorf("MaxLaunchGapUS = %v, want 120", sum.MaxLaunchGapUS)
	}
	if math.Abs(sum.MemcpyH2DMiB-8.0) > 1e-6 || math.Abs(sum.MemcpyD2HMiB-2.0) > 1e-6 || math.Abs(sum.MemAllocMiB-64.0) > 1e-6 {
		t.Errorf("unexpected memory MiB totals: H2D=%v D2H=%v Alloc=%v", sum.MemcpyH2DMiB, sum.MemcpyD2HMiB, sum.MemAllocMiB)
	}
	if math.Abs(sum.SyncStallMS-25.0) > 1e-6 || math.Abs(sum.SyncStallPct-25.0) > 1e-6 {
		t.Errorf("SyncStallMS=%v SyncStallPct=%v, want 25ms / 25%%", sum.SyncStallMS, sum.SyncStallPct)
	}
	if sum.SchedSwitches != 40 || math.Abs(sum.OffCPUMS-20.0) > 1e-6 || math.Abs(sum.HostStarvationPct-20.0) > 1e-6 {
		t.Errorf("SchedSwitches=%d OffCPUMS=%v HostStarvationPct=%v, want 40 / 20ms / 20%%",
			sum.SchedSwitches, sum.OffCPUMS, sum.HostStarvationPct)
	}
	if len(sum.Processes) != 2 {
		t.Fatalf("len(Processes) = %d, want 2", len(sum.Processes))
	}
}
