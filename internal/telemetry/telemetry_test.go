package telemetry

import (
	"context"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"gpumode/internal/nvml"
)

func TestParseNvidiaSMICSVLine(t *testing.T) {
	ts := time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)

	t.Run("standard RTX 4050 Laptop GPU output", func(t *testing.T) {
		raw := "NVIDIA GeForce RTX 4050 Laptop GPU, 42.68, 2355, 8001, 58, 76, 44, 1842, 6141\n"
		s, err := ParseNvidiaSMICSVLine(raw, ts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if s.GPUName != "NVIDIA GeForce RTX 4050 Laptop GPU" {
			t.Errorf("got GPUName %q, want %q", s.GPUName, "NVIDIA GeForce RTX 4050 Laptop GPU")
		}
		if math.Abs(s.PowerWatts-42.68) > 1e-6 {
			t.Errorf("got PowerWatts %v, want 42.68", s.PowerWatts)
		}
		if s.SMClockMHz != 2355 {
			t.Errorf("got SMClockMHz %v, want 2355", s.SMClockMHz)
		}
		if s.MemClockMHz != 8001 {
			t.Errorf("got MemClockMHz %v, want 8001", s.MemClockMHz)
		}
		if s.TemperatureC != 58 {
			t.Errorf("got TemperatureC %v, want 58", s.TemperatureC)
		}
		if s.GPUUtilPct != 76 {
			t.Errorf("got GPUUtilPct %v, want 76", s.GPUUtilPct)
		}
		if s.MemUtilPct != 44 {
			t.Errorf("got MemUtilPct %v, want 44", s.MemUtilPct)
		}
		if s.VRAMUsedMiB != 1842 || s.VRAMTotalMiB != 6141 {
			t.Errorf("got VRAM %v/%v, want 1842/6141", s.VRAMUsedMiB, s.VRAMTotalMiB)
		}
	})

	t.Run("handles [N/A] and [Not Supported] fields gracefully", func(t *testing.T) {
		raw := "NVIDIA GeForce RTX 4050 Laptop GPU, [N/A], 2100, [Not Supported], 51, [N/A], 12, 512, 6141"
		s, err := ParseNvidiaSMICSVLine(raw, ts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if s.PowerWatts != 0 {
			t.Errorf("expected PowerWatts=0 for [N/A], got %v", s.PowerWatts)
		}
		if s.MemClockMHz != 0 {
			t.Errorf("expected MemClockMHz=0 for [Not Supported], got %v", s.MemClockMHz)
		}
		if s.GPUUtilPct != 0 {
			t.Errorf("expected GPUUtilPct=0 for [N/A], got %v", s.GPUUtilPct)
		}
		if s.SMClockMHz != 2100 {
			t.Errorf("expected SMClockMHz=2100, got %v", s.SMClockMHz)
		}
	})

	t.Run("rejects malformed or empty output", func(t *testing.T) {
		if _, err := ParseNvidiaSMICSVLine("", ts); err == nil {
			t.Errorf("expected error for empty input")
		}
		if _, err := ParseNvidiaSMICSVLine("NVIDIA GPU, 35.0, 2100", ts); err == nil {
			t.Errorf("expected error for insufficient CSV columns")
		}
	})
}

func TestTrapezoidalEnergyIntegration(t *testing.T) {
	m := NewMonitor("nvidia-smi", 100*time.Millisecond, 50)
	t0 := time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)

	// Record linear ramp: t=0s (30W), t=1s (50W), t=2s (70W)
	// Integral from 0s to 2s = 0.5*(30+50)*1 + 0.5*(50+70)*1 = 40 + 60 = 100 Joules.
	// Avg power = 100J / 2s = 50W.
	samples := []Sample{
		{Timestamp: t0, GPUName: "RTX 4050", PowerWatts: 30, SMClockMHz: 2000, MemClockMHz: 8000, TemperatureC: 55, VRAMUsedMiB: 1200},
		{Timestamp: t0.Add(1 * time.Second), GPUName: "RTX 4050", PowerWatts: 50, SMClockMHz: 2200, MemClockMHz: 8000, TemperatureC: 60, VRAMUsedMiB: 1500},
		{Timestamp: t0.Add(2 * time.Second), GPUName: "RTX 4050", PowerWatts: 70, SMClockMHz: 2400, MemClockMHz: 8000, TemperatureC: 65, VRAMUsedMiB: 1800},
	}
	for _, s := range samples {
		m.RecordSample(s)
	}

	// 200 completed requests over 2.0s window, 10ms per request latency
	em := m.MeasureWindow(t0, t0.Add(2*time.Second), 200, 10.0)

	if math.Abs(em.DurationSec-2.0) > 1e-6 {
		t.Errorf("DurationSec = %v, want 2.0", em.DurationSec)
	}
	if math.Abs(em.TotalEnergyJoules-100.0) > 1e-6 {
		t.Errorf("TotalEnergyJoules = %v, want 100.0", em.TotalEnergyJoules)
	}
	if math.Abs(em.AvgPowerWatts-50.0) > 1e-6 {
		t.Errorf("AvgPowerWatts = %v, want 50.0", em.AvgPowerWatts)
	}
	if math.Abs(em.PeakPowerWatts-70.0) > 1e-6 {
		t.Errorf("PeakPowerWatts = %v, want 70.0", em.PeakPowerWatts)
	}
	if math.Abs(em.AvgSMClockMHz-2200.0) > 1e-6 {
		t.Errorf("AvgSMClockMHz = %v, want 2200.0", em.AvgSMClockMHz)
	}
	if math.Abs(em.PeakTempC-65.0) > 1e-6 {
		t.Errorf("PeakTempC = %v, want 65.0", em.PeakTempC)
	}
	if math.Abs(em.PeakVRAMMiB-1800.0) > 1e-6 {
		t.Errorf("PeakVRAMMiB = %v, want 1800.0", em.PeakVRAMMiB)
	}
	// JoulesPerRequest = 100J / 200 reqs = 0.5 J/req
	if math.Abs(em.JoulesPerRequest-0.5) > 1e-6 {
		t.Errorf("JoulesPerRequest = %v, want 0.5", em.JoulesPerRequest)
	}
	// JoulesPer1KRequests = 500 J
	if math.Abs(em.JoulesPer1KRequests-500.0) > 1e-6 {
		t.Errorf("JoulesPer1KRequests = %v, want 500.0", em.JoulesPer1KRequests)
	}
	// EDP = 0.5 J * 0.010 s = 0.005 J*s
	if math.Abs(em.EnergyDelayProduct-0.005) > 1e-6 {
		t.Errorf("EnergyDelayProduct = %v, want 0.005", em.EnergyDelayProduct)
	}
	// Perf/Watt = (200 req / 2s) / 50W = 100 RPS / 50W = 2.0 req/s/W
	if math.Abs(em.PerformancePerWatt-2.0) > 1e-6 {
		t.Errorf("PerformancePerWatt = %v, want 2.0", em.PerformancePerWatt)
	}
	if em.SampleCount != 3 {
		t.Errorf("SampleCount = %d, want 3", em.SampleCount)
	}

	// Short sub-poll microbenchmark window test (e.g. 20ms window between t0+1.5s and t0+1.52s)
	shortStart := t0.Add(1500 * time.Millisecond)
	shortEnd := t0.Add(1520 * time.Millisecond)
	shortEM := m.MeasureWindow(shortStart, shortEnd, 10, 2.0)
	// At t=1.5s, interpolated power is ~60W; over 0.02s energy is ~1.20J
	if shortEM.TotalEnergyJoules <= 0 || math.Abs(shortEM.AvgPowerWatts-60.2) > 0.5 {
		t.Errorf("short window interpolation failed: AvgPower=%v, Energy=%v", shortEM.AvgPowerWatts, shortEM.TotalEnergyJoules)
	}
}

func TestRingBufferConcurrencyAndPubSub(t *testing.T) {
	capacity := 10
	m := NewMonitor("nvidia-smi", 20*time.Millisecond, capacity)

	subCh, unsub := m.Subscribe()
	defer unsub()

	var wg sync.WaitGroup
	// Spawn concurrent writers and readers
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 15; i++ {
				m.RecordSample(Sample{
					Timestamp:  time.Now(),
					GPUName:    "RTX 4050",
					PowerWatts: float64(30 + worker + i),
					SMClockMHz: 2355,
				})
				_ = m.Latest()
				_ = m.History(5)
			}
		}(w)
	}
	wg.Wait()

	hist := m.History(0)
	if len(hist) != capacity {
		t.Fatalf("expected ring buffer length %d, got %d", capacity, len(hist))
	}

	limited := m.History(4)
	if len(limited) != 4 {
		t.Fatalf("expected limited history length 4, got %d", len(limited))
	}

	// Verify subscriber received streamed samples
	select {
	case s := <-subCh:
		if s.GPUName != "RTX 4050" {
			t.Errorf("unexpected sample from subscriber channel: %+v", s)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for sample on subscriber channel")
	}

	// Verify Start(ctx) polls custom sampler and exits cleanly when context is canceled
	ctx, cancel := context.WithCancel(context.Background())
	m.SetSampler(func(ctx context.Context) (Sample, error) {
		return Sample{
			Timestamp:  time.Now(),
			GPUName:    "Mock GPU",
			PowerWatts: 48.5,
			SMClockMHz: 2400,
		}, nil
	})
	m.Start(ctx)
	time.Sleep(50 * time.Millisecond)
	cancel()

	latest := m.Latest()
	if latest.GPUName != "Mock GPU" || latest.PowerWatts != 48.5 {
		t.Errorf("expected latest sample from background poller, got %+v", latest)
	}
}

func TestAggregateWindowMeasurementsIdleSubtractionAndUncertainty(t *testing.T) {
	w1 := EnergyMeasurement{
		DurationSec:         1.0,
		TotalEnergyJoules:   50.0,
		AvgPowerWatts:       50.0,
		JoulesPerRequest:    0.5,
		JoulesPer1KRequests: 500.0,
		SampleCount:         3,
		Provenance:          "MEASURED",
	}
	w2 := EnergyMeasurement{
		DurationSec:         1.0,
		TotalEnergyJoules:   54.0,
		AvgPowerWatts:       54.0,
		JoulesPerRequest:    0.54,
		JoulesPer1KRequests: 540.0,
		SampleCount:         3,
		Provenance:          "MEASURED",
	}
	agg := AggregateWindowMeasurements([]EnergyMeasurement{w1, w2}, 10.0, 2000.0, 50*time.Millisecond)
	if agg.Provenance != "MEASURED" {
		t.Errorf("expected MEASURED provenance, got %s", agg.Provenance)
	}
	if agg.RepeatWindows != 2 {
		t.Errorf("expected RepeatWindows=2, got %d", agg.RepeatWindows)
	}
	if math.Abs(agg.IdlePowerWatts-10.0) > 1e-6 || math.Abs(agg.ActivePowerWatts-42.0) > 1e-6 {
		t.Errorf("expected Idle=10W and Active=42W, got Idle=%.2f Active=%.2f", agg.IdlePowerWatts, agg.ActivePowerWatts)
	}
	if agg.EnergyUncertaintyPct <= 0 || agg.EnergyUncertaintyJoules <= 0 {
		t.Errorf("expected positive energy uncertainty, got pct=%.2f J=%.2f", agg.EnergyUncertaintyPct, agg.EnergyUncertaintyJoules)
	}
}

func TestParseBackendMode(t *testing.T) {
	cases := []struct {
		raw  string
		want BackendMode
	}{
		{raw: "auto", want: BackendAuto},
		{raw: "AUTO", want: BackendAuto},
		{raw: "nvml", want: BackendNVML},
		{raw: " NVML ", want: BackendNVML},
		{raw: "smi", want: BackendSMI},
		{raw: "SMI", want: BackendSMI},
		{raw: "", want: BackendAuto},
		{raw: "   ", want: BackendAuto},
		{raw: "unknown-backend", want: BackendAuto},
	}

	for _, tc := range cases {
		if got := ParseBackendMode(tc.raw); got != tc.want {
			t.Errorf("ParseBackendMode(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

func TestMonitorNVMLAndFallback(t *testing.T) {
	// Verify CSV parser sets Source = "NVIDIA_SMI"
	csvSample, err := ParseNvidiaSMICSVLine(
		"NVIDIA GeForce RTX 4050 Laptop GPU, 42.68, 2355, 8001, 58, 76, 44, 1842, 6141\n",
		time.Date(2026, 11, 20, 0, 0, 0, 0, time.UTC),
	)
	if err != nil {
		t.Fatalf("ParseNvidiaSMICSVLine error: %v", err)
	}
	if csvSample.Source != "NVIDIA_SMI" {
		t.Errorf("CSV Sample.Source = %q, want NVIDIA_SMI", csvSample.Source)
	}

	// BackendSMI forces NVIDIA_SMI even when NVML is present on the host
	smiMon := NewMonitorWithBackend("nvidia-smi", 0, 50*time.Millisecond, 32, BackendSMI)
	defer smiMon.Close()
	if smiMon.Backend() != "NVIDIA_SMI" {
		t.Fatalf("BackendSMI Backend() = %q, want NVIDIA_SMI", smiMon.Backend())
	}
	if smiMon.BurstInterval() != 50*time.Millisecond {
		t.Fatalf("BackendSMI BurstInterval() = %v, want 50ms", smiMon.BurstInterval())
	}
	if pids := smiMon.ActiveComputePIDs(); pids != nil {
		t.Fatalf("BackendSMI ActiveComputePIDs() = %v, want nil", pids)
	}

	// Custom non-standard binary path in BackendAuto sticks to NVIDIA_SMI so test overrides aren't bypassed
	customPathMon := NewMonitorWithBackend("testdata/fake-smi-binary", 0, 50*time.Millisecond, 32, BackendAuto)
	defer customPathMon.Close()
	if customPathMon.Backend() != "NVIDIA_SMI" {
		t.Fatalf("non-standard smi path Backend() = %q, want NVIDIA_SMI", customPathMon.Backend())
	}

	// Probe whether NVML with >=1 GPU is available on this host
	nvmlAvailable := false
	if probe, openErr := nvml.Open(); openErr == nil {
		if cnt, cntErr := probe.DeviceCount(); cntErr == nil && cnt > 0 {
			nvmlAvailable = true
		}
		_ = probe.Close()
	}

	autoMon := NewMonitorWithBackend("nvidia-smi", 0, 50*time.Millisecond, 32, BackendAuto)
	defer autoMon.Close()

	if nvmlAvailable {
		if autoMon.Backend() != "NVML" {
			t.Fatalf("expected Backend() == NVML on NVML-capable host, got %q", autoMon.Backend())
		}
		if autoMon.BurstInterval() != 10*time.Millisecond {
			t.Fatalf("expected BurstInterval() == 10ms under NVML, got %v", autoMon.BurstInterval())
		}

		s, pollErr := autoMon.PollOnce(context.Background())
		if pollErr != nil {
			t.Fatalf("NVML PollOnce failed: %v", pollErr)
		}
		if s.Source != "NVML" {
			t.Errorf("Sample.Source = %q, want NVML", s.Source)
		}
		if s.VRAMTotalMiB <= 0 {
			t.Errorf("Sample.VRAMTotalMiB = %v, want > 0", s.VRAMTotalMiB)
		}
		if strings.TrimSpace(s.GPUName) == "" {
			t.Errorf("expected non-empty Sample.GPUName")
		}

		pids := autoMon.ActiveComputePIDs()
		selfPID := os.Getpid()
		for i, pid := range pids {
			if pid <= 0 {
				t.Errorf("expected positive PID, got %d", pid)
			}
			if pid == selfPID {
				t.Errorf("expected ActiveComputePIDs to exclude self PID %d", selfPID)
			}
			if i > 0 && pids[i-1] > pid {
				t.Errorf("expected sorted PIDs, got %v", pids)
			}
		}

		// Verify SetSampler transitions to CUSTOM and nil restores NVML
		autoMon.SetSampler(func(ctx context.Context) (Sample, error) {
			return Sample{GPUName: "CustomGPU", PowerWatts: 33.0}, nil
		})
		if autoMon.Backend() != "CUSTOM" || autoMon.BurstInterval() != 50*time.Millisecond {
			t.Fatalf("after SetSampler(fn), Backend=%q BurstInterval=%v", autoMon.Backend(), autoMon.BurstInterval())
		}
		cs, err := autoMon.PollOnce(context.Background())
		if err != nil || cs.Source != "CUSTOM" {
			t.Fatalf("custom PollOnce err=%v Source=%q, want CUSTOM", err, cs.Source)
		}
		autoMon.SetSampler(nil)
		if autoMon.Backend() != "NVML" || autoMon.BurstInterval() != 10*time.Millisecond {
			t.Fatalf("after SetSampler(nil), Backend=%q BurstInterval=%v, want NVML / 10ms", autoMon.Backend(), autoMon.BurstInterval())
		}

		// Close is idempotent and transitions monitor cleanly to NVIDIA_SMI fallback
		if err := autoMon.Close(); err != nil {
			t.Fatalf("first Close() returned error: %v", err)
		}
		if err := autoMon.Close(); err != nil {
			t.Fatalf("second Close() returned error: %v", err)
		}
		if autoMon.Backend() != "NVIDIA_SMI" {
			t.Fatalf("after Close(), Backend() = %q, want NVIDIA_SMI", autoMon.Backend())
		}
	} else {
		if autoMon.Backend() != "NVIDIA_SMI" {
			t.Fatalf("expected fallback Backend() == NVIDIA_SMI, got %q", autoMon.Backend())
		}
		if autoMon.BurstInterval() != 50*time.Millisecond {
			t.Fatalf("expected fallback BurstInterval() == 50ms, got %v", autoMon.BurstInterval())
		}
		if pids := autoMon.ActiveComputePIDs(); pids != nil {
			t.Fatalf("expected nil ActiveComputePIDs without NVML, got %v", pids)
		}
		if err := autoMon.Close(); err != nil {
			t.Fatalf("Close() returned error: %v", err)
		}
	}
}

func TestIntegrateSamplesWindowGPUUtilStats(t *testing.T) {
	t0 := time.Date(2026, 11, 20, 12, 0, 0, 0, time.UTC)
	samples := []Sample{
		{Timestamp: t0, PowerWatts: 30, GPUUtilPct: 40, MemUtilPct: 20, VRAMUsedMiB: 2000},
		{Timestamp: t0.Add(1 * time.Second), PowerWatts: 50, GPUUtilPct: 80, MemUtilPct: 60, VRAMUsedMiB: 4500},
		{Timestamp: t0.Add(2 * time.Second), PowerWatts: 40, GPUUtilPct: 60, MemUtilPct: 40, VRAMUsedMiB: 4200},
	}
	em := IntegrateSamplesWindow(samples, t0, t0.Add(2*time.Second), 2, 1000)
	if math.Abs(em.AvgGPUUtilPct-65.0) > 1e-6 {
		t.Errorf("AvgGPUUtilPct = %v, want 65.0", em.AvgGPUUtilPct)
	}
	if em.PeakGPUUtilPct != 80.0 {
		t.Errorf("PeakGPUUtilPct = %v, want 80.0", em.PeakGPUUtilPct)
	}
	if math.Abs(em.AvgMemUtilPct-45.0) > 1e-6 {
		t.Errorf("AvgMemUtilPct = %v, want 45.0", em.AvgMemUtilPct)
	}
	if em.PeakMemUtilPct != 60.0 {
		t.Errorf("PeakMemUtilPct = %v, want 60.0", em.PeakMemUtilPct)
	}

	agg := AggregateWindowMeasurements([]EnergyMeasurement{em}, 5.0, 2000, 100*time.Millisecond)
	if math.Abs(agg.AvgGPUUtilPct-65.0) > 1e-6 || agg.PeakGPUUtilPct != 80.0 {
		t.Errorf("AggregateWindowMeasurements GPU util = avg %.1f peak %.1f, want 65.0 / 80.0", agg.AvgGPUUtilPct, agg.PeakGPUUtilPct)
	}
}

func TestMonitorWithMockedNVMLAndSMIFallbackFailurePaths(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mock NVML dlopen test requires Linux")
	}
	gccPath, err := exec.LookPath("gcc")
	if err != nil {
		t.Skip("gcc not available; skipping mock NVML monitor test")
	}

	dir := t.TempDir()
	cPath := filepath.Join(dir, "mock_nvml_mon.c")
	soPath := filepath.Join(dir, "libnvidia-ml.so.1")
	cSrc := `
#include <stdlib.h>
#include <string.h>
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef struct { unsigned int gpu; unsigned int memory; } nvmlUtilization_t;
typedef struct { unsigned long long total; unsigned long long free; unsigned long long used; } nvmlMemory_t;
typedef struct { unsigned int pid; unsigned long long usedGpuMemory; unsigned int gId; unsigned int cId; } nvmlProcessInfo_v2_t;

nvmlReturn_t nvmlInit_v2(void) { return 0; }
nvmlReturn_t nvmlShutdown(void) { return 0; }
const char* nvmlErrorString(nvmlReturn_t r) { return "Mock Error"; }
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* c) { *c = 1; return 0; }
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int idx, nvmlDevice_t* d) { *d = (void*)0x1; return 0; }
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t d, char* n, unsigned int c) { strncpy(n, "NVIDIA L40S Mock", c-1); n[c-1]=0; return 0; }
nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t d, unsigned int* mw) {
    const char* f = getenv("MOCK_FAIL_SAMPLE");
    if (f && f[0] == '1') return 99;
    *mw = 180000;
    return 0;
}
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t d, int t, unsigned int* m) { *m = (t==1)?2100:5000; return 0; }
nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t d, int s, unsigned int* t) { *t = 54; return 0; }
nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t d, nvmlUtilization_t* u) { u->gpu = 88; u->memory = 62; return 0; }
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t d, nvmlMemory_t* m) { m->total = 48ULL<<30; m->used = 12ULL<<30; m->free = 36ULL<<30; return 0; }
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t d, unsigned int* c, nvmlProcessInfo_v2_t* p) {
    if (!p || *c < 1) { *c = 1; return 7; }
    p[0].pid = 77777;
    p[0].usedGpuMemory = 1024ULL<<20;
    *c = 1;
    return 0;
}
`
	if err := os.WriteFile(cPath, []byte(cSrc), 0o600); err != nil {
		t.Fatalf("write mock_nvml_mon.c: %v", err)
	}
	if out, err := exec.Command(gccPath, "-shared", "-fPIC", cPath, "-o", soPath).CombinedOutput(); err != nil {
		t.Fatalf("compile mock_nvml_mon.so: %v (%s)", err, string(out))
	}

	// Create a fake nvidia-smi script to verify automatic fallback when NVML sample fails in BackendAuto
	fakeSMI := filepath.Join(dir, "nvidia-smi")
	smiScript := "#!/bin/sh\necho 'NVIDIA L40S SMI Fallback, 175.50, 2050, 5000, 55, 85, 60, 12288, 49152'\n"
	if err := os.WriteFile(fakeSMI, []byte(smiScript), 0o755); err != nil {
		t.Fatalf("write fake nvidia-smi: %v", err)
	}

	t.Setenv("GPUMODE_NVML_LIB", soPath)
	t.Setenv("MOCK_FAIL_SAMPLE", "")
	mon := NewMonitorWithBackend(fakeSMI, 0, 10*time.Millisecond, 32, BackendAuto)
	defer mon.Close()

	if mon.Backend() != "NVML" {
		t.Fatalf("expected Backend() == NVML with mocked libnvidia-ml.so.1, got %q", mon.Backend())
	}
	s, err := mon.PollOnce(context.Background())
	if err != nil {
		t.Fatalf("PollOnce with mocked NVML failed: %v", err)
	}
	if s.GPUName != "NVIDIA L40S Mock" || s.PowerWatts != 180.0 || s.Source != "NVML" {
		t.Fatalf("unexpected mocked NVML sample: %+v", s)
	}
	pids := mon.ActiveComputePIDs()
	if len(pids) != 1 || pids[0] != 77777 {
		t.Fatalf("expected ActiveComputePIDs=[77777], got %v", pids)
	}

	// Simulate NVML runtime device failure -> BackendAuto must fall back to fake nvidia-smi seamlessly
	t.Setenv("MOCK_FAIL_SAMPLE", "1")
	fbSample, err := mon.PollOnce(context.Background())
	if err != nil {
		t.Fatalf("expected BackendAuto fallback to fake nvidia-smi on NVML error, got: %v", err)
	}
	if fbSample.GPUName != "NVIDIA L40S SMI Fallback" || fbSample.Source != "NVIDIA_SMI" {
		t.Fatalf("unexpected fallback sample: %+v", fbSample)
	}

	// Simulate missing nvidia-smi binary when NVML is closed
	_ = mon.Close()
	badMon := NewMonitorWithBackend("/nonexistent/nvidia-smi-missing", 0, 20*time.Millisecond, 16, BackendSMI)
	defer badMon.Close()
	if _, err := badMon.PollOnce(context.Background()); err == nil {
		t.Fatal("expected error when polling nonexistent nvidia-smi binary")
	}
}
