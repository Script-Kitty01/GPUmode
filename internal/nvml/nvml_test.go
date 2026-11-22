package nvml

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

func TestOpenInvalidPathFails(t *testing.T) {
	client, err := OpenWithLibPath("/nonexistent/libnvidia-ml.so.999")
	if err == nil {
		_ = client.Close()
		t.Fatal("expected error for nonexistent library path, got nil")
	}
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("expected error wrapping ErrUnavailable, got: %v", err)
	}
	if client != nil {
		t.Fatalf("expected nil client on failure, got: %+v", client)
	}
}

func TestUsedVRAMBytesToMiB(t *testing.T) {
	if got := usedVRAMBytesToMiB(^uint64(0)); got != 0 {
		t.Fatalf("expected 0 MiB for NVML_VALUE_NOT_AVAILABLE (~0ULL), got %v", got)
	}
	if got := usedVRAMBytesToMiB(256 * 1024 * 1024); got != 256.0 {
		t.Fatalf("expected 256.0 MiB, got %v", got)
	}
}

func TestDecodeThrottleReasons(t *testing.T) {
	if got := DecodeThrottleReasons(0); len(got) != 0 {
		t.Fatalf("expected nil for mask 0, got %v", got)
	}
	mask := ThrottleReasonSwPowerCap | ThrottleReasonHwThermalSlowdown
	reasons := DecodeThrottleReasons(mask)
	if len(reasons) != 2 || reasons[0] != "SwPowerCap" || reasons[1] != "HwThermalSlowdown" {
		t.Fatalf("unexpected decoded throttle reasons: %v", reasons)
	}
}

func TestOpenEnvVarSearchOrder(t *testing.T) {
	t.Setenv("GPUMODE_NVML_LIB", "/nonexistent/custom-nvml.so.1")
	client, err := Open()
	if errors.Is(err, ErrUnavailable) {
		t.Skipf("NVML unavailable on this host: %v", err)
	}
	if err != nil {
		t.Fatalf("Open() failed when falling back past invalid GPUMODE_NVML_LIB: %v", err)
	}
	resolved := client.LibPath()
	_ = client.Close()

	t.Setenv("GPUMODE_NVML_LIB", resolved)
	client2, err := Open()
	if err != nil {
		t.Fatalf("Open() with valid GPUMODE_NVML_LIB=%q failed: %v", resolved, err)
	}
	defer client2.Close()
	if client2.LibPath() != resolved {
		t.Fatalf("expected LibPath() == %q, got %q", resolved, client2.LibPath())
	}
}

func TestLiveNVMLSampleAndProcesses(t *testing.T) {
	client, err := Open()
	if errors.Is(err, ErrUnavailable) {
		t.Skipf("NVML unavailable on this host: %v", err)
	}
	if err != nil {
		t.Fatalf("Open() returned unexpected error: %v", err)
	}
	defer func() {
		if closeErr := client.Close(); closeErr != nil {
			t.Errorf("deferred Close() failed: %v", closeErr)
		}
	}()

	if client.LibPath() == "" {
		t.Fatal("expected non-empty LibPath()")
	}

	count, err := client.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount() failed: %v", err)
	}
	if count < 1 {
		t.Fatalf("expected DeviceCount() >= 1, got %d", count)
	}

	sample, err := client.SampleDevice(0)
	if err != nil {
		t.Fatalf("SampleDevice(0) failed: %v", err)
	}
	if sample.GPUName == "" {
		t.Fatal("expected non-empty GPUName")
	}
	if sample.VRAMTotalMiB <= 0 {
		t.Fatalf("expected VRAMTotalMiB > 0, got %f", sample.VRAMTotalMiB)
	}
	if sample.SMClockMHz < 0 {
		t.Fatalf("expected SMClockMHz >= 0, got %f", sample.SMClockMHz)
	}
	if sample.PowerWatts < 0 {
		t.Fatalf("expected PowerWatts >= 0, got %f", sample.PowerWatts)
	}

	procs, err := client.ListComputeProcesses(0)
	if err != nil {
		t.Fatalf("ListComputeProcesses(0) failed: %v", err)
	}
	for _, proc := range procs {
		if proc.PID == 0 {
			t.Fatalf("unexpected PID 0 in compute processes: %+v", proc)
		}
		if proc.UsedVRAMMiB < 0 {
			t.Fatalf("expected UsedVRAMMiB >= 0, got %f", proc.UsedVRAMMiB)
		}
	}

	if _, err := client.SampleDevice(-1); err == nil {
		t.Fatal("expected error for negative gpuIndex in SampleDevice(-1)")
	}
	if _, err := client.ListComputeProcesses(-1); err == nil {
		t.Fatal("expected error for negative gpuIndex in ListComputeProcesses(-1)")
	}

	t.Logf("Live NVML (%s): GPU=%q Power=%.2fW TDP=%.2fW SM=%.0f/%.0fMHz Mem=%.0f/%.0fMHz Temp=%.0fC Util=%.0f%% VRAM=%.1f/%.1fMiB Procs=%d",
		client.LibPath(),
		sample.GPUName,
		sample.PowerWatts,
		sample.TDPWatts,
		sample.SMClockMHz,
		sample.MaxSMClockMHz,
		sample.MemClockMHz,
		sample.MaxMemClockMHz,
		sample.TemperatureC,
		sample.GPUUtilPct,
		sample.VRAMUsedMiB,
		sample.VRAMTotalMiB,
		len(procs),
	)

	// Verify concurrent SampleDevice / ListComputeProcesses / DeviceCount is race-free.
	const workers = 8
	const iterations = 10
	var wg sync.WaitGroup
	errCh := make(chan error, workers*iterations*2)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				if _, sErr := client.SampleDevice(0); sErr != nil {
					errCh <- sErr
					return
				}
				if _, pErr := client.ListComputeProcesses(0); pErr != nil {
					errCh <- pErr
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for cErr := range errCh {
		t.Fatalf("concurrent NVML call failed: %v", cErr)
	}

	// Verify Close() idempotency and thread-safety, and post-Close rejection.
	var closeWG sync.WaitGroup
	for i := 0; i < 4; i++ {
		closeWG.Add(1)
		go func() {
			defer closeWG.Done()
			if cErr := client.Close(); cErr != nil {
				t.Errorf("concurrent Close() failed: %v", cErr)
			}
		}()
	}
	closeWG.Wait()

	if cErr := client.Close(); cErr != nil {
		t.Fatalf("second Close() failed: %v", cErr)
	}
	if _, err := client.SampleDevice(0); err == nil {
		t.Fatal("expected SampleDevice(0) to fail after Close()")
	}
}

func buildMockNVMLLib(t *testing.T, cSource string) string {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("dlopen mock NVML shared library test requires Linux")
	}
	gccPath, err := exec.LookPath("gcc")
	if err != nil {
		t.Skip("gcc not installed; skipping mock NVML .so compilation")
	}
	dir := t.TempDir()
	cPath := filepath.Join(dir, "mock_nvml.c")
	soPath := filepath.Join(dir, "libnvidia-ml.so.1")
	if err := os.WriteFile(cPath, []byte(cSource), 0o600); err != nil {
		t.Fatalf("write mock_nvml.c: %v", err)
	}
	cmd := exec.Command(gccPath, "-shared", "-fPIC", "-O2", cPath, "-o", soPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("gcc compile mock_nvml.so failed: %v (%s)", err, string(out))
	}
	return soPath
}

const mockNVMLFullCSource = `
#include <stdlib.h>
#include <string.h>

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

typedef struct { unsigned int gpu; unsigned int memory; } nvmlUtilization_t;
typedef struct { unsigned long long total; unsigned long long free; unsigned long long used; } nvmlMemory_t;
typedef struct {
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
} nvmlProcessInfo_v2_t;

static int get_mode(void) {
    const char* m = getenv("MOCK_NVML_MODE");
    if (!m) return 0;
    if (strcmp(m, "unsupported_metrics") == 0) return 1;
    if (strcmp(m, "init_fail") == 0) return 2;
    if (strcmp(m, "mem_fail") == 0) return 3;
    return 0;
}

nvmlReturn_t nvmlInit_v2(void) {
    if (get_mode() == 2) return 9; /* NVML_ERROR_DRIVER_NOT_LOADED */
    return 0;
}
nvmlReturn_t nvmlShutdown(void) { return 0; }
const char* nvmlErrorString(nvmlReturn_t r) {
    if (r == 3) return "Not Supported";
    if (r == 7) return "Insufficient Size";
    if (r == 9) return "Driver Not Loaded";
    return "Mock NVML Error";
}
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* cnt) {
    *cnt = 1;
    return 0;
}
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int idx, nvmlDevice_t* dev) {
    if (idx != 0) return 2;
    *dev = (void*)0x1234;
    return 0;
}
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t dev, char* name, unsigned int cap) {
    strncpy(name, "NVIDIA H100 Mock 80GB", cap - 1);
    name[cap - 1] = '\0';
    return 0;
}
nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t dev, unsigned int* mw) {
    if (get_mode() == 1) return 3; /* NVML_ERROR_NOT_SUPPORTED */
    *mw = 245500;
    return 0;
}
nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t dev, unsigned int* mw) {
    if (get_mode() == 1) return 3;
    *mw = 700000;
    return 0;
}
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t dev, int typ, unsigned int* mhz) {
    if (get_mode() == 1) return 3;
    *mhz = (typ == 1) ? 1980 : 2619;
    return 0;
}
nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t dev, int typ, unsigned int* mhz) {
    if (get_mode() == 1) return 3;
    *mhz = (typ == 1) ? 1980 : 2619;
    return 0;
}
nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t dev, int sensor, unsigned int* temp) {
    if (get_mode() == 1) return 3;
    *temp = 62;
    return 0;
}
nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t dev, nvmlUtilization_t* u) {
    if (get_mode() == 1) return 3;
    u->gpu = 94;
    u->memory = 78;
    return 0;
}
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t dev, nvmlMemory_t* m) {
    if (get_mode() == 3) return 99;
    m->total = 80ULL * 1024ULL * 1024ULL * 1024ULL;
    m->used  = 40ULL * 1024ULL * 1024ULL * 1024ULL;
    m->free  = 40ULL * 1024ULL * 1024ULL * 1024ULL;
    return 0;
}
nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t dev, unsigned long long* mask) {
    if (get_mode() == 1) return 3;
    *mask = 0x0000000000000004ULL; /* SwPowerCap */
    return 0;
}
nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t dev, int counter, unsigned int* kbps) {
    if (get_mode() == 1) return 3;
    *kbps = (counter == 0) ? 102400 : 204800;
    return 0;
}
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t dev, unsigned int* count, nvmlProcessInfo_v2_t* infos) {
    if (infos == NULL || *count < 2) {
        *count = 2;
        return 7; /* NVML_ERROR_INSUFFICIENT_SIZE */
    }
    infos[0].pid = 424242;
    infos[0].usedGpuMemory = 8192ULL * 1024ULL * 1024ULL;
    infos[1].pid = 999999;
    infos[1].usedGpuMemory = ~0ULL; /* NVML_VALUE_NOT_AVAILABLE */
    *count = 2;
    return 0;
}
`

func TestMockedNVMLSharedLibraryIntegrationAndFailurePaths(t *testing.T) {
	soPath := buildMockNVMLLib(t, mockNVMLFullCSource)

	t.Run("normal mocked NVML sampling and insufficient-size process list retry", func(t *testing.T) {
		t.Setenv("MOCK_NVML_MODE", "normal")
		client, err := OpenWithLibPath(soPath)
		if err != nil {
			t.Fatalf("OpenWithLibPath(mock_nvml.so) failed: %v", err)
		}
		defer client.Close()

		cnt, err := client.DeviceCount()
		if err != nil || cnt != 1 {
			t.Fatalf("DeviceCount() = %d, err = %v; want 1, nil", cnt, err)
		}

		s, err := client.SampleDevice(0)
		if err != nil {
			t.Fatalf("SampleDevice(0) failed: %v", err)
		}
		if s.GPUName != "NVIDIA H100 Mock 80GB" {
			t.Errorf("GPUName = %q, want NVIDIA H100 Mock 80GB", s.GPUName)
		}
		if s.PowerWatts != 245.5 || s.TDPWatts != 700.0 {
			t.Errorf("Power=%.1fW TDP=%.1fW, want 245.5W / 700.0W", s.PowerWatts, s.TDPWatts)
		}
		if s.SMClockMHz != 1980 || s.MemClockMHz != 2619 || s.TemperatureC != 62 {
			t.Errorf("unexpected clocks/temp: %+v", s)
		}
		if s.GPUUtilPct != 94 || s.MemUtilPct != 78 || s.VRAMUsedMiB != 40960 || s.VRAMTotalMiB != 81920 {
			t.Errorf("unexpected util/vram: %+v", s)
		}
		if len(s.ThrottleReasons) != 1 || s.ThrottleReasons[0] != "SwPowerCap" {
			t.Errorf("ThrottleReasons = %v, want [SwPowerCap]", s.ThrottleReasons)
		}
		if s.PCIeTxKBps != 102400 || s.PCIeRxKBps != 204800 {
			t.Errorf("PCIe Tx/Rx = %d/%d, want 102400/204800", s.PCIeTxKBps, s.PCIeRxKBps)
		}

		procs, err := client.ListComputeProcesses(0)
		if err != nil {
			t.Fatalf("ListComputeProcesses(0) failed: %v", err)
		}
		if len(procs) != 2 || procs[0].PID != 424242 || procs[0].UsedVRAMMiB != 8192 || procs[1].PID != 999999 || procs[1].UsedVRAMMiB != 0 {
			t.Fatalf("unexpected compute processes: %+v", procs)
		}
	})

	t.Run("unsupported metrics degrade gracefully without failing SampleDevice", func(t *testing.T) {
		t.Setenv("MOCK_NVML_MODE", "unsupported_metrics")
		client, err := OpenWithLibPath(soPath)
		if err != nil {
			t.Fatalf("OpenWithLibPath failed: %v", err)
		}
		defer client.Close()

		s, err := client.SampleDevice(0)
		if err != nil {
			t.Fatalf("SampleDevice(0) with unsupported metrics should succeed, got: %v", err)
		}
		if s.PowerWatts != 0 || s.SMClockMHz != 0 || s.TemperatureC != 0 || s.GPUUtilPct != 0 || len(s.ThrottleReasons) != 0 {
			t.Errorf("expected unsupported metrics to degrade to 0, got: %+v", s)
		}
		if s.VRAMTotalMiB != 81920 {
			t.Errorf("expected VRAMTotalMiB=81920 even when power/clock sensors are unsupported, got %.0f", s.VRAMTotalMiB)
		}
	})

	t.Run("missing driver / nvmlInit failure returns ErrUnavailable", func(t *testing.T) {
		t.Setenv("MOCK_NVML_MODE", "init_fail")
		client, err := OpenWithLibPath(soPath)
		if err == nil {
			_ = client.Close()
			t.Fatal("expected error when nvmlInit_v2 returns Driver Not Loaded")
		}
		if !errors.Is(err, ErrUnavailable) || !strings.Contains(err.Error(), "Driver Not Loaded") {
			t.Fatalf("expected ErrUnavailable with 'Driver Not Loaded', got: %v", err)
		}
	})

	t.Run("incomplete library missing required symbol fails load cleanly", func(t *testing.T) {
		incompleteC := `
int nvmlInit_v2(void) { return 0; }
int nvmlShutdown(void) { return 0; }
int nvmlDeviceGetCount_v2(unsigned int* c) { *c = 1; return 0; }
`
		badSO := buildMockNVMLLib(t, incompleteC)
		client, err := OpenWithLibPath(badSO)
		if err == nil {
			_ = client.Close()
			t.Fatal("expected error for incomplete NVML shared library")
		}
		if !errors.Is(err, ErrUnavailable) || !strings.Contains(err.Error(), "missing symbol") {
			t.Fatalf("expected ErrUnavailable missing symbol error, got: %v", err)
		}
	})
}
