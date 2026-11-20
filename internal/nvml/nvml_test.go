package nvml

import (
	"errors"
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
