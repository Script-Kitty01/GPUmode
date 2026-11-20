package config

import "testing"

func TestObservabilityConfigEnvResolution(t *testing.T) {
	t.Setenv("GPUMODE_TELEMETRY", "")
	t.Setenv("GPUMODE_EBPF", "")
	t.Setenv("GPUMODE_TRACE_PID", "")

	def := DefaultDaemonConfig()
	if def.TelemetryBackend != "auto" {
		t.Fatalf("expected default TelemetryBackend=auto, got %q", def.TelemetryBackend)
	}
	if def.EBPFMode != "auto" {
		t.Fatalf("expected default EBPFMode=auto, got %q", def.EBPFMode)
	}
	if def.TracePID != 0 {
		t.Fatalf("expected default TracePID=0, got %d", def.TracePID)
	}

	t.Setenv("GPUMODE_TELEMETRY", "NVML")
	t.Setenv("GPUMODE_EBPF", "ON")
	t.Setenv("GPUMODE_TRACE_PID", "4242")

	custom := DefaultDaemonConfig()
	if custom.TelemetryBackend != "nvml" {
		t.Fatalf("expected TelemetryBackend=nvml, got %q", custom.TelemetryBackend)
	}
	if custom.EBPFMode != "on" {
		t.Fatalf("expected EBPFMode=on, got %q", custom.EBPFMode)
	}
	if custom.TracePID != 4242 {
		t.Fatalf("expected TracePID=4242, got %d", custom.TracePID)
	}

	t.Setenv("GPUMODE_TELEMETRY", "invalid")
	t.Setenv("GPUMODE_EBPF", "invalid")
	t.Setenv("GPUMODE_TRACE_PID", "-5")

	fallback := DefaultDaemonConfig()
	if fallback.TelemetryBackend != "auto" {
		t.Fatalf("expected invalid GPUMODE_TELEMETRY to fall back to auto, got %q", fallback.TelemetryBackend)
	}
	if fallback.EBPFMode != "auto" {
		t.Fatalf("expected invalid GPUMODE_EBPF to fall back to auto, got %q", fallback.EBPFMode)
	}
	if fallback.TracePID != 0 {
		t.Fatalf("expected negative GPUMODE_TRACE_PID to fall back to 0, got %d", fallback.TracePID)
	}
}
