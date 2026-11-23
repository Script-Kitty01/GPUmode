package tui

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"gpumode/internal/physics"
	"gpumode/internal/telemetry"
)

func TestExtractModelOrWorkloadFromArgs(t *testing.T) {
	cases := []struct {
		comm string
		args []string
		want string
	}{
		{
			comm: "llama-server",
			args: []string{"/usr/bin/llama-server", "-m", "/home/user/AI/Models/Qwen3.5-4B-Q4_K_M.gguf", "-ngl", "99", "--port", "8081"},
			want: "Model: Qwen3.5-4B-Q4_K_M.gguf",
		},
		{
			comm: "llama-cli",
			args: []string{"llama-cli", "--model=/models/DeepSeek-V4-9B.gguf", "-p", "hello"},
			want: "Model: DeepSeek-V4-9B.gguf",
		},
		{
			comm: "python3",
			args: []string{"/home/user/.venv/bin/python3", "scripts/profile_pytorch.py", "--batch", "256"},
			want: "PyTorch/Script: profile_pytorch.py",
		},
	}

	for _, tc := range cases {
		got := ExtractModelOrWorkloadFromArgs(tc.comm, tc.args)
		if got != tc.want {
			t.Errorf("ExtractModelOrWorkloadFromArgs(%q, %v) = %q, want %q", tc.comm, tc.args, got, tc.want)
		}
	}
}

func TestRenderSnapshotCardAndLiveTUI(t *testing.T) {
	hw := physics.GPUHardwareSpec{
		Name:           "NVIDIA GeForce RTX 4050 Laptop GPU",
		ArchFlag:       "sm_89",
		TotalVRAMMiB:   6141,
		MaxSMClockMHz:  3105,
		MaxMemClockMHz: 8001,
		TDPWatts:       65.0,
	}

	state := SnapshotState{
		Hardware:         hw,
		TelemetryBackend: "NVML",
		Sample: telemetry.Sample{
			GPUName:         hw.Name,
			PowerWatts:      52.4,
			SMClockMHz:      2355,
			MemClockMHz:     8001,
			TemperatureC:    56.0,
			GPUUtilPct:      94.0,
			MemUtilPct:      68.0,
			VRAMUsedMiB:     4242,
			VRAMTotalMiB:    6141,
			ThrottleReasons: []string{"SwPowerCap"},
			PCIeTxKBps:      51200,
			PCIeRxKBps:      102400,
		},
		PowerHistory:     []float64{12.0, 28.0, 45.0, 52.4},
		PeakPowerWatts:   54.1,
		CumulativeJoules: 98.6,
		ElapsedSec:       2.0,
		ActiveProcesses: []ActiveModelProcess{
			{
				PID:         12345,
				Comm:        "llama-server",
				ModelOrArgs: "Model: Qwen3.5-4B-Q4_K_M.gguf",
				UsedVRAMMiB: 3820,
			},
		},
	}

	card := RenderSnapshotCard(state, false)
	for _, want := range []string{
		"gpuMode CLI Telemetry Monitor",
		"NVIDIA GeForce RTX 4050 Laptop GPU",
		"52.40 W /  65.0 W TDP",
		"4242 /   6141 MiB",
		"98.60 J",
		"PID 12345",
		"llama-server",
		"Model: Qwen3.5-4B-Q4_K_M.gguf",
		"SwPowerCap",
	} {
		if !strings.Contains(card, want) {
			t.Errorf("expected RenderSnapshotCard output to contain %q, got:\n%s", want, card)
		}
	}

	// Test RunLiveTUI for 3 frames with a deterministic monitor sampler
	mon := telemetry.NewMonitor("nvidia-smi", 20*time.Millisecond, 32)
	defer mon.Close()
	mon.SetSampler(func(ctx context.Context) (telemetry.Sample, error) {
		return state.Sample, nil
	})

	var buf bytes.Buffer
	err := RunLiveTUI(context.Background(), &buf, LiveConfig{
		Hardware:  hw,
		Monitor:   mon,
		Interval:  50 * time.Millisecond,
		MaxFrames: 3,
		UseColor:  false,
	})
	if err != nil {
		t.Fatalf("RunLiveTUI failed: %v", err)
	}
	if !strings.Contains(buf.String(), "NVIDIA GeForce RTX 4050 Laptop GPU") {
		t.Fatalf("unexpected RunLiveTUI output:\n%s", buf.String())
	}
}
