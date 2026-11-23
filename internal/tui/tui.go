package tui

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gpumode/internal/ebpftrace"
	"gpumode/internal/nvml"
	"gpumode/internal/physics"
	"gpumode/internal/telemetry"
)

const (
	ansiReset        = "\033[0m"
	ansiBold         = "\033[1m"
	ansiCyan         = "\033[36m"
	ansiGreen        = "\033[32m"
	ansiYellow       = "\033[33m"
	ansiRed          = "\033[31m"
	ansiDim          = "\033[2m"
	ansiHome         = "\033[H"
	ansiClearLineEnd = "\033[K"
	ansiClearBelow   = "\033[J"
	ansiAltScreenOn  = "\033[?1049h\033[?25l\033[H\033[2J"
	ansiAltScreenOff = "\033[?25h\033[?1049l"
)

var sparkBlocks = []rune{'▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'}

// ActiveModelProcess describes a live process holding a GPU compute context and any detected AI model.
type ActiveModelProcess struct {
	PID         uint32  `json:"pid"`
	Comm        string  `json:"comm"`
	ModelOrArgs string  `json:"modelOrArgs"`
	UsedVRAMMiB float64 `json:"usedVramMib"`
}

// SnapshotState holds all data required to render a single frame of the terminal GPU monitor.
type SnapshotState struct {
	Hardware             physics.GPUHardwareSpec `json:"hardware"`
	TelemetryBackend     string                  `json:"telemetryBackend"`
	Sample               telemetry.Sample        `json:"sample"`
	PowerHistory         []float64               `json:"powerHistory"`
	PeakPowerWatts       float64                 `json:"peakPowerWatts"`
	CumulativeJoules     float64                 `json:"cumulativeJoules"`
	ElapsedSec           float64                 `json:"elapsedSec"`
	ActiveProcesses      []ActiveModelProcess    `json:"activeProcesses"`
	EBPFSummary          *ebpftrace.Summary      `json:"ebpfSummary,omitempty"`
	InteractiveFrameMode bool                    `json:"interactiveFrameMode"`
}

// LiveConfig configures the live refreshing terminal monitor (gpumode top / watch).
type LiveConfig struct {
	Hardware  physics.GPUHardwareSpec
	Monitor   *telemetry.Monitor
	Interval  time.Duration
	MaxFrames int // 0 = run until ctx is canceled (Ctrl+C)
	UseColor  bool
	EBPFMode  ebpftrace.Mode
}

// InspectComputeProcesses enriches raw NVML compute PIDs with process names and detected .gguf / Python model names from /proc.
func InspectComputeProcesses(rawProcs []nvml.ComputeProcess) []ActiveModelProcess {
	if len(rawProcs) == 0 {
		return nil
	}
	out := make([]ActiveModelProcess, 0, len(rawProcs))
	for _, p := range rawProcs {
		comm, modelDesc := readProcCommandAndModel(p.PID)
		out = append(out, ActiveModelProcess{
			PID:         p.PID,
			Comm:        comm,
			ModelOrArgs: modelDesc,
			UsedVRAMMiB: p.UsedVRAMMiB,
		})
	}
	return out
}

func readProcCommandAndModel(pid uint32) (comm string, modelOrArgs string) {
	procDir := fmt.Sprintf("/proc/%d", pid)
	if b, err := os.ReadFile(filepath.Join(procDir, "comm")); err == nil {
		comm = strings.TrimSpace(string(b))
	}
	if comm == "" {
		comm = "cuda-proc"
	}

	rawCmd, err := os.ReadFile(filepath.Join(procDir, "cmdline"))
	var args []string
	if err == nil && len(rawCmd) > 0 {
		parts := bytes.Split(rawCmd, []byte{0})
		args = make([]string, 0, len(parts))
		for _, part := range parts {
			s := strings.TrimSpace(string(part))
			if s != "" {
				args = append(args, s)
			}
		}
	}

	desc := ExtractModelOrWorkloadFromArgs(comm, args)

	// 1. If Ollama is running with a sha256 blob path, query Ollama /api/ps for the human-readable model tag (e.g. qwen2.5:7b)
	if strings.Contains(strings.ToLower(comm), "ollama") || strings.Contains(desc, "sha256-") {
		if ollamaModel := queryOllamaRunningModel(); ollamaModel != "" {
			return comm, truncateStr("Model: "+ollamaModel, 38)
		}
	}

	// 2. If the process is a Python/C++ process that opened a .gguf/.safetensors/.pt file in code (not CLI args),
	// inspect /proc/<pid>/fd/* symlinks to auto-detect the open model file on disk!
	if !strings.HasPrefix(desc, "Model:") {
		if openModel := findOpenModelFileInProcFD(procDir); openModel != "" {
			return comm, truncateStr("Model: "+openModel, 38)
		}
	}

	return comm, desc
}

// findOpenModelFileInProcFD inspects /proc/<pid>/fd/* symlinks (and /proc/<pid>/maps) to find any
// memory-mapped or open .gguf, .safetensors, or .bin model weights file even when loaded via Python code.
func findOpenModelFileInProcFD(procDir string) string {
	entries, err := os.ReadDir(filepath.Join(procDir, "fd"))
	if err == nil {
		limit := len(entries)
		if limit > 128 {
			limit = 128
		}
		for i := 0; i < limit; i++ {
			target, lErr := os.Readlink(filepath.Join(procDir, "fd", entries[i].Name()))
			if lErr != nil {
				continue
			}
			base := filepath.Base(target)
			lower := strings.ToLower(base)
			if strings.HasSuffix(lower, ".gguf") || strings.HasSuffix(lower, ".safetensors") {
				return base
			}
		}
	}
	if mapsRaw, mErr := os.ReadFile(filepath.Join(procDir, "maps")); mErr == nil {
		for _, line := range strings.Split(string(mapsRaw), "\n") {
			lower := strings.ToLower(line)
			if strings.Contains(lower, ".gguf") || strings.Contains(lower, ".safetensors") {
				fields := strings.Fields(line)
				if len(fields) >= 6 {
					return filepath.Base(fields[len(fields)-1])
				}
			}
		}
	}
	return ""
}

func queryOllamaRunningModel() string {
	client := &http.Client{Timeout: 120 * time.Millisecond}
	resp, err := client.Get("http://127.0.0.1:11434/api/ps")
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	var payload struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil || len(payload.Models) == 0 {
		return ""
	}
	if payload.Models[0].Name != "" {
		return payload.Models[0].Name
	}
	return payload.Models[0].Model
}

// ExtractModelOrWorkloadFromArgs parses command-line tokens to highlight any active .gguf model,
// --model/-m flag, or Python workload script running on the GPU.
func ExtractModelOrWorkloadFromArgs(comm string, args []string) string {
	const maxCol = 36
	if len(args) == 0 {
		return truncateStr(comm, maxCol)
	}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if (a == "-m" || a == "--model") && i+1 < len(args) {
			modelBase := filepath.Base(args[i+1])
			return truncateStr(fmt.Sprintf("Model: %s", modelBase), maxCol)
		}
		if strings.HasPrefix(a, "--model=") {
			return truncateStr(fmt.Sprintf("Model: %s", filepath.Base(strings.TrimPrefix(a, "--model="))), maxCol)
		}
		if strings.HasSuffix(strings.ToLower(a), ".gguf") || strings.HasSuffix(strings.ToLower(a), ".safetensors") {
			return truncateStr(fmt.Sprintf("Model: %s", filepath.Base(a)), maxCol)
		}
	}
	for _, a := range args {
		if strings.HasSuffix(strings.ToLower(a), ".py") {
			return truncateStr(fmt.Sprintf("PyTorch/Script: %s", filepath.Base(a)), maxCol)
		}
	}
	fullJoined := strings.Join(args, " ")
	lowerComm := strings.ToLower(comm)
	if lowerComm == "code" || strings.Contains(fullJoined, "--gpu-preferences=") || strings.Contains(fullJoined, "--type=gpu-process") {
		return "VS Code / Electron (GPU)"
	}
	if lowerComm == "xorg" || lowerComm == "gnome-shell" || lowerComm == "kwin_wayland" || lowerComm == "hyprland" {
		return "Desktop Display Compositor"
	}
	baseBin := filepath.Base(args[0])
	if idx := strings.IndexByte(baseBin, ' '); idx > 0 {
		baseBin = baseBin[:idx]
	}
	if len(args) > 1 {
		joined := baseBin + " " + strings.Join(args[1:], " ")
		return truncateStr(joined, maxCol)
	}
	return truncateStr(baseBin, maxCol)
}

// RenderProgressBar renders a fixed-width Unicode bar [██████░░░░] with percentage.
func RenderProgressBar(pct float64, width int) string {
	if width <= 0 {
		width = 22
	}
	clamped := math.Max(0, math.Min(100, pct))
	filled := int(math.Round((clamped / 100.0) * float64(width)))
	if filled > width {
		filled = width
	}
	return "[" + strings.Repeat("█", filled) + strings.Repeat("░", width-filled) + "]"
}

// RenderSparkline converts a slice of power samples into a Unicode sparkline (▁▂▃▄▅▆▇█).
func RenderSparkline(values []float64, maxCols int) string {
	if len(values) == 0 {
		return "—"
	}
	if maxCols > 0 && len(values) > maxCols {
		values = values[len(values)-maxCols:]
	}
	minV, maxV := values[0], values[0]
	for _, v := range values[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	span := maxV - minV
	if span < 1.0 {
		span = 1.0
	}
	var b strings.Builder
	for _, v := range values {
		norm := (v - minV) / span
		idx := int(math.Round(norm * float64(len(sparkBlocks)-1)))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(sparkBlocks) {
			idx = len(sparkBlocks) - 1
		}
		b.WriteRune(sparkBlocks[idx])
	}
	return b.String()
}

// RenderSnapshotCard renders a complete terminal GPU & active-model telemetry card.
func RenderSnapshotCard(st SnapshotState, useColor bool) string {
	c := func(code, text string) string {
		if !useColor {
			return text
		}
		return code + text + ansiReset
	}

	s := st.Sample
	hw := st.Hardware
	gpuName := s.GPUName
	if gpuName == "" {
		gpuName = hw.Name
	}
	if gpuName == "" {
		gpuName = "NVIDIA CUDA GPU"
	}
	arch := hw.ArchFlag
	if arch == "" {
		arch = "cuda"
	}
	backend := st.TelemetryBackend
	if backend == "" {
		backend = s.Source
	}
	if backend == "" {
		backend = "NVML"
	}

	tdpW := hw.TDPWatts
	if tdpW <= 0 {
		tdpW = 65.0
	}
	powerPct := (s.PowerWatts / tdpW) * 100.0
	vramTotal := s.VRAMTotalMiB
	if vramTotal <= 0 {
		vramTotal = hw.TotalVRAMMiB
	}
	vramPct := 0.0
	if vramTotal > 0 {
		vramPct = (s.VRAMUsedMiB / vramTotal) * 100.0
	}

	throttleText := c(ansiGreen, "NONE (Full Boost)")
	if len(s.ThrottleReasons) > 0 {
		joined := strings.Join(s.ThrottleReasons, ", ")
		if joined == "GpuIdle" {
			throttleText = c(ansiCyan, "GpuIdle (Low-Power P-State)")
		} else {
			throttleText = c(ansiYellow, joined)
		}
	}

	var b strings.Builder
	b.WriteString("====================================================================================\n")
	b.WriteString(fmt.Sprintf(" %s — %s (%s) [%s]\n",
		c(ansiBold+ansiCyan, "gpuMode CLI Telemetry Monitor"),
		c(ansiBold, gpuName),
		arch,
		c(ansiGreen, backend),
	))
	b.WriteString("====================================================================================\n")

	// Gauges
	b.WriteString(fmt.Sprintf("  GPU SM Compute  : %s %5.1f%%  | SM Clock  : %4.0f / %4.0f MHz\n",
		c(colorForPct(s.GPUUtilPct, false), RenderProgressBar(s.GPUUtilPct, 22)),
		s.GPUUtilPct, s.SMClockMHz, math.Max(s.SMClockMHz, hw.MaxSMClockMHz),
	))
	b.WriteString(fmt.Sprintf("  Memory Bus Util : %s %5.1f%%  | Mem Clock : %4.0f / %4.0f MHz\n",
		c(colorForPct(s.MemUtilPct, false), RenderProgressBar(s.MemUtilPct, 22)),
		s.MemUtilPct, s.MemClockMHz, math.Max(s.MemClockMHz, hw.MaxMemClockMHz),
	))
	b.WriteString(fmt.Sprintf("  VRAM Footprint  : %s %5.1f%%  | %6.0f / %6.0f MiB\n",
		c(colorForPct(vramPct, true), RenderProgressBar(vramPct, 22)),
		vramPct, s.VRAMUsedMiB, vramTotal,
	))
	peakW := math.Max(s.PowerWatts, st.PeakPowerWatts)
	b.WriteString(fmt.Sprintf("  Board Power Draw: %s %5.1f%%  | %5.2f W / %5.1f W TDP (Peak: %.2f W)\n",
		c(colorForPct(powerPct, true), RenderProgressBar(powerPct, 22)),
		powerPct, s.PowerWatts, tdpW, peakW,
	))

	pcieTxMB := float64(s.PCIeTxKBps) / 1024.0
	pcieRxMB := float64(s.PCIeRxKBps) / 1024.0
	b.WriteString(fmt.Sprintf("  Thermal & PCIe  : Temp: %.1f °C | PCIe: ↑%.1f ↓%.1f MB/s | Throttle: %s\n",
		s.TemperatureC, pcieTxMB, pcieRxMB, throttleText,
	))

	// Power sparkline & live integrated energy
	b.WriteString("------------------------------------------------------------------------------------\n")
	spark := RenderSparkline(st.PowerHistory, 36)
	wh := st.CumulativeJoules / 3600.0
	b.WriteString(fmt.Sprintf("  Power Trend     : %s  | Session Energy: %s (%.4f Wh in %.1fs)\n",
		c(ansiCyan, spark),
		c(ansiBold, fmt.Sprintf("%.2f J", st.CumulativeJoules)),
		wh,
		st.ElapsedSec,
	))

	// Active GPU processes & running AI models
	b.WriteString("------------------------------------------------------------------------------------\n")
	b.WriteString(fmt.Sprintf("  %s\n", c(ansiBold, "Active GPU Compute Processes & Running AI Models:")))
	if len(st.ActiveProcesses) == 0 {
		b.WriteString(fmt.Sprintf("    %s\n", c(ansiDim, "• No active external CUDA compute processes (GPU idle)")))
	} else {
		for _, proc := range st.ActiveProcesses {
			vramTag := "shared/ctx"
			if proc.UsedVRAMMiB > 0 {
				vramTag = fmt.Sprintf("%6.0f MiB VRAM", proc.UsedVRAMMiB)
			}
			b.WriteString(fmt.Sprintf("    • PID %-7d | %-14s | %-15s | %s\n",
				proc.PID,
				truncateStr(proc.Comm, 14),
				vramTag,
				c(ansiGreen, proc.ModelOrArgs),
			))
		}
	}

	if st.EBPFSummary != nil && st.EBPFSummary.Active {
		eb := st.EBPFSummary
		b.WriteString("------------------------------------------------------------------------------------\n")
		b.WriteString(fmt.Sprintf("  Host ↔ CUDA eBPF: %d launches (%.0f/s) | Gap: %.1f µs avg / %.1f µs p95 | H2D/D2H: %.2f/%.2f MiB | Sync: %.2f ms\n",
			eb.KernelLaunches, eb.LaunchesPerSec, eb.AvgLaunchGapUS, eb.P95LaunchGapUS,
			eb.MemcpyH2DMiB, eb.MemcpyD2HMiB, eb.SyncStallMS,
		))
	}

	b.WriteString("====================================================================================\n")
	if st.InteractiveFrameMode {
		b.WriteString(fmt.Sprintf("  %s\n", c(ansiDim, "Press Ctrl+C to exit live monitor • Runs with 0 MiB CUDA VRAM overhead alongside active models")))
	}
	return b.String()
}

// RunLiveTUI runs the live refreshing terminal dashboard in-place until ctx is canceled or cfg.MaxFrames is reached.
func RunLiveTUI(ctx context.Context, out io.Writer, cfg LiveConfig) error {
	if cfg.Monitor == nil {
		return fmt.Errorf("telemetry monitor is nil")
	}
	interval := cfg.Interval
	if interval < 50*time.Millisecond {
		interval = 200 * time.Millisecond
	}

	useInPlaceScreen := cfg.UseColor && cfg.MaxFrames != 1
	if useInPlaceScreen {
		// Enter Alternate Screen Buffer & hide cursor so updates happen 100% in-place without scrollback duplication
		_, _ = io.WriteString(out, ansiAltScreenOn)
		defer func() {
			_, _ = io.WriteString(out, ansiAltScreenOff)
		}()
	}

	start := time.Now()
	prevTime := start
	var prevPower float64
	var cumulativeJoules float64
	var peakPower float64
	var powerHistory []float64
	var lastRenderedCard string
	var prevLineCount int

	renderFrame := func(frameIdx int) error {
		sample, err := cfg.Monitor.PollOnce(ctx)
		if err != nil && frameIdx == 0 {
			return err
		}
		now := time.Now()
		if frameIdx > 0 && sample.PowerWatts > 0 {
			dt := now.Sub(prevTime).Seconds()
			if dt > 0 && dt < 10.0 {
				cumulativeJoules += 0.5 * (prevPower + sample.PowerWatts) * dt
			}
		}
		prevTime = now
		prevPower = sample.PowerWatts
		if sample.PowerWatts > peakPower {
			peakPower = sample.PowerWatts
		}
		if sample.PowerWatts > 0 {
			powerHistory = append(powerHistory, sample.PowerWatts)
			if len(powerHistory) > 36 {
				powerHistory = powerHistory[len(powerHistory)-36:]
			}
		}

		procs := InspectComputeProcesses(cfg.Monitor.ActiveComputeProcesses())
		state := SnapshotState{
			Hardware:             cfg.Hardware,
			TelemetryBackend:     cfg.Monitor.Backend(),
			Sample:               sample,
			PowerHistory:         powerHistory,
			PeakPowerWatts:       peakPower,
			CumulativeJoules:     cumulativeJoules,
			ElapsedSec:           now.Sub(start).Seconds(),
			ActiveProcesses:      procs,
			InteractiveFrameMode: cfg.MaxFrames == 0,
		}

		card := RenderSnapshotCard(state, cfg.UseColor)
		lastRenderedCard = card
		lines := strings.Split(strings.TrimSuffix(card, "\n"), "\n")

		if useInPlaceScreen {
			var frameBuf strings.Builder
			if prevLineCount > 0 {
				// Rewind cursor up by exact line count of the previous frame AND home on alt screen
				frameBuf.WriteString(fmt.Sprintf("\r\033[%dA", prevLineCount))
			}
			frameBuf.WriteString(ansiHome)
			for _, line := range lines {
				frameBuf.WriteString("\r\033[2K")
				frameBuf.WriteString(line)
				frameBuf.WriteByte('\n')
			}
			frameBuf.WriteString(ansiClearBelow)
			prevLineCount = len(lines)
			_, writeErr := io.WriteString(out, frameBuf.String())
			return writeErr
		}
		prevLineCount = len(lines)
		_, writeErr := io.WriteString(out, card)
		return writeErr
	}

	if err := renderFrame(0); err != nil {
		return err
	}
	if cfg.MaxFrames == 1 {
		return nil
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	frame := 1
	for {
		select {
		case <-ctx.Done():
			if useInPlaceScreen && lastRenderedCard != "" {
				// Leave final clean snapshot visible after exiting alternate screen
				defer func(finalCard string) {
					_, _ = io.WriteString(out, finalCard)
				}(lastRenderedCard)
			}
			return nil
		case <-ticker.C:
			if err := renderFrame(frame); err != nil {
				return err
			}
			frame++
			if cfg.MaxFrames > 0 && frame >= cfg.MaxFrames {
				return nil
			}
		}
	}
}

func truncateStr(s string, maxLen int) string {
	if maxLen <= 0 || len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

func colorForPct(pct float64, highIsWarn bool) string {
	if !highIsWarn {
		if pct >= 60 {
			return ansiGreen
		}
		if pct >= 20 {
			return ansiCyan
		}
		return ansiDim
	}
	if pct >= 90 {
		return ansiRed
	}
	if pct >= 75 {
		return ansiYellow
	}
	return ansiGreen
}
