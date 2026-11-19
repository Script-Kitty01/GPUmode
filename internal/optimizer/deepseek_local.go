package optimizer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"gpumode/internal/harness"
	"gpumode/internal/physics"
)

// ResolveDefaultModelPath returns the configured GGUF model path from GPUMODE_MODEL_PATH.
func ResolveDefaultModelPath() string {
	if v := strings.TrimSpace(os.Getenv("GPUMODE_MODEL_PATH")); v != "" {
		if _, err := os.Stat(v); err == nil {
			return v
		}
	}

	searchDirs := []string{}
	if dir := strings.TrimSpace(os.Getenv("GPUMODE_MODEL_DIR")); dir != "" {
		searchDirs = append(searchDirs, dir)
	}
	if home, err := os.UserHomeDir(); err == nil {
		searchDirs = append(searchDirs,
			filepath.Join(home, "AI", "Models"),
			filepath.Join(home, "models"),
			filepath.Join(home, ".cache", "huggingface", "hub"),
		)
	}
	searchDirs = append(searchDirs, "models")

	var fallback string
	for _, dir := range searchDirs {
		info, err := os.Stat(dir)
		if err != nil || !info.IsDir() {
			continue
		}
		var found []string
		_ = filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if entry.IsDir() {
				name := strings.ToLower(entry.Name())
				if name == ".git" || name == "node_modules" {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.EqualFold(filepath.Ext(entry.Name()), ".gguf") {
				found = append(found, path)
			}
			return nil
		})
		if len(found) == 0 {
			continue
		}

		for _, path := range found {
			name := strings.ToLower(filepath.Base(path))
			if strings.Contains(name, "deepseek") || strings.Contains(name, "llm") || strings.Contains(name, "qwen") {
				return path
			}
		}
		if fallback == "" {
			fallback = found[0]
		}
	}
	return fallback
}

// DeepSeekKernelProposal holds the structured JSON proposal returned by the local DeepSeek GGUF model.
type DeepSeekKernelProposal struct {
	BlockSize int    `json:"blockSize"`
	VecWidth  int    `json:"vecWidth"`
	Fused     bool   `json:"fused"`
	Reason    string `json:"reason"`
}

// LocalDeepSeekProposer invokes a local GGUF model via llama-cli in single-turn non-interactive mode
// (--single-turn --reasoning off) to propose a CUDA kernel configuration, then releases VRAM so nvcc
// and compute-sanitizer can benchmark the kernels without GPU memory contention.
type LocalDeepSeekProposer struct {
	LlamaCLIPath string
	ModelPath    string
	Hardware     physics.GPUHardwareSpec
	GPULayers    int
	Timeout      time.Duration
	Fallback     *DeterministicProposer
	LastProposal *DeepSeekKernelProposal
	LastPerfInfo string
}

// NewLocalDeepSeekProposer creates a live local GGUF proposer backed by llama-cli and a deterministic fallback.
func NewLocalDeepSeekProposer(llamaCLIPath, modelPath string) *LocalDeepSeekProposer {
	return NewLocalDeepSeekProposerForHardware(llamaCLIPath, modelPath, physics.GPUHardwareSpec{})
}

// NewLocalDeepSeekProposerForHardware creates a hardware-aware local GGUF proposer.
func NewLocalDeepSeekProposerForHardware(llamaCLIPath, modelPath string, hw physics.GPUHardwareSpec) *LocalDeepSeekProposer {
	if llamaCLIPath == "" {
		llamaCLIPath = "llama-cli"
	}
	if modelPath == "" {
		modelPath = ResolveDefaultModelPath()
	}
	ngl := 0
	if v := strings.TrimSpace(os.Getenv("GPUMODE_NGL")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			ngl = n
		}
	}
	if ngl == 0 {
		if hw.TotalVRAMMiB >= 12000 {
			ngl = 99
		} else if hw.TotalVRAMMiB > 0 {
			ngl = int(hw.TotalVRAMMiB / 256.0)
			if ngl < 8 {
				ngl = 8
			}
		} else {
			ngl = 24
		}
	}
	return &LocalDeepSeekProposer{
		LlamaCLIPath: llamaCLIPath,
		ModelPath:    modelPath,
		Hardware:     hw,
		GPULayers:    ngl,
		Timeout:      45 * time.Second,
		Fallback:     NewDeterministicProposerForHardware(hw),
	}
}

func (p *LocalDeepSeekProposer) Name() string {
	return fmt.Sprintf("Local GGUF KernelProposer (%s)", p.ModelPath)
}

var (
	reLlamaPerf = regexp.MustCompile(`\[\s*Prompt:\s*[0-9.]+\s*t/s\s*\|\s*Generation:\s*[0-9.]+\s*t/s\s*\]`)
	reJSONObj   = regexp.MustCompile(`\{[^{}]*"blockSize"[^{}]*\}`)
)

func (p *LocalDeepSeekProposer) ProposeCandidates(opCategory string, batchSize, hiddenDim, iterations int) (harness.KernelSpec, []harness.KernelSpec) {
	base, cands := p.Fallback.ProposeCandidates(opCategory, batchSize, hiddenDim, iterations)

	proposal, perf, err := p.queryLocalGGUF(opCategory, batchSize, hiddenDim)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[gpumode/deepseek] local GGUF query fallback (%v)\n", err)
		for i := range cands {
			cands[i].Name = "[GGUF Fallback] " + cands[i].Name
		}
		return base, cands
	}

	p.LastProposal = &proposal
	p.LastPerfInfo = perf
	fmt.Fprintf(os.Stderr, "[gpumode/deepseek] Local LLM proposed: BlockSize=%d, VecWidth=float%d, Fused=%v | Reason: %q %s\n",
		proposal.BlockSize, proposal.VecWidth, proposal.Fused, proposal.Reason, perf)

	// Normalize proposed blockSize & vecWidth to valid hardware-safe parameters
	targetBlock := proposal.BlockSize
	if targetBlock != 128 && targetBlock != 256 && targetBlock != 512 {
		targetBlock = 256
	}
	targetVec := proposal.VecWidth
	if targetVec != 1 && targetVec != 4 {
		targetVec = 4
	}

	// Tag and prioritize the LLM-selected kernel at the front of the candidate list
	matchedIdx := -1
	for i := range cands {
		if cands[i].BlockSize == targetBlock && cands[i].VecWidth == targetVec {
			matchedIdx = i
			break
		}
	}
	if matchedIdx >= 0 {
		reasonSuffix := ""
		if proposal.Reason != "" {
			reasonSuffix = fmt.Sprintf(" [%s]", proposal.Reason)
		}
		cands[matchedIdx].Name = fmt.Sprintf("[LLM Proposed] %s%s", cands[matchedIdx].Name, reasonSuffix)
		if matchedIdx > 0 {
			selected := cands[matchedIdx]
			copy(cands[1:matchedIdx+1], cands[0:matchedIdx])
			cands[0] = selected
		}
	}

	return base, cands
}

func (p *LocalDeepSeekProposer) queryLocalGGUF(opCategory string, batchSize, hiddenDim int) (DeepSeekKernelProposal, string, error) {
	if strings.TrimSpace(p.ModelPath) == "" {
		return DeepSeekKernelProposal{}, "", fmt.Errorf("no GGUF model path configured (set GPUMODE_MODEL_PATH or -deepseek-model)")
	}
	if _, err := os.Stat(p.ModelPath); err != nil {
		return DeepSeekKernelProposal{}, "", fmt.Errorf("GGUF model not found at %s: %w", p.ModelPath, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), p.Timeout)
	defer cancel()

	archDesc := "NVIDIA CUDA"
	if p.Hardware.ArchFlag != "" {
		if p.Hardware.Name != "" {
			archDesc = fmt.Sprintf("NVIDIA %s (%s, %d SMs, %.0f GB/s BW)",
				p.Hardware.Name, p.Hardware.ArchFlag, p.Hardware.SMCount, p.Hardware.PeakMemBandwidthGBps)
		} else {
			archDesc = "CUDA " + p.Hardware.ArchFlag
		}
	}

	prompt := fmt.Sprintf(
		`For a memory-bound %s %s kernel (batch=%d, hidden=%d), pick optimal blockSize (128, 256, or 512) and vecWidth (1 or 4). Output ONLY compact JSON: {"blockSize":256,"vecWidth":4,"fused":true,"reason":"short reason"}`,
		archDesc, opCategory, batchSize, hiddenDim,
	)

	args := []string{
		"-m", p.ModelPath,
		"-ngl", fmt.Sprintf("%d", p.GPULayers),
		"-c", "512",
		"-st",
		"--no-display-prompt",
		"--reasoning", "off",
		"-n", "80",
		"--temp", "0.1",
		"-p", prompt,
	}

	cmd := exec.CommandContext(ctx, p.LlamaCLIPath, args...)
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	if err := cmd.Run(); err != nil {
		return DeepSeekKernelProposal{}, "", fmt.Errorf("llama-cli failed: %w", err)
	}

	raw := outBuf.String()
	perf := ""
	if m := reLlamaPerf.FindString(raw); m != "" {
		perf = m
	}

	// Extract the last valid JSON object from llama-cli stdout (skipping any echoed prompt line)
	matches := reJSONObj.FindAllString(raw, -1)
	if len(matches) == 0 {
		return DeepSeekKernelProposal{}, perf, fmt.Errorf("no JSON object found in llama-cli output")
	}

	jsonChunk := matches[len(matches)-1]
	var prop DeepSeekKernelProposal
	if err := json.Unmarshal([]byte(jsonChunk), &prop); err != nil {
		return DeepSeekKernelProposal{}, perf, fmt.Errorf("invalid JSON from LLM (%s): %w", jsonChunk, err)
	}

	return prop, perf, nil
}
