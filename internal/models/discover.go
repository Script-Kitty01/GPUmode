package models

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Source constants identify where a local AI model was discovered.
const (
	SourceGGUFLocal   = "gguf_local"
	SourceHFCache     = "hf_cache"
	SourceLlamaServer = "llama_server"
	SourceOllama      = "ollama"
)

// LocalModel describes a discovered AI model on disk or served by a local endpoint.
type LocalModel struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Source       string `json:"source"`
	SourceLabel  string `json:"sourceLabel"`
	Path         string `json:"path,omitempty"`
	EndpointURL  string `json:"endpointUrl,omitempty"`
	SizeBytes    int64  `json:"sizeBytes,omitempty"`
	SizeHuman    string `json:"sizeHuman,omitempty"`
	Quantization string `json:"quantization,omitempty"`
	Running      bool   `json:"running"`
}

// DiscoverOptions controls which directories and local inference endpoints are scanned.
type DiscoverOptions struct {
	CustomDir     string
	ExtraGGUFDirs []string
	OpenAIURL     string
	OllamaURL     string
	SkipProbes    bool
	ProbeTimeout  time.Duration
}

// DiscoveryResult summarizes all discovered local models and scanned locations.
type DiscoveryResult struct {
	Models           []LocalModel `json:"models"`
	ScannedDirs      []string     `json:"scannedDirs"`
	CheckedEndpoints []string     `json:"checkedEndpoints"`
	Notes            []string     `json:"notes,omitempty"`
}

var reQuant = regexp.MustCompile(`(?i)\b(Q[2-8]_[0-9A-Z_]+|IQ[1-4]_[A-Z]+|F16|BF16|F32)\b`)

// ResolveDirectModelPath resolves a user-supplied file or folder path (including ~/... paths)
// directly to a runnable LocalModel without requiring the file to live under ~/models or ~/AI/Models.
func ResolveDirectModelPath(rawPath string) (LocalModel, error) {
	cleaned := expandHome(strings.TrimSpace(rawPath))
	if cleaned == "" {
		return LocalModel{}, fmt.Errorf("direct model path is empty")
	}
	absPath, err := filepath.Abs(cleaned)
	if err != nil {
		absPath = cleaned
	}
	info, err := os.Stat(absPath)
	if err != nil {
		return LocalModel{}, fmt.Errorf("model path not accessible (%s): %w", absPath, err)
	}

	targetFile := absPath
	targetInfo := info
	if info.IsDir() {
		var candidates []string
		_ = filepath.WalkDir(absPath, func(path string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if entry.IsDir() {
				name := strings.ToLower(entry.Name())
				if name == ".git" || name == "node_modules" || name == ".locks" {
					return filepath.SkipDir
				}
				return nil
			}
			baseLower := strings.ToLower(entry.Name())
			if strings.HasSuffix(baseLower, ".gguf") && !strings.Contains(baseLower, "mmproj") {
				candidates = append(candidates, path)
			}
			return nil
		})
		if len(candidates) == 0 {
			return LocalModel{}, fmt.Errorf("no .gguf model file found inside directory %s", absPath)
		}
		sort.Strings(candidates)
		targetFile = candidates[0]
		fi, statErr := os.Stat(targetFile)
		if statErr != nil {
			return LocalModel{}, fmt.Errorf("model file not accessible (%s): %w", targetFile, statErr)
		}
		targetInfo = fi
	} else {
		if !strings.EqualFold(filepath.Ext(targetFile), ".gguf") {
			return LocalModel{}, fmt.Errorf("direct model file must be a .gguf file (got %s)", filepath.Base(targetFile))
		}
	}

	if targetInfo.Size() == 0 {
		return LocalModel{}, fmt.Errorf("model file is empty: %s", targetFile)
	}

	base := filepath.Base(targetFile)
	return LocalModel{
		ID:           makeModelID(SourceGGUFLocal, targetFile),
		Name:         strings.TrimSuffix(base, filepath.Ext(base)),
		Source:       SourceGGUFLocal,
		SourceLabel:  "Direct Path",
		Path:         targetFile,
		SizeBytes:    targetInfo.Size(),
		SizeHuman:    formatBytesHuman(targetInfo.Size()),
		Quantization: parseQuantization(base),
		Running:      false,
	}, nil
}

// DiscoverLocalModels scans local GGUF folders, a custom repo/folder path, the Hugging Face cache,
// and running OpenAI-compatible (llama-server) / Ollama endpoints.
func DiscoverLocalModels(ctx context.Context, opts DiscoverOptions) DiscoveryResult {
	if ctx == nil {
		ctx = context.Background()
	}
	timeout := opts.ProbeTimeout
	if timeout <= 0 {
		timeout = 900 * time.Millisecond
	}

	res := DiscoveryResult{
		Models: []LocalModel{},
	}

	seenFiles := make(map[string]struct{})
	seenDirs := make(map[string]struct{})

	addGGUFFile := func(path, source, label string) {
		abs, err := filepath.Abs(path)
		if err != nil {
			abs = path
		}
		realPath := abs
		if resolved, err := filepath.EvalSymlinks(abs); err == nil {
			realPath = resolved
		}
		if _, dup := seenFiles[realPath]; dup {
			return
		}
		info, err := os.Stat(abs)
		if err != nil || info.IsDir() || info.Size() == 0 {
			return
		}
		// Skip mmproj / split helper files if obviously auxiliary? No, keep GGUFs except obvious mmproj.
		base := filepath.Base(abs)
		if strings.Contains(strings.ToLower(base), "mmproj") {
			return
		}
		seenFiles[realPath] = struct{}{}

		name := strings.TrimSuffix(base, filepath.Ext(base))
		if source == SourceHFCache {
			if repoName := extractHFRepoName(abs); repoName != "" {
				name = fmt.Sprintf("%s (%s)", name, repoName)
			}
		}

		res.Models = append(res.Models, LocalModel{
			ID:           makeModelID(source, abs),
			Name:         name,
			Source:       source,
			SourceLabel:  label,
			Path:         abs,
			SizeBytes:    info.Size(),
			SizeHuman:    formatBytesHuman(info.Size()),
			Quantization: parseQuantization(base),
			Running:      false,
		})
	}

	scanDir := func(dir, source, label string) {
		dir = expandHome(strings.TrimSpace(dir))
		if dir == "" {
			return
		}
		absDir, err := filepath.Abs(dir)
		if err != nil {
			absDir = dir
		}
		if _, done := seenDirs[absDir]; done {
			return
		}
		info, err := os.Stat(absDir)
		if err != nil {
			if opts.CustomDir != "" && dir == expandHome(strings.TrimSpace(opts.CustomDir)) {
				res.Notes = append(res.Notes, fmt.Sprintf("Custom path not accessible: %s", absDir))
			}
			return
		}
		if !info.IsDir() {
			if strings.EqualFold(filepath.Ext(absDir), ".gguf") {
				seenDirs[absDir] = struct{}{}
				res.ScannedDirs = append(res.ScannedDirs, absDir)
				addGGUFFile(absDir, source, label)
			}
			return
		}
		seenDirs[absDir] = struct{}{}
		res.ScannedDirs = append(res.ScannedDirs, absDir)

		_ = filepath.WalkDir(absDir, func(path string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if entry.IsDir() {
				name := strings.ToLower(entry.Name())
				if name == ".git" || name == "node_modules" || name == ".locks" {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.EqualFold(filepath.Ext(entry.Name()), ".gguf") {
				addGGUFFile(path, source, label)
			}
			return nil
		})
	}

	// 1. Custom directory / repo path from UI (highest priority if supplied)
	if strings.TrimSpace(opts.CustomDir) != "" {
		scanDir(opts.CustomDir, SourceGGUFLocal, "Custom Folder")
	}

	// 2. Explicit GPUMODE_MODEL_PATH file if configured
	if envFile := strings.TrimSpace(os.Getenv("GPUMODE_MODEL_PATH")); envFile != "" {
		addGGUFFile(envFile, SourceGGUFLocal, "Configured GGUF")
	}

	// 3. Configured & standard GGUF directories
	var ggufDirs []string
	if envDir := strings.TrimSpace(os.Getenv("GPUMODE_MODEL_DIR")); envDir != "" {
		ggufDirs = append(ggufDirs, envDir)
	}
	if home, err := os.UserHomeDir(); err == nil {
		ggufDirs = append(ggufDirs,
			filepath.Join(home, "AI", "Models"),
			filepath.Join(home, "models"),
		)
	}
	ggufDirs = append(ggufDirs, "models")
	ggufDirs = append(ggufDirs, opts.ExtraGGUFDirs...)
	for _, d := range ggufDirs {
		scanDir(d, SourceGGUFLocal, "Local GGUF")
	}

	// 4. Hugging Face cache
	if home, err := os.UserHomeDir(); err == nil {
		scanDir(filepath.Join(home, ".cache", "huggingface", "hub"), SourceHFCache, "HF Cache")
	}

	// 5. Running local inference endpoints (llama-server / OpenAI-compatible & Ollama)
	if !opts.SkipProbes {
		openAIURL := strings.TrimSpace(opts.OpenAIURL)
		if openAIURL == "" {
			openAIURL = "http://127.0.0.1:8081"
		}
		openAIURL = strings.TrimRight(openAIURL, "/")
		res.CheckedEndpoints = append(res.CheckedEndpoints, openAIURL)
		for _, m := range probeOpenAIModels(ctx, openAIURL, timeout) {
			res.Models = append(res.Models, m)
		}

		ollamaURL := strings.TrimSpace(opts.OllamaURL)
		if ollamaURL == "" {
			ollamaURL = "http://127.0.0.1:11434"
		}
		ollamaURL = strings.TrimRight(ollamaURL, "/")
		res.CheckedEndpoints = append(res.CheckedEndpoints, ollamaURL)
		for _, m := range probeOllamaModels(ctx, ollamaURL, timeout) {
			res.Models = append(res.Models, m)
		}
	}

	sort.SliceStable(res.Models, func(i, j int) bool {
		// Prefer local GGUF first (supports full baseline vs gpuMode tuning), then running endpoints, then HF cache.
		rank := func(m LocalModel) int {
			switch m.Source {
			case SourceGGUFLocal:
				return 0
			case SourceHFCache:
				return 1
			case SourceLlamaServer:
				return 2
			case SourceOllama:
				return 3
			default:
				return 4
			}
		}
		ri, rj := rank(res.Models[i]), rank(res.Models[j])
		if ri != rj {
			return ri < rj
		}
		return strings.ToLower(res.Models[i].Name) < strings.ToLower(res.Models[j].Name)
	})

	return res
}

func probeOpenAIModels(ctx context.Context, baseURL string, timeout time.Duration) []LocalModel {
	client := &http.Client{Timeout: timeout}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/models", nil)
	if err != nil {
		return nil
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var parsed struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil
	}
	out := make([]LocalModel, 0, len(parsed.Data))
	for _, item := range parsed.Data {
		id := strings.TrimSpace(item.ID)
		if id == "" {
			continue
		}
		out = append(out, LocalModel{
			ID:          makeModelID(SourceLlamaServer, baseURL+"#"+id),
			Name:        id,
			Source:      SourceLlamaServer,
			SourceLabel: "Running Server",
			EndpointURL: baseURL,
			Running:     true,
		})
	}
	return out
}

func probeOllamaModels(ctx context.Context, baseURL string, timeout time.Duration) []LocalModel {
	client := &http.Client{Timeout: timeout}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/tags", nil)
	if err != nil {
		return nil
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var parsed struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil
	}
	out := make([]LocalModel, 0, len(parsed.Models))
	for _, item := range parsed.Models {
		name := strings.TrimSpace(item.Name)
		if name == "" {
			continue
		}
		out = append(out, LocalModel{
			ID:          makeModelID(SourceOllama, baseURL+"#"+name),
			Name:        name,
			Source:      SourceOllama,
			SourceLabel: "Ollama",
			EndpointURL: baseURL,
			SizeBytes:   item.Size,
			SizeHuman:   formatBytesHuman(item.Size),
			Running:     true,
		})
	}
	return out
}

func extractHFRepoName(absPath string) string {
	parts := strings.Split(filepath.ToSlash(absPath), "/")
	for _, p := range parts {
		if strings.HasPrefix(p, "models--") {
			trimmed := strings.TrimPrefix(p, "models--")
			return strings.Replace(trimmed, "--", "/", 1)
		}
	}
	return ""
}

func expandHome(p string) string {
	if p == "" || p[0] != '~' {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return p
	}
	if p == "~" {
		return home
	}
	if strings.HasPrefix(p, "~/") {
		return filepath.Join(home, p[2:])
	}
	return p
}

func makeModelID(source, key string) string {
	h := sha1.Sum([]byte(source + ":" + key))
	return source + "_" + hex.EncodeToString(h[:6])
}

func parseQuantization(name string) string {
	m := reQuant.FindString(name)
	return strings.ToUpper(m)
}

func formatBytesHuman(sz int64) string {
	if sz <= 0 {
		return ""
	}
	const giB = 1024.0 * 1024.0 * 1024.0
	const miB = 1024.0 * 1024.0
	if float64(sz) >= giB {
		return fmt.Sprintf("%.1f GiB", float64(sz)/giB)
	}
	return fmt.Sprintf("%.0f MiB", float64(sz)/miB)
}
