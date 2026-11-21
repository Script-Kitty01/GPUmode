package models

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDiscoverLocalModelsMultiSourceAndDedup(t *testing.T) {
	t.Setenv("GPUMODE_MODEL_PATH", "")
	t.Setenv("GPUMODE_MODEL_DIR", "")

	tmp := t.TempDir()
	homeDir := filepath.Join(tmp, "home")
	t.Setenv("HOME", homeDir)

	customDir := filepath.Join(tmp, "repo-models")
	if err := os.MkdirAll(customDir, 0o755); err != nil {
		t.Fatalf("MkdirAll customDir: %v", err)
	}
	modelA := filepath.Join(customDir, "DeepSeek-V4-Q4_K_M.gguf")
	modelB := filepath.Join(customDir, "Qwen2.5-7B-Q8_0.gguf")
	mmproj := filepath.Join(customDir, "mmproj-model-f16.gguf")
	if err := os.WriteFile(modelA, []byte("gguf-a"), 0o600); err != nil {
		t.Fatalf("write modelA: %v", err)
	}
	if err := os.WriteFile(modelB, []byte("gguf-b"), 0o600); err != nil {
		t.Fatalf("write modelB: %v", err)
	}
	if err := os.WriteFile(mmproj, []byte("mmproj"), 0o600); err != nil {
		t.Fatalf("write mmproj: %v", err)
	}

	// Simulate HF cache with a blob + symlink snapshot (should deduplicate to 1 entry)
	hfRepo := filepath.Join(homeDir, ".cache", "huggingface", "hub", "models--org--Llama3-GGUF")
	hfSnap := filepath.Join(hfRepo, "snapshots", "rev1")
	if err := os.MkdirAll(hfSnap, 0o755); err != nil {
		t.Fatalf("MkdirAll hfSnap: %v", err)
	}
	hfTarget := filepath.Join(hfSnap, "Llama3-8B-IQ4_XS.gguf")
	if err := os.WriteFile(hfTarget, []byte("hf-gguf"), 0o600); err != nil {
		t.Fatalf("write hfTarget: %v", err)
	}
	_ = os.Symlink(hfTarget, filepath.Join(customDir, "Llama3-symlink.gguf"))

	// Mock OpenAI-compatible (llama-server) and Ollama endpoints
	openAISrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			fmt.Fprint(w, `{"data":[{"id":"local-llama-server-model"}]}`)
			return
		}
		http.NotFound(w, r)
	}))
	defer openAISrv.Close()

	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			fmt.Fprint(w, `{"models":[{"name":"deepseek-r1:8b","size":4900000000}]}`)
			return
		}
		http.NotFound(w, r)
	}))
	defer ollamaSrv.Close()

	res := DiscoverLocalModels(context.Background(), DiscoverOptions{
		CustomDir: customDir,
		OpenAIURL: openAISrv.URL,
		OllamaURL: ollamaSrv.URL,
	})

	// Expect: modelA, modelB, Llama3 (deduplicated once), local-llama-server-model, deepseek-r1:8b (mmproj excluded)
	byName := make(map[string]LocalModel)
	for _, m := range res.Models {
		byName[m.Name] = m
	}

	if _, ok := byName["mmproj-model-f16"]; ok {
		t.Errorf("expected auxiliary mmproj file to be excluded")
	}
	if m, ok := byName["DeepSeek-V4-Q4_K_M"]; !ok || m.Source != SourceGGUFLocal || m.Quantization != "Q4_K_M" {
		t.Errorf("missing or invalid DeepSeek model: %+v", m)
	}
	if m, ok := byName["Qwen2.5-7B-Q8_0"]; !ok || m.Source != SourceGGUFLocal || m.Quantization != "Q8_0" {
		t.Errorf("missing or invalid Qwen model: %+v", m)
	}
	if m, ok := byName["local-llama-server-model"]; !ok || m.Source != SourceLlamaServer || !m.Running || m.EndpointURL != openAISrv.URL {
		t.Errorf("missing or invalid running llama-server model: %+v", m)
	}
	if m, ok := byName["deepseek-r1:8b"]; !ok || m.Source != SourceOllama || !m.Running || m.EndpointURL != ollamaSrv.URL {
		t.Errorf("missing or invalid Ollama model: %+v", m)
	}

	// Verify symlink deduplication (5 total models, not 6)
	if len(res.Models) != 5 {
		t.Errorf("expected 5 deduplicated models, got %d: %+v", len(res.Models), res.Models)
	}
}

func TestResolveDirectModelPath(t *testing.T) {
	tmp := t.TempDir()
	homeDir := filepath.Join(tmp, "home")
	dlDir := filepath.Join(homeDir, "Downloads", "CustomFolder")
	if err := os.MkdirAll(dlDir, 0o755); err != nil {
		t.Fatalf("MkdirAll dlDir: %v", err)
	}
	t.Setenv("HOME", homeDir)

	ggufPath := filepath.Join(dlDir, "Direct-Qwen3.5-4B-Q4_K_M.gguf")
	mmprojPath := filepath.Join(dlDir, "mmproj-vision-f16.gguf")
	txtPath := filepath.Join(dlDir, "notes.txt")
	if err := os.WriteFile(ggufPath, []byte("gguf-payload"), 0o600); err != nil {
		t.Fatalf("write ggufPath: %v", err)
	}
	if err := os.WriteFile(mmprojPath, []byte("mmproj"), 0o600); err != nil {
		t.Fatalf("write mmprojPath: %v", err)
	}
	if err := os.WriteFile(txtPath, []byte("not-a-model"), 0o600); err != nil {
		t.Fatalf("write txtPath: %v", err)
	}

	// 1. Direct ~/... file path
	mFile, err := ResolveDirectModelPath("~/Downloads/CustomFolder/Direct-Qwen3.5-4B-Q4_K_M.gguf")
	if err != nil {
		t.Fatalf("ResolveDirectModelPath file failed: %v", err)
	}
	if mFile.Path != ggufPath || mFile.Name != "Direct-Qwen3.5-4B-Q4_K_M" || mFile.Quantization != "Q4_K_M" || mFile.SourceLabel != "Direct Path" {
		t.Fatalf("unexpected resolved file model: %+v", mFile)
	}

	// 2. Direct directory path resolves non-mmproj .gguf inside it
	mDir, err := ResolveDirectModelPath(dlDir)
	if err != nil {
		t.Fatalf("ResolveDirectModelPath directory failed: %v", err)
	}
	if mDir.Path != ggufPath || mDir.ID != mFile.ID {
		t.Fatalf("expected directory resolution to match %s, got %+v", ggufPath, mDir)
	}

	// 3. Invalid / missing / non-gguf paths return errors
	for _, bad := range []string{"", filepath.Join(tmp, "missing.gguf"), txtPath, t.TempDir()} {
		if _, err := ResolveDirectModelPath(bad); err == nil {
			t.Errorf("expected error for invalid direct path %q, got nil", bad)
		}
	}
}

