package store

import (
	"testing"
	"time"

	"gpumode/internal/profiler"
)

func TestStoreSaveAndReload(t *testing.T) {
	tmp := t.TempDir()
	s, err := NewStore(tmp)
	if err != nil {
		t.Fatalf("NewStore failed: %v", err)
	}

	rep := profiler.CausalChainReport{
		ID:         "run_test_123",
		Timestamp:  time.Now(),
		OpCategory: "rmsnorm_residual",
		BatchSize:  128,
		HiddenDim:  4096,
	}
	if err := s.Save(rep); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	latest, ok := s.Latest()
	if !ok || latest.ID != "run_test_123" {
		t.Fatalf("expected latest run_test_123, got %+v (ok=%v)", latest, ok)
	}

	// Re-open store from disk
	s2, err := NewStore(tmp)
	if err != nil {
		t.Fatalf("NewStore reload failed: %v", err)
	}
	got, ok := s2.Get("run_test_123")
	if !ok || got.OpCategory != "rmsnorm_residual" {
		t.Fatalf("expected persisted report on reload, got %+v", got)
	}
}
