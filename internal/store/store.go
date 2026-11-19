package store

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"gpumode/internal/profiler"
)

// Store manages thread-safe in-memory and disk-persisted CausalChainReport runs.
type Store struct {
	mu      sync.RWMutex
	dataDir string
	reports []profiler.CausalChainReport
}

func NewStore(dataDir string) (*Store, error) {
	if dataDir == "" {
		dataDir = ".gpumode_data"
	}
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, err
	}
	s := &Store{
		dataDir: dataDir,
	}
	_ = s.loadFromDisk()
	return s, nil
}

func (s *Store) Save(rep profiler.CausalChainReport) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	replaced := false
	for i := range s.reports {
		if s.reports[i].ID == rep.ID {
			s.reports[i] = rep
			replaced = true
			break
		}
	}
	if !replaced {
		s.reports = append([]profiler.CausalChainReport{rep}, s.reports...)
		if len(s.reports) > 50 {
			s.reports = s.reports[:50]
		}
	}

	filePath := filepath.Join(s.dataDir, rep.ID+".json")
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0644)
}

func (s *Store) Latest() (profiler.CausalChainReport, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.reports) == 0 {
		return profiler.CausalChainReport{}, false
	}
	return s.reports[0], true
}

func (s *Store) Get(id string) (profiler.CausalChainReport, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, r := range s.reports {
		if r.ID == id {
			return r, true
		}
	}
	return profiler.CausalChainReport{}, false
}

func (s *Store) List() []profiler.CausalChainReport {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]profiler.CausalChainReport, len(s.reports))
	copy(out, s.reports)
	return out
}

func (s *Store) loadFromDisk() error {
	entries, err := os.ReadDir(s.dataDir)
	if err != nil {
		return err
	}
	var loaded []profiler.CausalChainReport
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(s.dataDir, e.Name()))
		if err != nil {
			continue
		}
		var rep profiler.CausalChainReport
		if err := json.Unmarshal(raw, &rep); err == nil && rep.ID != "" {
			loaded = append(loaded, rep)
		}
	}
	sort.Slice(loaded, func(i, j int) bool {
		return loaded[i].Timestamp.After(loaded[j].Timestamp)
	})
	s.reports = loaded
	return nil
}
