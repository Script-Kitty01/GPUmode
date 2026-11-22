//go:build linux && amd64

package ebpftrace

import (
	"testing"

	"github.com/cilium/ebpf"
)

func TestEmbeddedBPFCollectionSpecValid(t *testing.T) {
	spec, err := loadCudaTrace()
	if err != nil {
		t.Fatalf("loadCudaTrace() CollectionSpec parse failed: %v", err)
	}

	expectedMaps := map[string]struct {
		typ        ebpf.MapType
		maxEntries uint32
	}{
		"tgid_stats":       {typ: ebpf.LRUHash, maxEntries: 1024},
		"target_tgids":     {typ: ebpf.Hash, maxEntries: 256},
		"tid_sync_enter":   {typ: ebpf.LRUHash, maxEntries: 4096},
		"tid_offcpu_enter": {typ: ebpf.LRUHash, maxEntries: 4096},
	}

	for name, want := range expectedMaps {
		ms, ok := spec.Maps[name]
		if !ok {
			t.Errorf("missing expected BPF map %q in compiled ELF", name)
			continue
		}
		if ms.Type != want.typ {
			t.Errorf("map %q Type = %v, want %v", name, ms.Type, want.typ)
		}
		if ms.MaxEntries != want.maxEntries {
			t.Errorf("map %q MaxEntries = %d, want %d", name, ms.MaxEntries, want.maxEntries)
		}
	}

	expectedProgs := []string{
		"handle_cuda_launch",
		"handle_graph_launch",
		"handle_memcpy_h2d",
		"handle_memcpy_d2h",
		"handle_cudart_memcpy",
		"handle_mem_alloc",
		"handle_sync_enter",
		"handle_sync_exit",
		"handle_sched_switch",
	}
	for _, progName := range expectedProgs {
		ps, ok := spec.Programs[progName]
		if !ok {
			t.Errorf("missing expected BPF program %q in compiled ELF", progName)
			continue
		}
		if len(ps.Instructions) == 0 {
			t.Errorf("BPF program %q has 0 instructions", progName)
		}
	}
}
