//go:build !linux || !amd64

package ebpftrace

import (
	"fmt"
	"sync"
)

// Session is a fallback stub on non-linux/amd64 targets.
type Session struct {
	mu            sync.Mutex
	stopped       bool
	cachedSummary Summary
	mode          Mode
	targetPIDSet  map[uint32]struct{}
}

func StartSession(mode Mode, targetPIDs []int) (*Session, error) {
	normMode := ParseMode(string(mode))
	targetSet := make(map[uint32]struct{}, len(targetPIDs))
	for _, pid := range targetPIDs {
		if pid > 0 {
			targetSet[uint32(pid)] = struct{}{}
		}
	}

	if normMode == ModeOff {
		return &Session{
			mode:         ModeOff,
			targetPIDSet: targetSet,
		}, nil
	}

	const detail = "eBPF CUDA tracing is only supported on linux/amd64"
	if normMode == ModeOn {
		return nil, fmt.Errorf("ebpftrace: %s", detail)
	}
	return &Session{
		mode:         normMode,
		targetPIDSet: targetSet,
	}, nil
}

func ProbeAvailability(mode Mode) Summary {
	normMode := ParseMode(string(mode))
	sess, _ := StartSession(normMode, nil)
	if sess == nil {
		return Summary{
			Enabled:      true,
			Active:       false,
			Mode:         normMode,
			Provenance:   ProvenanceUnavailableUnsupported,
			StatusDetail: "eBPF CUDA tracing is only supported on linux/amd64",
		}
	}
	return sess.Stop()
}

func (s *Session) AddTargetPID(pid int) {
	if s == nil || pid <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return
	}
	if s.targetPIDSet == nil {
		s.targetPIDSet = make(map[uint32]struct{})
	}
	s.targetPIDSet[uint32(pid)] = struct{}{}
}

func (s *Session) Stop() Summary {
	if s == nil {
		return Summary{
			Enabled:    false,
			Active:     false,
			Mode:       ModeOff,
			Provenance: ProvenanceDisabled,
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return s.cachedSummary
	}
	s.stopped = true

	targetPIDs := sortedUint32Keys(s.targetPIDSet)
	if s.mode == ModeOff {
		s.cachedSummary = Summary{
			Enabled:      false,
			Active:       false,
			Mode:         ModeOff,
			Provenance:   ProvenanceDisabled,
			StatusDetail: "eBPF tracing disabled",
			TargetPIDs:   targetPIDs,
		}
		return s.cachedSummary
	}

	s.cachedSummary = Summary{
		Enabled:      true,
		Active:       false,
		Mode:         s.mode,
		Provenance:   ProvenanceUnavailableUnsupported,
		StatusDetail: "eBPF CUDA tracing is only supported on linux/amd64",
		TargetPIDs:   targetPIDs,
	}
	return s.cachedSummary
}
