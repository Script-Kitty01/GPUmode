package ebpftrace

import (
	"bufio"
	"context"
	"debug/elf"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// DiscoverCUDALibraries locates the host's active libcuda.so.1 (CUDA Driver API)
// and libcudart.so (CUDA Runtime API), preferring libraries mapped into target PIDs.
func DiscoverCUDALibraries(pids ...int) (libCudaPath string, libCudartPath string) {
	// 1. Inspect /proc/<pid>/maps for any supplied target PIDs first.
	for _, pid := range pids {
		if pid <= 0 {
			continue
		}
		mappedCuda, mappedCudart := discoverFromProcMaps(pid)
		if libCudaPath == "" && mappedCuda != "" {
			libCudaPath = mappedCuda
		}
		if libCudartPath == "" && mappedCudart != "" {
			libCudartPath = mappedCudart
		}
		if libCudaPath != "" && libCudartPath != "" {
			return libCudaPath, libCudartPath
		}
	}

	// 2. Check explicit environment overrides.
	if libCudaPath == "" {
		if env := strings.TrimSpace(os.Getenv("GPUMODE_LIBCUDA")); env != "" {
			if resolved := resolveValidSharedObject(env); resolved != "" {
				libCudaPath = resolved
			}
		}
	}
	if libCudartPath == "" {
		if env := strings.TrimSpace(os.Getenv("GPUMODE_LIBCUDART")); env != "" {
			if resolved := resolveValidSharedObject(env); resolved != "" {
				libCudartPath = resolved
			}
		}
	}

	// 3. Probe standard Linux / CUDA installation paths.
	if libCudaPath == "" {
		cudaCandidates := []string{
			"/usr/lib/libcuda.so.1",
			"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
			"/usr/lib64/libcuda.so.1",
			"/lib/x86_64-linux-gnu/libcuda.so.1",
			"/lib64/libcuda.so.1",
			"/usr/local/nvidia/lib64/libcuda.so.1",
			"/usr/local/nvidia/lib/libcuda.so.1",
			"/run/opengl-driver/lib/libcuda.so.1",
			"/usr/lib/wsl/lib/libcuda.so.1",
			"/usr/lib/libcuda.so",
		}
		for _, p := range cudaCandidates {
			if resolved := resolveValidSharedObject(p); resolved != "" {
				libCudaPath = resolved
				break
			}
		}
	}

	if libCudartPath == "" {
		cudartCandidates := []string{
			"/opt/cuda/lib64/libcudart.so",
			"/opt/cuda/targets/x86_64-linux/lib/libcudart.so",
			"/usr/local/cuda/lib64/libcudart.so",
			"/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so",
			"/usr/lib/x86_64-linux-gnu/libcudart.so",
			"/usr/lib64/libcudart.so",
			"/usr/lib/libcudart.so",
		}
		for _, pattern := range []string{
			"/opt/cuda*/lib64/libcudart.so*",
			"/usr/local/cuda*/lib64/libcudart.so*",
			"/usr/lib/x86_64-linux-gnu/libcudart.so*",
			"/usr/lib64/libcudart.so*",
			"/usr/lib/libcudart.so*",
		} {
			if matches, err := filepath.Glob(pattern); err == nil {
				sort.Strings(matches)
				cudartCandidates = append(cudartCandidates, matches...)
			}
		}
		for _, p := range cudartCandidates {
			if !isLibCudartFilename(filepath.Base(p)) {
				continue
			}
			if resolved := resolveValidSharedObject(p); resolved != "" {
				libCudartPath = resolved
				break
			}
		}
	}

	// 4. Fall back to ldconfig -p if either library is still missing.
	if libCudaPath == "" || libCudartPath == "" {
		ldCuda, ldCudart := discoverFromLdconfig()
		if libCudaPath == "" && ldCuda != "" {
			libCudaPath = ldCuda
		}
		if libCudartPath == "" && ldCudart != "" {
			libCudartPath = ldCudart
		}
	}

	return libCudaPath, libCudartPath
}

func discoverFromProcMaps(pid int) (libCudaPath string, libCudartPath string) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return "", ""
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 512*1024)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		mappedPath := fields[5]
		if !strings.HasPrefix(mappedPath, "/") || strings.HasSuffix(mappedPath, "(deleted)") {
			continue
		}
		base := filepath.Base(mappedPath)
		if libCudaPath == "" && isLibCudaFilename(base) {
			if resolved := resolveValidSharedObject(mappedPath); resolved != "" {
				libCudaPath = resolved
			}
		} else if libCudartPath == "" && isLibCudartFilename(base) {
			if resolved := resolveValidSharedObject(mappedPath); resolved != "" {
				libCudartPath = resolved
			}
		}
		if libCudaPath != "" && libCudartPath != "" {
			break
		}
	}
	if err := scanner.Err(); err != nil && libCudaPath == "" && libCudartPath == "" {
		return "", ""
	}
	return libCudaPath, libCudartPath
}

func discoverFromLdconfig() (libCudaPath string, libCudartPath string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	out, err := exec.CommandContext(ctx, "ldconfig", "-p").Output()
	if err != nil {
		return "", ""
	}

	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		idx := strings.Index(line, "=>")
		if idx < 0 {
			continue
		}
		left := strings.TrimSpace(line[:idx])
		right := strings.TrimSpace(line[idx+2:])
		if right == "" {
			continue
		}
		// Prefer x86-64 libraries if multiarch entries are present.
		if strings.Contains(left, "(") && !strings.Contains(left, "x86-64") && !strings.Contains(left, "64bit") {
			continue
		}
		base := filepath.Base(right)
		if libCudaPath == "" && isLibCudaFilename(base) {
			if resolved := resolveValidSharedObject(right); resolved != "" {
				libCudaPath = resolved
			}
		} else if libCudartPath == "" && isLibCudartFilename(base) {
			if resolved := resolveValidSharedObject(right); resolved != "" {
				libCudartPath = resolved
			}
		}
		if libCudaPath != "" && libCudartPath != "" {
			break
		}
	}
	return libCudaPath, libCudartPath
}

func isLibCudaFilename(base string) bool {
	return base == "libcuda.so" || strings.HasPrefix(base, "libcuda.so.")
}

func isLibCudartFilename(base string) bool {
	if strings.HasSuffix(base, ".a") {
		return false
	}
	return base == "libcudart.so" || strings.HasPrefix(base, "libcudart.so.")
}

func resolveValidSharedObject(path string) string {
	if path == "" {
		return ""
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		resolved = path
	}
	info, err := os.Stat(resolved)
	if err != nil || info.IsDir() || info.Size() == 0 {
		return ""
	}
	return resolved
}

// inspectELFSymbols checks which of the requested symbol names exist in the ELF file.
func inspectELFSymbols(elfPath string, wanted ...string) (map[string]bool, error) {
	f, err := elf.Open(elfPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	wantSet := make(map[string]struct{}, len(wanted))
	for _, w := range wanted {
		wantSet[w] = struct{}{}
	}

	found := make(map[string]bool, len(wanted))
	checkList := func(syms []elf.Symbol) {
		for _, sym := range syms {
			if sym.Value == 0 || elf.ST_TYPE(sym.Info) != elf.STT_FUNC {
				continue
			}
			name := sym.Name
			if idx := strings.IndexByte(name, '@'); idx > 0 {
				name = name[:idx]
			}
			if _, ok := wantSet[name]; ok {
				found[name] = true
			}
		}
	}

	if dynSyms, err := f.DynamicSymbols(); err == nil {
		checkList(dynSyms)
	}
	if len(found) < len(wantSet) {
		if symTab, err := f.Symbols(); err == nil {
			checkList(symTab)
		}
	}
	return found, nil
}

// FindPIDsByTCPPort inspects /proc/net/tcp and /proc/net/tcp6 for TCP_LISTEN (0A)
// sockets bound to port, then resolves owning process IDs via /proc/<pid>/fd/*.
func FindPIDsByTCPPort(port int) []int {
	if port <= 0 || port > 65535 {
		return nil
	}

	inodes := make(map[string]struct{})
	for _, netFile := range []string{"/proc/net/tcp", "/proc/net/tcp6"} {
		collectListeningInodesFromFile(netFile, port, inodes)
	}
	if len(inodes) == 0 {
		return nil
	}

	procEntries, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}

	pidSet := make(map[int]struct{})
	for _, entry := range procEntries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil || pid <= 0 {
			continue
		}
		if processOwnsAnySocketInode(pid, inodes) {
			pidSet[pid] = struct{}{}
		}
	}

	if len(pidSet) == 0 {
		return nil
	}

	pids := make([]int, 0, len(pidSet))
	for pid := range pidSet {
		pids = append(pids, pid)
	}
	sort.Ints(pids)
	return pids
}

func collectListeningInodesFromFile(procNetPath string, port int, out map[string]struct{}) {
	f, err := os.Open(procNetPath)
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	firstLine := true
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if firstLine {
			firstLine = false
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 10 {
			continue
		}
		// st == 0A corresponds to TCP_LISTEN in Linux net/tcp_states.h.
		if !strings.EqualFold(fields[3], "0A") {
			continue
		}
		localAddr := fields[1]
		colon := strings.LastIndexByte(localAddr, ':')
		if colon < 0 || colon+1 >= len(localAddr) {
			continue
		}
		parsedPort, err := strconv.ParseUint(localAddr[colon+1:], 16, 16)
		if err != nil || int(parsedPort) != port {
			continue
		}
		inode := fields[9]
		if inode != "" && inode != "0" {
			out[inode] = struct{}{}
		}
	}
	if err := scanner.Err(); err != nil {
		return
	}
}

func processOwnsAnySocketInode(pid int, inodes map[string]struct{}) bool {
	fdDir := fmt.Sprintf("/proc/%d/fd", pid)
	fds, err := os.ReadDir(fdDir)
	if err != nil {
		return false
	}
	for _, fd := range fds {
		target, err := os.Readlink(filepath.Join(fdDir, fd.Name()))
		if err != nil {
			continue
		}
		if strings.HasPrefix(target, "socket:[") && strings.HasSuffix(target, "]") {
			inode := target[len("socket:[") : len(target)-1]
			if _, ok := inodes[inode]; ok {
				return true
			}
		} else if strings.HasPrefix(target, "[0000]:") {
			inode := target[len("[0000]:"):]
			if _, ok := inodes[inode]; ok {
				return true
			}
		}
	}
	return false
}

func readProcParentPID(pid uint32) uint32 {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0
	}
	raw := string(data)
	closeParen := strings.LastIndexByte(raw, ')')
	if closeParen < 0 || closeParen+2 >= len(raw) {
		return 0
	}
	rest := strings.Fields(raw[closeParen+1:])
	// rest[0] = state, rest[1] = ppid
	if len(rest) < 2 {
		return 0
	}
	ppid, err := strconv.ParseUint(rest[1], 10, 32)
	if err != nil {
		return 0
	}
	return uint32(ppid)
}

func readProcComm(pid uint32) string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/comm", pid))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}
