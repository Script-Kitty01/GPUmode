#!/usr/bin/env zsh
set -e

# Always run from the project root directory
cd "${0:A:h}"

ADDR="${GPUMODE_ADDR:-127.0.0.1:8080}"
MODE="${1:-dashboard}"

echo "================================================================================"
echo " gpuMode — Building & Launching GPU Resource Intelligence Platform"
echo "================================================================================"

if [[ "$MODE" == "bpf-gen" ]]; then
  echo "[gpuMode] Regenerating eBPF C bytecode & Go bindings (internal/ebpftrace)..."
  go generate ./internal/ebpftrace/...
  go build -o ./gpumode ./cmd/gpumode
  echo "[gpuMode] eBPF bytecode regenerated and ./gpumode rebuilt."
  exit 0
fi

# Build ./gpumode only if missing or if Go/module/BPF artifacts are newer,
# preserving Linux file capabilities (setcap) across normal ./run.sh invocations.
needs_build=0
if [[ ! -x ./gpumode ]]; then
  needs_build=1
elif [[ go.mod -nt ./gpumode || go.sum -nt ./gpumode ]]; then
  needs_build=1
else
  newer_files=( cmd/**/*.go(N) internal/**/*.go(N) internal/**/*.o(N) frontend/**/*(N) )
  for f in "${newer_files[@]}"; do
    if [[ "$f" -nt ./gpumode ]]; then
      needs_build=1
      break
    fi
  done
fi

if (( needs_build )); then
  go build -o ./gpumode ./cmd/gpumode
fi

ensure_ebpf_caps() {
  local required="${1:-0}"
  if [[ "${GPUMODE_EBPF:l}" == "off" ]]; then
    return 0
  fi
  if (( EUID == 0 )); then
    export GPUMODE_EBPF="${GPUMODE_EBPF:-on}"
    return 0
  fi
  if ! command -v setcap >/dev/null 2>&1 || ! command -v getcap >/dev/null 2>&1; then
    return 0
  fi
  local cur_caps
  cur_caps="$(getcap ./gpumode 2>/dev/null || true)"
  if [[ "$cur_caps" == *"cap_bpf"* && "$cur_caps" == *"cap_perfmon"* && "$cur_caps" == *"cap_sys_admin"* ]]; then
    export GPUMODE_EBPF="${GPUMODE_EBPF:-on}"
    return 0
  fi
  echo "[gpuMode] Granting rootless eBPF capabilities (cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep) to ./gpumode..."
  local rc=0
  if [[ -n "${SUDO_ASKPASS:-}" ]]; then
    sudo -A setcap cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep ./gpumode || rc=$?
  else
    sudo setcap cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep ./gpumode || rc=$?
  fi
  if (( rc != 0 )); then
    if (( required )); then
      return $rc
    fi
    echo "[gpuMode] Warning: could not grant eBPF capabilities; continuing with -ebpf=auto fallback."
    return 0
  fi
  export GPUMODE_EBPF="${GPUMODE_EBPF:-on}"
  return 0
}

case "$MODE" in
  setcap)
    ensure_ebpf_caps 1
    echo "[gpuMode] Verifying rootless NVML + eBPF CUDA attachment..."
    ./gpumode ebpf
    ;;
  ebpf|probe)
    ensure_ebpf_caps
    shift || true
    ./gpumode ebpf "$@"
    ;;
  optimize|analyze)
    ensure_ebpf_caps
    shift || true
    ./gpumode optimize "$@"
    ;;
  deepseek)
    ensure_ebpf_caps
    shift || true
    echo "[gpuMode] Querying local DeepSeek-V4-9B GGUF model & running live GPU verification..."
    ./gpumode deepseek -proposer deepseek "$@"
    ;;
  workload)
    ensure_ebpf_caps
    shift || true
    echo "[gpuMode] Running real streaming inference workload benchmark against local DeepSeek/Ollama server..."
    ./gpumode workload "$@"
    ;;
  monitor)
    shift || true
    ./gpumode monitor "$@"
    ;;
  schedule)
    shift || true
    ./gpumode schedule "$@"
    ;;
  check)
    ensure_ebpf_caps
    shift || true
    ./gpumode check "$@"
    ;;
  test)
    go test -v ./...
    ;;
  dashboard|serve|*)
    ensure_ebpf_caps
    echo "[gpuMode] Running initial PyTorch/CUDA trace & starting Web Dashboard at http://${ADDR} ..."
    PY_BIN="python3"
    if [[ -x ./.venv/bin/python3 ]]; then
      PY_BIN="./.venv/bin/python3"
    fi
    "$PY_BIN" ./scripts/profile_pytorch.py --verify-extension
    echo ""
    if ss -ltn 2>/dev/null | grep -q ":${ADDR##*:} "; then
      echo "[gpuMode] Dashboard is already running at http://${ADDR}"
      echo "[gpuMode] Running a fresh live GPU optimization pass..."
      ./gpumode optimize -op rmsnorm_residual -batch 256 -hidden 4096 -iters 40 -qps 350
    else
      echo "[gpuMode] Starting live Web Dashboard at http://${ADDR} (Ctrl+C to stop)"
      exec ./gpumode dashboard -addr "$ADDR"
    fi
    ;;
esac
