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
  newer_files=( cmd/**/*.go(N) internal/**/*.go(N) internal/**/*.o(N) )
  for f in "${newer_files[@]}"; do
    if [[ "$f" -nt ./gpumode ]]; then
      needs_build=1
      break
    fi
  done
fi

if (( needs_build )); then
  had_caps=0
  if command -v getcap >/dev/null 2>&1 && getcap ./gpumode 2>/dev/null | grep -q "cap_bpf"; then
    had_caps=1
  fi
  go build -o ./gpumode ./cmd/gpumode
  if (( had_caps )) && [[ "$MODE" != "setcap" ]]; then
    echo "[gpuMode] Note: rebuilt ./gpumode from updated sources; re-run './run.sh setcap' if you want rootless eBPF."
  fi
fi

case "$MODE" in
  setcap)
    echo "[gpuMode] Granting rootless eBPF & perf capabilities (cap_bpf,cap_perfmon,cap_sys_resource+ep) to ./gpumode..."
    sudo setcap cap_bpf,cap_perfmon,cap_sys_resource+ep ./gpumode
    echo "[gpuMode] Verifying rootless NVML + eBPF CUDA attachment..."
    ./gpumode ebpf
    ;;
  ebpf|probe)
    shift || true
    ./gpumode ebpf "$@"
    ;;
  optimize|analyze)
    shift || true
    ./gpumode optimize "$@"
    ;;
  deepseek)
    shift || true
    echo "[gpuMode] Querying local DeepSeek-V4-9B GGUF model & running live GPU verification..."
    ./gpumode deepseek -proposer deepseek "$@"
    ;;
  workload)
    shift || true
    echo "[gpuMode] Running real streaming inference workload benchmark against local DeepSeek/Ollama server..."
    ./gpumode workload "$@"
    ;;
  monitor)
    shift || true
    ./gpumode monitor "$@"
    ;;
  test)
    go test -v ./...
    ;;
  dashboard|serve|*)
    echo "[gpuMode] Running initial PyTorch/CUDA trace & starting Web Dashboard at http://${ADDR} ..."
    python3 ./scripts/profile_pytorch.py
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
