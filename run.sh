#!/usr/bin/env zsh
set -e

# Always run from the project root directory
cd "${0:A:h}"

ADDR="${GPUMODE_ADDR:-127.0.0.1:8080}"
MODE="${1:-dashboard}"

echo "================================================================================"
echo " gpuMode — Building & Launching GPU Resource Intelligence Platform"
echo "================================================================================"

# Build the Go binary if missing or if any Go source files are newer
go build -o ./gpumode ./cmd/gpumode

case "$MODE" in
  optimize|analyze)
    shift || true
    ./gpumode optimize "$@"
    ;;
  monitor)
    ./gpumode monitor
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
