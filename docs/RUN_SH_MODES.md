# `run.sh` Startup Modes & Local Model Guide

`./run.sh` is the single-entrypoint build, capability, and launch wrapper for **gpuMode**. It always executes from the repository root (`cd "${0:A:h}"`), manages incremental Go builds, preserves rootless Linux eBPF capabilities (`setcap`), and dispatches all daemon and CLI modes.

For the project overview, see [README.md](../README.md). For step-by-step setup and Web Dashboard usage, see [STARTUP_GUIDE.md](./STARTUP_GUIDE.md).

---

## How `./run.sh` Works Before Launching a Mode

1. **Incremental Build Preservation:**
   - `./run.sh` checks whether `./gpumode` exists and compares its timestamp against `go.mod`, `go.sum`, `cmd/**/*.go`, `internal/**/*.go`, and `internal/**/*.o`.
   - It rebuilds `./gpumode` **only** when source or BPF object files have changed, preserving Linux file capabilities (`setcap`) across normal startups.
2. **Automatic Rootless eBPF Capability Check (`ensure_ebpf_caps`):**
   - Unless `GPUMODE_EBPF=off` is set, `./run.sh` checks `getcap ./gpumode` for `cap_bpf`, `cap_perfmon`, `cap_sys_resource`, and `cap_sys_admin`.
   - If capabilities are missing, it attempts `sudo setcap cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep ./gpumode` (using `sudo -A` when `SUDO_ASKPASS` is set). If unprivileged, non-`setcap` modes continue gracefully with `-ebpf=auto`.
3. **Port-Aware Dashboard Launch:**
   - Default address is `127.0.0.1:8080` (override via `GPUMODE_ADDR=host:port` or `-addr host:port`).
   - In `dashboard` mode, if the target port is already listening, `./run.sh` automatically runs a live GPU optimization pass instead of failing with an `address already in use` error.

---

## Quick Reference: All Startup Modes in `run.sh`

| Mode | Alias | Purpose | Typical Command |
| :--- | :--- | :--- | :--- |
| **`dashboard`** *(default)* | `serve` | Run initial trace and start the live HTTP/SSE Web Dashboard (`http://127.0.0.1:8080`) | `./run.sh` or `./run.sh dashboard` |
| **`optimize`** | `analyze` | Compile, sanitize, verify, and benchmark CUDA kernels in the terminal | `./run.sh optimize -op rmsnorm_residual` |
| **`deepseek`** | — | Use a local GGUF LLM (`llama-cli`) to propose CUDA kernel configs + verify on GPU | `./run.sh deepseek -deepseek-model /path/to/model.gguf` |
| **`workload`** | — | Benchmark end-to-end streaming LLM inference (TTFT, TPOT, tok/s, VRAM, Watts, `$/1M tok`, eBPF) | `./run.sh workload -deepseek-model /path/to/model.gguf` |
| **`schedule`** | — | Run VRAM-aware GPU pool placement, priority admission control (`HTTP 429`), and strategy simulator | `./run.sh schedule -deepseek-model /path/to/model.gguf` |
| **`check`** | — | CI/CD hardware regression gate (verifies speedup, `0` PTXAS spills, `compute-sanitizer`, and `1027` parity) | `./run.sh check -op rope_qkv -min-speedup 1.15` |
| **`monitor`** | — | Print an instant GPU hardware, clock, VRAM, and live power snapshot | `./run.sh monitor` |
| **`ebpf`** | `probe` | Inspect NVML backend, `libcuda.so.1` / `libcudart.so` paths, active GPU PIDs, and eBPF uprobes | `./run.sh ebpf` |
| **`setcap`** | — | Grant rootless Linux eBPF capabilities to `./gpumode` and verify probe attachment | `./run.sh setcap` |
| **`bpf-gen`** | — | Regenerate eBPF C bytecode (`internal/ebpftrace`) via `go generate` and rebuild `./gpumode` | `./run.sh bpf-gen` |
| **`test`** | — | Run the complete Go unit and live CUDA hardware test suite | `./run.sh test` |

---

## Detailed Mode Reference

### 1. `dashboard` / `serve` (Default Mode)

Starts the real-time Go daemon, `10 ms` NVML telemetry monitor, Host ↔ CUDA eBPF tracer, REST API, and interactive Web Dashboard.

```zsh
# Default launch on http://127.0.0.1:8080
./run.sh

# Explicit mode with custom bind address
GPUMODE_ADDR=127.0.0.1:9090 ./run.sh dashboard

# Select GPU #1 on a multi-GPU machine (when calling ./gpumode directly or passing flags)
./gpumode dashboard -addr 127.0.0.1:8080 -gpu 1
```

**What happens on startup:**
1. Runs `python3 ./scripts/profile_pytorch.py` (uses `torch.profiler` if PyTorch with CUDA is installed, or outputs an analytical operator hotspot trace otherwise).
2. Pre-warms an initial hardware-verified CUDA causal report if `.gpumode_data` is empty or predates eBPF attachment.
3. Serves the live dashboard and SSE telemetry stream at `http://127.0.0.1:8080`.

---

### 2. `optimize` / `analyze`

Runs the closed-loop CUDA optimization pipeline directly in the terminal (`Roofline → Autotuner → nvcc -Xptxas -v → compute-sanitizer → Numerical Parity → Live NVML Energy → Pareto Right-Sizing`).

```zsh
# Default Bayesian UCB optimization on fused RMSNorm + Residual
./run.sh optimize -op rmsnorm_residual -batch 256 -hidden 4096 -iters 40 -qps 350

# Optimize other supported LLM operators
./run.sh optimize -op swiglu -strategy BAYESIAN_UCB
./run.sh optimize -op softmax -strategy GRID_SEARCH
./run.sh optimize -op layernorm -batch 128 -hidden 4096 -iters 50
./run.sh optimize -op rmsnorm -json
```

* **Supported `-op` values:** `rmsnorm_residual`, `rmsnorm`, `layernorm`, `softmax`, `swiglu`, `rope_qkv`, `qgemv_decode`, `gqa_attention`
* **Supported `-strategy` values:** `BAYESIAN_UCB` *(default)*, `GRID_SEARCH`
* Pass `-json` to emit the full `CausalChainReport` as formatted JSON.

---

### 3. `deepseek` (Local GGUF Model Kernel Proposer)

Queries a local `.gguf` language model via `llama-cli` (`--single-turn`) with live GPU architecture specs (SM count, L2 cache, shared memory, ridge point) to propose CUDA block/vector/fusion parameters, frees VRAM, and then compiles and verifies the proposed kernel with `nvcc` and `compute-sanitizer`.

```zsh
# Auto-discovers a local GGUF model (DeepSeek / Qwen / LLM in ~/AI/Models, ~/models, ./models, or HF cache)
./run.sh deepseek -op rmsnorm_residual -batch 256 -hidden 4096 -iters 40

# Explicitly specify any local .gguf model path
./run.sh deepseek \
  -deepseek-model ~/AI/Models/Qwen3.5-4B-Q4_K_M.gguf \
  -op swiglu \
  -iters 40
```

---

### 4. `workload` (Streaming Local LLM Inference Benchmark)

Benchmarks real end-to-end OpenAI-compatible streaming inference (`TTFT`, `TPOT`, `tokens/sec`, peak VRAM, `10 ms` NVML board & active Watts, `Joules / 1K tokens`, and eBPF `launches/tok` + PCIe `H2D/D2H` + sync stalls).

```zsh
# Auto-spawns temporary llama-server using auto-discovered local GGUF model
./run.sh workload -llm-reqs 4 -llm-tokens 48

# Benchmark a specific local .gguf file directly
./run.sh workload \
  -deepseek-model ~/AI/Models/Qwen3.5-4B-Q4_K_M.gguf \
  -llm-reqs 4 \
  -llm-tokens 64

# Benchmark an already-running Ollama server (disable auto-server spawn)
./run.sh workload \
  -auto-server=false \
  -llm-url http://127.0.0.1:11434 \
  -llm-model qwen2.5:7b \
  -llm-reqs 4 \
  -llm-tokens 48

# Benchmark an already-running external llama-server / vLLM endpoint and output JSON
./run.sh workload \
  -auto-server=false \
  -llm-url http://127.0.0.1:8081 \
  -json
```

> **Tip:** To run **Without Project (Baseline)** vs. **With Project (gpuMode Tuned)** side-by-side on any `.gguf` path without typing flags, start `./run.sh` and use **Section 2 (Local AI Model Runner)** in the Web Dashboard (`http://127.0.0.1:8080`).

---

### 5. `schedule` (Production Fleet Cost, Placement & Admission Control Optimizer)

Evaluates how a target model fits across your local GPU and cloud GPU tiers (`Local GPU`, `L4 24GB`, `A10G 24GB`, `L40S 48GB` — On-Demand & Spot), calculates `$ / 1M tokens` and `$ / 1K requests`, configures priority admission control (`CRITICAL_REALTIME`, `STANDARD_INTERACTIVE`, `BATCH_SPOT_INTERRUPTIBLE`), and simulates 3 fleet scheduling strategies (`ROUND_ROBIN`, `LEAST_LOADED_VRAM_AWARE`, `COST_SLO_WEIGHTED_PARETO`).

```zsh
# Evaluate placement and scheduling for a specific GGUF model
./run.sh schedule -deepseek-model ~/AI/Models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf -llm-model Qwen3.5-4B-Q4_K_M

# Emit full FleetScheduleReport as JSON
./run.sh schedule -json
```

---

### 6. `check` (CI/CD Hardware Regression Gate)

Compiles and benchmarks the target operation on the live GPU and exits non-zero if speedup falls below `-min-speedup`, `nvcc -Xptxas -v` introduces register spills (`SpillStoresBytes > 0`), `compute-sanitizer` reports memory errors, or `hidden_dim = 1027` numerical parity fails.

```zsh
./run.sh check -op rope_qkv -batch 64 -hidden 2048 -iters 20 -min-speedup 1.15
```

---

### 7. `monitor`

Polls the GPU once and prints a single-line hardware and live telemetry summary (no root or eBPF capabilities required).

```zsh
./run.sh monitor

# Force nvidia-smi fallback instead of direct NVML
./run.sh monitor -telemetry smi
```

---

### 6. `ebpf` / `probe`

Runs a diagnostic check on direct NVML telemetry, active CUDA compute PIDs, discovered `libcuda.so.1` and `libcudart.so` shared libraries, and attached Linux eBPF uprobes/tracepoints.

```zsh
# Human-readable diagnostic table
./run.sh ebpf

# Full JSON diagnostic payload
./run.sh ebpf -json
```

---

### 7. `setcap`

Explicitly grants rootless Linux capabilities (`cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep`) to `./gpumode` via `sudo setcap` and immediately runs `./gpumode ebpf` to verify live attachment.

```zsh
./run.sh setcap
```

*Note: Recompiling `./gpumode` creates a new binary file and clears Linux file capabilities; running `./run.sh` or `./run.sh setcap` restores them.*

---

### 8. `bpf-gen`

Recompiles the C eBPF program (`internal/ebpftrace/bpf/cuda_trace.bpf.c`) into Go bindings via `bpf2go` (`go generate ./internal/ebpftrace/...`) and rebuilds `./gpumode`. Only needed when editing BPF C code.

```zsh
./run.sh bpf-gen
```

---

### 9. `test`

Runs the full Go unit test and live GPU hardware verification suite across all packages.

```zsh
./run.sh test
```

---

## Startup Models & Model Resolution in `run.sh`

Modes that interact with local AI models (`dashboard`, `deepseek`, and `workload`) support four ways to supply models:

### 1. Direct Path in the Web Dashboard (`dashboard` mode)
In Section 2 of `http://127.0.0.1:8080`, paste any `.gguf` file path (e.g. `~/Downloads/MyModel.gguf` or `/mnt/models/Qwen3.5-4B-Q4_K_M.gguf`) or model folder directly into **Direct Model Path** and click **Run Without Project (Baseline)** or **Run With Project (gpuMode)** — no need to move files into `~/models`.

### 2. Automatic Multi-Source Model Discovery
Without extra flags, `gpumode` automatically scans:
1. `GPUMODE_MODEL_PATH` (single `.gguf` file) and `GPUMODE_MODEL_DIR` (custom directory)
2. `~/AI/Models` (recursive `.gguf` scan, skipping `mmproj` helper files)
3. `~/models` and `./models`
4. `~/.cache/huggingface/hub` (Hugging Face Hub cache, deduplicating symlinks)
5. Live OpenAI-compatible server at `http://127.0.0.1:8081/v1/models` (`llama-server` / vLLM)
6. Live Ollama daemon at `http://127.0.0.1:11434/api/tags`

When auto-selecting a single default CLI model (`deepseek` / `workload`), filenames containing `deepseek`, `qwen`, or `llm` are prioritized first, followed by deterministic alphabetical order.

### 3. CLI Model Flags (Passed Through `./run.sh <mode> [flags]`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-deepseek-model <path>` | Auto-discovered `.gguf` | Local `.gguf` model path for `deepseek` proposer or `workload` auto-spawned `llama-server` |
| `-proposer <engine>` | `deterministic` (`deepseek` in `deepseek` mode) | Kernel proposer engine (`deterministic` or `deepseek`) |
| `-llm-url <url>` | `http://127.0.0.1:8081` | OpenAI-compatible or Ollama server URL for `workload` |
| `-llm-model <name>` | Auto-discovered | Explicit served model ID/tag (e.g. `qwen2.5:7b` for Ollama) |
| `-llm-reqs <n>` | `4` | Number of streaming inference requests in `workload` |
| `-llm-concurrency <n>` | `1` | Concurrent client workers (`1..32`) for multi-client `workload` load testing |
| `-llm-tokens <n>` | `48` | Max generated tokens per request in `workload` |
| `-auto-server` | `true` | Auto-launch temporary `llama-server` using `-deepseek-model` when `-llm-url` is not running |
| `-min-speedup <x>` | `1.10` | Minimum required speedup factor for `./run.sh check` CI hardware regression gate |

### 4. Core Hardware, Telemetry & SLA Flags

| Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `-addr` | `GPUMODE_ADDR` *(in `run.sh`)* | `127.0.0.1:8080` | Web Dashboard listen address |
| `-gpu` | `GPUMODE_GPU_INDEX` | `0` | Target NVIDIA GPU device index |
| `-telemetry` | `GPUMODE_TELEMETRY` | `auto` | Telemetry backend (`auto`, `nvml`, `smi`) |
| `-ebpf` | `GPUMODE_EBPF` | `auto` (`on` after `setcap` in `run.sh`) | Host ↔ CUDA eBPF tracer mode (`auto`, `on`, `off`) |
| `-trace-pid` | `GPUMODE_TRACE_PID` | `0` *(auto)* | Specific target PID tree for eBPF tracing |
| `-nvcc` | `GPUMODE_NVCC` | Auto-resolved | Path to `nvcc` CUDA compiler |
| `-sanitizer` | `GPUMODE_SANITIZER` | Auto-resolved | Path to `compute-sanitizer` |
| `-nvidia-smi` | `GPUMODE_NVIDIA_SMI` | Auto-resolved | Path to `nvidia-smi` |
| `-python` | `GPUMODE_PYTHON` | `python3` | Python interpreter for `scripts/profile_pytorch.py` |
| `-qps` | — | `120` | Target workload QPS for fleet capacity sizing |
| `-sla` | — | `25` | Target p95 latency SLA in milliseconds |
| `-json` | — | `false` | Emit structured JSON output in CLI modes |
