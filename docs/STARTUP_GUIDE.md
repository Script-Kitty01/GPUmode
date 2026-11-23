# Startup & Web Dashboard Guide

[← Back to README](../README.md) • [All `run.sh` Modes & Model Flags](./RUN_SH_MODES.md) • [Dependencies](./DEPENDENCIES.md)

---

## 1. Quick First-Time Setup

### Step 1 — Verify Prerequisites
Ensure your Linux host has an NVIDIA GPU with proprietary/open kernel drivers (`nvidia-smi`), the CUDA compiler (`nvcc`) and memory checker (`compute-sanitizer`), Go, Python 3, and Zsh:

```zsh
nvidia-smi -L
nvcc --version
compute-sanitizer --version
go version
python3 --version
```

### Step 2 — Clone & Optional One-Time Setup

```zsh
# 1. Clone and enter the repository
git clone https://github.com/Script-Kitty01/GPUmode.git
cd GPUmode

# 2. (Optional, one-time) Grant rootless Linux eBPF capabilities to ./gpumode
#    Prompts once via sudo so the server never needs to run as root
./run.sh setcap

# 3. (Optional) Set up a local .venv with PyTorch + CUDA for live torch.profiler
#    and torch.utils.cpp_extension.load_inline verification (auto-detected by gpuMode)
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip torch

# 4. Build (if needed) and launch the Web Dashboard
./run.sh
```

Open **`http://127.0.0.1:8080`** in your browser.

### What `./run.sh` Does on Launch
1. Performs an incremental build check (`./gpumode` is rebuilt only when `cmd/**/*.go`, `internal/**/*.go`, `internal/**/*.o`, or `frontend/**/*` files change, preserving `setcap` capabilities).
2. Verifies rootless eBPF capabilities (`cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep`), falling back gracefully to `-ebpf=auto` if unprivileged.
3. Auto-detects `./.venv/bin/python3` (when present) and runs `scripts/profile_pytorch.py --verify-extension` to capture live `torch.profiler` operator attribution and inline C++/CUDA extension verification.
4. Pre-warms an initial hardware-verified CUDA report if `.gpumode_data` is empty.
5. Starts the HTTP + Server-Sent Events (SSE) Web Dashboard at `http://127.0.0.1:8080`.

---

## 2. Using the Web Dashboard (`http://127.0.0.1:8080`)

### Section 1 — Live CUDA Hardware Optimization, Occupancy & PyTorch Verification
1. Choose any of the **8 target LLM operators** (`rmsnorm_residual`, `rmsnorm`, `layernorm`, `softmax`, `swiglu`, `rope_qkv`, `qgemv_decode`, or `gqa_attention`), tuning strategy (`BAYESIAN_UCB` or `GRID_SEARCH`), batch size, and hidden dimension.
2. Click **Compile, Verify & Benchmark on GPU**.
3. Inspect the **PyTorch `2.15 [MEASURED]` Profiler & Inline Extension Banner**, **6-Question Causal Verification Grid**, **Candidate Kernel & SM Warp Occupancy Table** (`PTXAS Regs / SMEM / Spills`, `48/48 warps = 100%`), live **Power Sparkline**, **Host ↔ CUDA eBPF** metrics (with one-click **Export Perfetto JSON**, **Prometheus `/metrics`**, and **Scheduler JSON**), verified **CUDA C++ kernel**, and self-contained **PyTorch C++/CUDA extension**.

### Section 2 — Local AI Model Runner (Baseline vs. gpuMode Tuned)
1. **Option A — Direct Model Path (no `/models` folder needed):**
   - Paste any `.gguf` file path (e.g. `~/Downloads/Qwen3.5-4B-Q4_K_M.gguf` or `/abs/path/to/model.gguf`) or model directory into **Direct Model Path**.
   - The **Model Source** dropdown automatically switches to `Direct Local Model Path`.
2. **Option B — Auto-Detected Models:**
   - Select any model auto-discovered from `~/AI/Models`, `~/models`, `./models`, `~/.cache/huggingface/hub`, `llama-server` (`:8081`), or `Ollama` (`:11434`), or specify an optional folder and click **Scan Models**.
3. Configure **Streaming Requests**, **Concurrent Clients (`1..32`)**, and **Max Output Tokens / Req**.
4. Click **Run Without Project (Baseline)** to record baseline throughput, TTFT/TPOT, GPU SM %, VRAM, Watts/Joules, **`$ / 1M Tokens` (On-Demand & Spot)**, and eBPF launch/sync overhead.
5. Click **Run With Project (gpuMode)** to apply VRAM-fitted GPU tuning (`-ngl`, FlashAttention, quantized KV cache, tuned batching) **plus** local-model-guided CUDA kernel optimization, then compare **Before vs. After** side-by-side.

### Section 3 — Production GPU Fleet Cost, VRAM Placement & Admission Control Optimizer
1. Set your **Target Fleet Traffic (`QPS`)**, **Decode TPOT Latency SLO (`ms/tok`)**, **Local / Base GPU Rate (`$/GPU-hr`)**, and **Spot / Preemptible Discount (`%`)**, then click **Recompute Fleet Placement & Cost**.
2. Inspect the **Recommended On-Demand Pool** and **Recommended Spot / Preemptible Pool** cards (showing full model + KV-cache VRAM fit %, estimated `ms/tok`, `$ / 1M tokens`, and required GPU count).
3. Review the **Priority Admission Control & Backpressure Gate** table (`CRITICAL_REALTIME`, `STANDARD_INTERACTIVE`, `BATCH_SPOT_INTERRUPTIBLE` with `HTTP 429` backpressure and Spot offloading) and the **Multi-Strategy Fleet Scheduling Simulator** table (`ROUND_ROBIN` vs. `LEAST_LOADED_VRAM_AWARE` vs. `COST_SLO_WEIGHTED_PARETO`).

---

## 3. Common Startup Customizations

```zsh
# Custom listen address / port
GPUMODE_ADDR=127.0.0.1:9090 ./run.sh

# Select GPU #1 on a multi-GPU Linux host
GPUMODE_GPU_INDEX=1 ./run.sh

# Force nvidia-smi telemetry instead of direct NVML
GPUMODE_TELEMETRY=smi ./run.sh

# Disable eBPF tracing (e.g., inside restricted containers)
GPUMODE_EBPF=off ./run.sh

# Stop the foreground server cleanly
# Press Ctrl+C in the terminal running ./run.sh
```

---

## 4. Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `GPUMODE_ADDR` | `127.0.0.1:8080` | HTTP listen address used by `./run.sh` |
| `GPUMODE_GPU_INDEX` | `0` | NVIDIA GPU index passed to CUDA, NVML, and `nvidia-smi` |
| `GPUMODE_TELEMETRY` | `auto` | Telemetry backend (`auto` prefers `NVML`, `nvml`, or `smi`) |
| `GPUMODE_NVML_LIB` | Auto-resolved | Explicit path override for `libnvidia-ml.so.1` |
| `GPUMODE_EBPF` | `auto` | Host ↔ CUDA eBPF mode (`auto`, `on`, or `off`) |
| `GPUMODE_TRACE_PID` | `0` *(auto)* | Specific target PID tree for eBPF CUDA & scheduler tracing |
| `GPUMODE_LIBCUDA` | Auto-resolved | Explicit path override for `libcuda.so.1` |
| `GPUMODE_LIBCUDART` | Auto-resolved | Explicit path override for `libcudart.so` |
| `GPUMODE_NVCC` | Auto-resolved | `nvcc` compiler path override |
| `GPUMODE_SANITIZER` | Auto-resolved | `compute-sanitizer` path override |
| `GPUMODE_NVIDIA_SMI` | Auto-resolved | `nvidia-smi` path override |
| `GPUMODE_PYTHON` | `python3` | Python interpreter for `scripts/profile_pytorch.py` |
| `GPUMODE_MODEL_PATH` | Auto-discovered | Explicit `.gguf` model file override |
| `GPUMODE_MODEL_DIR` | — | Additional directory scanned for `.gguf` models |
| `GPUMODE_MODEL_NAME` | Auto-discovered | Default served model name/tag for `workload` benchmarks |
| `GPUMODE_NGL` | Auto-sized | Explicit GPU layer offload override (`-ngl`) for `llama-server` / `llama-cli` |
| `GPUMODE_PASSES_PER_REQ` | Default | Layer passes per inference used in kernel benchmark harnesses |

---

## 5. REST API & Observability Endpoints (`http://127.0.0.1:8080`)

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| **`/`** | `GET` | Serves the modular Web Dashboard assembled from `frontend/components/*.html` |
| **`/static/*`** | `GET` | Serves embedded static stylesheet (`/static/css/dashboard.css`) and JS modules (`/static/js/*.js`) |
| **`/api/state`** | `GET` | Returns full GPU hardware specs, empirical DRAM/L2 ceilings, latest causal report, telemetry history, eBPF status, and discovered models |
| **`/api/models`** | `GET` | Scans local directories (`?dir=/path`), Hugging Face cache, `llama-server`, and `Ollama` for available AI models |
| **`/api/optimize`** | `POST` | Runs the live CUDA compilation, `compute-sanitizer`, occupancy, energy integration, and Bayesian/Grid autotuning loop |
| **`/api/recompute`** | `POST` | Recomputes unit economics, Pareto recommendations, and fleet sizing under updated QPS/SLA/hourly rate parameters |
| **`/api/workload`** | `POST` | Executes streaming LLM inference benchmarks (`baseline` vs. `optimized`) with concurrent client workers (`1..32`) |
| **`/api/schedule`** | `GET`/`POST` | Evaluates VRAM-aware GPU pool placement, priority admission control (`HTTP 429` backpressure), Spot routing, and 3-strategy scheduling simulation |
| **`/api/telemetry/history`** | `GET` | Returns the recent `10 ms` / background NVML telemetry ring buffer |
| **`/api/stream/telemetry`** | `GET` | Live Server-Sent Events (`text/event-stream`) stream of GPU power, clocks, temperature, utilization, VRAM, and throttle reasons |
| **`/api/trace/perfetto`** | `GET` | Exports a Chrome Trace Event / [ui.perfetto.dev](https://ui.perfetto.dev) JSON timeline of eBPF CUDA launches, sync stalls, and `sched_switch` intervals |
| **`/metrics`** | `GET` | Exports live NVML, empirical bandwidth ceiling, and eBPF gauges in Prometheus exposition format |

---

## 6. Running Self-Hosted CUDA Hardware CI on GitHub Actions

To execute the **`Live NVIDIA GPU Hardware Gate (Self-Hosted Runner)`** job in [`.github/workflows/cuda-hardware.yml`](../.github/workflows/cuda-hardware.yml) on your own Linux + NVIDIA GPU machine:

```zsh
# 1. Download and extract the GitHub Actions runner in a temporary directory
mkdir -p /tmp/gpumode_gh_runner && cd /tmp/gpumode_gh_runner
curl -fsSL -o actions-runner.tar.gz https://github.com/actions/runner/releases/download/v2.338.2/actions-runner-linux-x64-2.338.2.tar.gz
tar xzf actions-runner.tar.gz

# 2. Register an ephemeral runner with labels [self-hosted, linux, cuda] via GitHub CLI
REG_TOKEN=$(gh api -X POST repos/Script-Kitty01/GPUmode/actions/runners/registration-token --jq .token)
./config.sh --url https://github.com/Script-Kitty01/GPUmode --token "$REG_TOKEN" --name "cuda-gpu-runner" --labels "self-hosted,linux,cuda" --ephemeral --unattended --replace

# 3. Trigger the workflow with run_self_hosted_gpu=true and start the runner
gh workflow run cuda-hardware.yml -f run_self_hosted_gpu=true
./run.sh
```

