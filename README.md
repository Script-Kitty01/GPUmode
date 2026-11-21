# gpuMode — Intelligent GPU Resource & CUDA Hardware Optimization Platform

> **Accomplish the same AI computation with higher GPU utilization, verified CUDA kernels, lower board energy, and fewer GPUs.**

**gpuMode** is a real-time Go + CUDA + eBPF systems engineering platform for Linux and NVIDIA GPUs. It connects **host-to-CUDA kernel observability**, **Roofline bottleneck diagnosis**, **automated CUDA kernel tuning & safety verification**, and **end-to-end local LLM benchmarking** in one closed-loop pipeline:

```text
Observe (NVML 10ms + eBPF) → Diagnose (Roofline + Launch Gap) → Optimize (CUDA + Bayesian) → Verify (Sanitizer + Parity) → Measure Energy (∫P(t)dt) → Right-Size (Pareto)
```

Unlike passive GPU monitors that only display utilization percentages, **gpuMode**:
- Samples board power, clocks, utilization, VRAM, and compute PIDs at **`10 ms`** intervals directly through `libnvidia-ml.so.1` (`NVML`, zero CGo `dlopen`).
- Traces host-side CUDA driver boundaries (`cuLaunchKernel*`, `cuMemcpy*`, `cuStreamSynchronize*`) and Linux scheduler `sched_switch` off-CPU starvation via **rootless eBPF**.
- Compiles, sanitizes (`compute-sanitizer`), numerically verifies, and benchmarks parameterized CUDA C++ kernels (`nvcc -Xptxas -v`) on your live GPU.
- Benchmarks local AI models (`.gguf` via `llama-server`/`llama-cli` or `Ollama`) **Without Project (Baseline)** vs. **With Project (gpuMode Tuned)** side-by-side.

> **Looking for all `./run.sh` commands, modes, and model flags?** See the dedicated [RUN_SH_MODES.md](./RUN_SH_MODES.md) guide.

---

## Core Metrics & What gpuMode Proves

| Metric | What It Measures | How gpuMode Computes It |
| :--- | :--- | :--- |
| **`Joules / 1K Inferences` (`J / 1K Tokens`)** | Hardware Energy Efficiency | Trapezoidal integration of live GPU board & active power $E = \int_{t_0}^{t_1} P(t)\,dt$ across timed execution windows, plus Energy-Delay Product ($\text{EDP} = E \times T$). |
| **`Requests / GPU-Hour` & `Tokens / sec`** | Single-GPU Throughput Capacity | Hardware-verified sustained throughput ($3600 \times \text{RPS}$) and fleet GPU count required to meet target QPS & p95 latency SLA. |
| **Host ↔ CUDA eBPF Telemetry** | Launch & PCIe Bottlenecks | Exact `cuLaunchKernel` count (`launches/tok`), inter-launch CPU gap (`µs`), PCIe `H2D/D2H` volume (`MiB`), stream sync stalls (`ms`), and scheduler off-CPU time (`ms`). |
| **Verification Gate (`PASS / FAIL`)** | Memory & Numerical Safety | `compute-sanitizer --tool memcheck` (0 illegal accesses/races) + FP64 reference `atol/rtol` parity on standard and unaligned (`hidden_dim = 1027`) shapes. |

---

## Startup Guide

### 1. System Prerequisites

* **OS:** Linux (`x86_64` recommended for full eBPF `libcuda.so.1` uprobe tracing)
* **Hardware & Drivers:** Live NVIDIA GPU + proprietary/open NVIDIA driver (`nvidia-smi` and `libnvidia-ml.so.1`)
* **CUDA Toolkit:** `nvcc` and `compute-sanitizer` (auto-detected from `PATH`, `CUDA_HOME`, `/opt/cuda/bin`, or `/usr/local/cuda/bin`)
* **Go Toolchain:** Go `1.23+`
* **Python 3:** `python3` (optional `torch` with CUDA enables live `torch.profiler` attribution; otherwise falls back automatically to analytical hotspot tracing)
* **Optional (for Local LLM Runner & Proposer):** `llama-server` and `llama-cli` (`llama.cpp` with CUDA) and/or `ollama`
* **Optional (for Rootless eBPF):** `setcap` / `getcap` (`libcap` package)

> **Dynamic Hardware Policy:** `gpuMode` detects your actual GPU name, `sm_XX` compute capability, SM count, L2 cache, bus width, VRAM, clocks, FP32 ceiling, and DRAM bandwidth at runtime. If no NVIDIA GPU or `nvcc` is present, it exits cleanly rather than using fabricated fallback specs.

---

### 2. First-Time Setup & One-Command Launch

```zsh
# 1. Enter the repository
cd gpuMode

# 2. (Optional, one-time) Grant rootless Linux eBPF capabilities to ./gpumode
#    Prompts once via sudo so the daemon never needs to run as root
./run.sh setcap

# 3. Build (if needed) and start the live Go Daemon & Web Dashboard
./run.sh
```

Once started, open **`http://127.0.0.1:8080`** in your browser.

**What `./run.sh` does automatically on startup:**
1. Builds `./gpumode` only when source files have changed (preserving `setcap` eBPF capabilities across restarts).
2. Verifies/grants rootless eBPF capabilities (`cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep`), falling back gracefully to `-ebpf=auto` if unprivileged.
3. Runs the initial operator trace (`scripts/profile_pytorch.py`) and pre-warms an initial hardware-verified CUDA report if none is cached.
4. Starts the HTTP + Server-Sent Events (SSE) Web Dashboard at `http://127.0.0.1:8080`.

---

### 3. Using the Web Dashboard (`http://127.0.0.1:8080`)

#### A. CUDA Kernel Autotuner, Roofline & Pareto Explorer (Section 1)
1. Pick a hotspot operator (`rmsnorm_residual`, `swiglu`, `softmax`, `layernorm`, or `rmsnorm`), strategy (`BAYESIAN_UCB` or `GRID_SEARCH`), batch/token size, and hidden dimension.
2. Click **Compile, Verify & Benchmark on GPU**.
3. Inspect the **6-Question Causal Breakdown**, live **Power Sparkline**, **Host ↔ CUDA eBPF** metrics, **Roofline Chart**, **Pareto Frontier**, verified **CUDA C++ kernel**, and auto-generated **PyTorch C++/CUDA extension**.

#### B. Running Any Local AI Model — Direct Path or Auto-Detected (Section 2)
1. **Option A — Direct Model Path (no `/models` folder needed):**
   - Paste any `.gguf` file path (e.g. `~/Downloads/Qwen3.5-4B-Q4_K_M.gguf` or `/abs/path/to/model.gguf`) or model folder directly into **Direct Model Path**. The dropdown automatically selects `Direct Local Model Path`.
2. **Option B — Auto-Detected Models:**
   - Pick any model auto-discovered from `~/AI/Models`, `~/models`, `./models`, `~/.cache/huggingface/hub`, `llama-server` (`:8081`), or `Ollama` (`:11434`), or enter a custom scan directory and click **Scan Models**.
3. Click **Run Without Project (Baseline)** to measure baseline throughput, TTFT/TPOT, GPU SM %, VRAM, Watts/Joules, and eBPF launch/sync overhead.
4. Click **Run With Project (gpuMode)** to run `gpuMode`'s VRAM-fitted GPU profile (`-ngl`, FlashAttention, quantized KV cache, tuned batching) **plus** local-model-guided CUDA kernel optimization, and compare **Before vs. After** side-by-side.

---

### 4. Common Lifecycle Operations (Custom Port, Multi-GPU, Restart, Stop)

```zsh
# Start the dashboard on a custom host/port
GPUMODE_ADDR=127.0.0.1:9090 ./run.sh

# Target a specific GPU on a multi-GPU Linux host
GPUMODE_GPU_INDEX=1 ./run.sh

# Disable eBPF tracing explicitly if running in a restricted container
GPUMODE_EBPF=off ./run.sh

# Stop the foreground server
# Press Ctrl+C in the terminal running ./run.sh (graceful HTTP + NVML shutdown)
```

---

### 5. `run.sh` Startup Modes Summary

All `./run.sh` startup modes and local model flags are documented in full in **[RUN_SH_MODES.md](./RUN_SH_MODES.md)**.

| Command | Description |
| :--- | :--- |
| `./run.sh` *(or `./run.sh dashboard`)* | Launch the live Web Dashboard & SSE telemetry daemon at `http://127.0.0.1:8080` |
| `./run.sh optimize [flags]` | Run the CUDA kernel autotuner, `compute-sanitizer`, parity, and energy pipeline in CLI |
| `./run.sh deepseek [flags]` | Propose CUDA kernel configs using a local `.gguf` model (`llama-cli`) and verify on GPU |
| `./run.sh workload [flags]` | Benchmark real streaming LLM inference (`llama-server` / `Ollama`) with NVML + eBPF |
| `./run.sh monitor [flags]` | Print an instant GPU power, clock, utilization, and VRAM telemetry line |
| `./run.sh ebpf [flags]` | Inspect NVML backend, `libcuda.so.1`, active GPU PIDs, and attached eBPF uprobes |
| `./run.sh setcap` | Grant rootless Linux capabilities (`setcap`) to `./gpumode` and verify eBPF attachment |
| `./run.sh bpf-gen` | Recompile `cuda_trace.bpf.c` via `bpf2go` and rebuild `./gpumode` |
| `./run.sh test` | Run all Go unit and live CUDA hardware verification tests |

---

## Causal Engineering Chain Answered on Every Run

1. **Why is the GPU underutilized?** — Classifies operations as `MEMORY_BOUND`, `COMPUTE_BOUND`, or `LAUNCH_BOUND` using Roofline Arithmetic Intensity ($\text{FLOPs}/\text{Byte}$ vs. hardware ridge point) and eBPF inter-launch gaps.
2. **Which operations are responsible?** — Combines operator latency/DRAM attribution with live eBPF driver launch counts, PCIe `H2D/D2H` MiB, host sync stalls, and scheduler off-CPU starvation.
3. **What CUDA optimization helps?** — Tunes kernel fusion, `float4` 128-bit vectorized loads, block sizes, and `__shfl_xor_sync` warp reductions via Bayesian UCB or Grid Search.
4. **Is that optimization numerically safe?** — Gates every candidate through `compute-sanitizer --tool memcheck`, standard batch FP64 parity, and odd unaligned edge-case dimensions (`hidden_dim = 1027`).
5. **How much energy does it save?** — Integrates `10 ms` NVML power samples ($E = \int P(t)\,dt$) across interleaved windows to quantify `Joules / 1K inferences`, confidence/uncertainty, and EDP improvement.
6. **Can fewer GPUs serve the workload?** — Evaluates peak activation VRAM savings, sustained `Requests / GPU-hr`, and Multi-Objective Pareto winners against your target QPS and latency SLA.

---

## Architecture Overview

```text
                      gpuMode Go Daemon (HTTP + SSE Server & CLI)
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
1. CONCURRENT TELEMETRY & ENERGY   2. WORKLOAD & MODEL RUNNER  3. HARDWARE PHYSICS & SLA ENGINE
   - Direct NVML (10ms) + SMI         - Auto-Scan & Direct .gguf  - Runtime CUDA Device Probe
   - Rootless eBPF CUDA + Scheduler   - Baseline vs Tuned Runner  - Roofline Ridge Point & Peak BW
   - E = ∫ P(t) dt (Joules), EDP      - PyTorch / Operator Trace  - QPS Capacity & Latency SLA
       └──────────────────────────────────┼──────────────────────────────────┘
                                          ▼
                        4. CUDA & MEMORY OPTIMIZATION ENGINE
                           - Bottleneck-Driven Kernel Selector (RMSNorm, SwiGLU, Softmax, LayerNorm)
                           - Parallel nvcc Worker Pool + Serialized GPU Benchmark Queue
                           - Bayesian UCB + Grid Search Autotuner J(θ) = αT + βE + γM
                           - Pluggable KernelProposer (Deterministic + Local GGUF via llama-cli)
                                          │
                                          ▼
                           5. HARDWARE VERIFICATION ENGINE
                           - nvcc -Xptxas -v (Registers, Shared Mem, Spills)
                           - compute-sanitizer --tool memcheck
                           - Standard + Odd-Dimension (1027) FP64 Numerical Parity
                           - Interleaved CUDA Events + Dynamic 2x L2 Flush + Power Sync
                                          │
                                          ▼
                        6. MULTI-OBJECTIVE PARETO & WEB DASHBOARD
                           - Non-Dominated Pareto Frontier (Latency, Energy, VRAM)
                           - Live SSE Telemetry, Before/After LLM Table, PyTorch C++ Extension
```

---

## Repository Structure

```text
gpuMode/
├── cmd/gpumode/
│   └── main.go                        # CLI & Real-Time HTTP/SSE Daemon entrypoint
├── internal/
│   ├── config/                        # Dynamic CUDA binary, NVML/eBPF, model, and SLA config
│   ├── nvml/                          # Pure-Go libnvidia-ml.so.1 bindings (10ms power/clock/PIDs)
│   ├── models/                        # Multi-source model discovery & direct .gguf path resolver
│   ├── ebpftrace/                     # Kernel BPF C program + libcuda uprobes & sched_switch tracer
│   ├── physics/                       # Live cudaGetDeviceProperties probe & capacity/energy KPIs
│   ├── telemetry/                     # Concurrent NVML/SMI sampler, SSE broadcaster, & E = ∫P(t)dt
│   ├── roofline/                      # Arithmetic Intensity, ridge points, & bottleneck classifier
│   ├── harness/                       # Parallel nvcc -Xptxas -v worker pool & GPU verification runner
│   ├── optimizer/                     # Parameterized CUDA C++ catalog, Bayesian autotuner, GGUF proposer
│   ├── pareto/                        # Multi-objective non-dominated Pareto solver
│   ├── profiler/                      # End-to-end causal pipeline & PyTorch C++ extension generator
│   ├── store/                         # Thread-safe local JSON persistence (.gpumode_data)
│   ├── server/                        # REST API (/api/state, /api/models, /api/optimize, /api/workload) & UI
│   └── workload/                      # Streaming llama-server / Ollama benchmark (Baseline vs Tuned)
├── scripts/
│   └── profile_pytorch.py             # PyTorch operator profiler bridge with standalone fallback
├── run.sh                             # Incremental build, setcap, & multi-mode launcher
└── RUN_SH_MODES.md                    # Complete reference for all ./run.sh modes & model options
```

---

## Environment Variables Reference

| Variable | Purpose |
| :--- | :--- |
| `GPUMODE_ADDR` | HTTP listen address used by `./run.sh` (default `127.0.0.1:8080`) |
| `GPUMODE_GPU_INDEX` | NVIDIA GPU index passed to CUDA, NVML, and `nvidia-smi` (default `0`) |
| `GPUMODE_TELEMETRY` | Telemetry backend: `auto` (default, prefers `NVML`), `nvml`, or `smi` |
| `GPUMODE_NVML_LIB` | Explicit path override for `libnvidia-ml.so.1` |
| `GPUMODE_EBPF` | Host ↔ CUDA eBPF tracer mode: `auto` (default), `on`, or `off` |
| `GPUMODE_TRACE_PID` | Explicit target PID for eBPF CUDA & scheduler tracing (`0` = auto-discover) |
| `GPUMODE_LIBCUDA` | Explicit path override for `libcuda.so.1` (CUDA Driver API uprobes) |
| `GPUMODE_LIBCUDART` | Explicit path override for `libcudart.so` (CUDA Runtime API fallback uprobes) |
| `GPUMODE_NVCC` | `nvcc` executable override |
| `GPUMODE_SANITIZER` | `compute-sanitizer` executable override |
| `GPUMODE_NVIDIA_SMI` | `nvidia-smi` executable override |
| `GPUMODE_PYTHON` | Python interpreter for `scripts/profile_pytorch.py` |
| `GPUMODE_MODEL_PATH` | Explicit `.gguf` model file override |
| `GPUMODE_MODEL_DIR` | Additional directory scanned for `.gguf` models |
| `GPUMODE_MODEL_NAME` | Default served model name/tag for `workload` benchmarks |
| `GPUMODE_NGL` | Explicit GPU layer offload override (`-ngl`) for `llama-server` / `llama-cli` |
| `GPUMODE_PASSES_PER_REQ` | Layer passes per inference used in kernel benchmark harnesses |

---

## Independent Verification on Linux

You can independently cross-check every metric reported by **gpuMode** using standard Linux/NVIDIA tools:

```zsh
# 1. Watch live GPU SM %, Memory BW %, board Watts, and per-PID VRAM in a second terminal
nvtop

# 2. Cross-check raw Baseline vs Tuned token generation speed with llama.cpp's official benchmark
llama-bench -m /path/to/model.gguf -ngl 24 -fa off -ctk f16 -ctv f16 -b 512 -ub 256 -p 128 -n 48
llama-bench -m /path/to/model.gguf -ngl 99 -fa on  -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -p 128 -n 48

# 3. Cross-check eBPF kernel launch counts, stream syncs, and PCIe H2D/D2H transfers with Nsight Systems
nsys profile --trace=cuda --stats=true llama-bench -m /path/to/model.gguf -ngl 99 -fa on -p 0 -n 48 -r 1
```

---

## Example Verified CLI Output (Live Hardware Run)

*(Measured on an NVIDIA GeForce RTX 4050 Laptop GPU `sm_89` — values are dynamically measured at runtime, never hardcoded.)*

```text
================================================================================
 gpuMode — CUDA Hardware & Energy Verification Report (NVIDIA GeForce RTX 4050 Laptop GPU / sm_89)
================================================================================
Observability: Telemetry: NVML | Host-CUDA eBPF: MEASURED_EBPF (76391 launches, 38.8 µs avg gap, 433.38 ms sync stall, 41172.16 ms off-CPU)

1. Why is the GPU underutilized?
   [LAUNCH_BOUND] GPU SMs are starved at the host-device launch boundary (0.0% compute util, 6.7% memory bandwidth util): eBPF measured 38.8 µs inter-launch host gap and 100.0% off-CPU scheduler starvation across 4 kernel launches.

2. Which operations are responsible?
   [ESTIMATED] RMSNorm + Residual Add (Unfused) accounts for 34.5% of memory-bound execution time, issuing multiple unfused kernel launches and transferring 12.9 GB/s across DRAM while keeping FP32 compute units at only 0.0% capacity.

3. What optimization helps?
   Selected RMSNorm + Residual — Fused Scalar + Warp Shuffle (Block 256) (Block=256, VecWidth=float1, 40 registers/thread, 128 B shared mem), fusing intermediate memory round-trips into registers and using __shfl_xor_sync warp reductions to achieve a 2.02× measured speedup (2.268 ms → 1.121 ms).

4. Is that optimization numerically safe?
   VERIFIED SAFE: Passed compute-sanitizer memcheck (0 illegal accesses/races), standard batch numerical parity, and odd unaligned dimension (hidden_dim=1027) edge-case verification with max absolute error 2.00e-07 (< 1e-3 threshold).

5. How much energy does it save?
   [MEASURED, MEDIUM confidence, ±16.2% uncertainty across 3 windows] Reduces GPU board energy by 38.1% (34.66 J → 21.44 J per 1K inferences) and improves EDP by 3.38×.

6. Can fewer/cheaper GPUs serve the workload?
   [MEASURED] Peak activation VRAM drops by 8 MiB (20 → 12 MiB) and single-GPU capacity rises from 441 to 892 req/s (1,587,162 → 3,210,559 Requests/GPU-hr). Fleet requirement drops from 1 to 1 GPU(s) (Pareto Balanced winner: RMSNorm + Residual — Fused Scalar + Warp Shuffle (Block 256)).
================================================================================
```
