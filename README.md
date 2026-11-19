# gpuMode — Intelligent GPU Infrastructure & CUDA-to-Economics Optimization Platform

> **Don't just make GPU workloads faster. Accomplish the same amount of AI computation using fewer GPUs, less electricity, and less money.**

**gpuMode** is a real-time Go + CUDA systems engineering platform that bridges **low-level CUDA hardware physics** directly to **infrastructure & energy economics** in one closed-loop pipeline:

```text
Observe → Diagnose (Roofline) → Optimize (CUDA + Bayesian) → Verify (Sanitizer + Parity) → Measure Energy → Multi-Objective Pareto → Calculate Savings
```

Unlike standard GPU dashboards that merely report utilization percentages, **gpuMode** compiles, sanitizes, verifies, and benchmarks parameterized CUDA C++ kernels on physical NVIDIA hardware while integrating live GPU power draw $E = \int_{t_0}^{t_1} P(t)\,dt$ to answer seven causal engineering and financial questions.

gpuMode is NVIDIA-CUDA dynamic:

- It requires a live NVIDIA GPU, `nvidia-smi`, `nvcc`, and `compute-sanitizer`.
- GPU name, compute capability, SM count, clocks, VRAM, bus width, L2 cache, FP32 ceiling, and memory bandwidth are detected at runtime.
- It does not use a fabricated RTX 4050, `sm_89`, VRAM, power, or clock fallback.
- The target GPU architecture is passed to `nvcc`; use `-gpu <index>` to select another NVIDIA GPU in a multi-GPU system.
- CUDA binaries are resolved through CLI flags, `GPUMODE_*` environment variables, `PATH`, `CUDA_HOME`/`CUDA_PATH`, and standard install paths.
- Reference GPU SKUs used in the Digital Twin remain clearly labeled as **SIMULATED** and can be overridden with `-presets <path>`.

---

## Headline Business & Systems KPIs

Every optimization and right-sizing run in **gpuMode** is evaluated against three primary efficiency metrics:

| Headline Metric | What It Proves | How gpuMode Computes It |
| :--- | :--- | :--- |
| **`$ / 1M Inferences`** | Financial Efficiency | Combines amortized/cloud GPU hourly cost (`$/GPU-hr`) + PUE-adjusted server electricity cost (`$/kWh`) per 1,000,000 completed requests. |
| **`Joules / 1K Inferences`** | Energy Efficiency | Trapezoidal integration of live GPU board power $E = \int_{t_0}^{t_1} P(t)\,dt$ across CUDA-event timed execution windows, plus Energy-Delay Product ($\text{EDP} = E \times T$) and Performance/Watt. |
| **`Requests / GPU-Hour`** | Hardware Capacity Efficiency | Verified sustained single-GPU throughput capacity ($3600 \times \text{RPS}$) within the target p95 latency SLA. |

---

## The 7-Step CUDA-to-Economics Causal Chain (Live NVIDIA GPU Reference)

For every workload, **gpuMode** traces the full causal chain from low-level CUDA physics to monthly infrastructure and energy savings:

1. **Why is the GPU underutilized?**
   - **Roofline Diagnosis:** Uses the **Roofline Performance Model** ($\text{Arithmetic Intensity} = \text{FLOPs}/\text{Bytes}$ vs. the GPU's hardware ridge point) to classify bottlenecks as `MEMORY_BOUND`, `COMPUTE_BOUND`, or `LAUNCH_BOUND`.
   - *Verified `sm_89` Example:* `[MEMORY_BOUND] Operation has low Arithmetic Intensity (0.25 FLOPs/Byte vs GPU ridge point 54.8 FLOPs/Byte). Floating-point cores sit at 0.0% utilization waiting on DRAM memory transfers (4.3 GB/s achieved of 192.0 GB/s peak).`
2. **Which operations are responsible?**
   - **Operator Attribution:** Attributes latency share, DRAM memory bandwidth consumption, kernel launch overhead, and peak activation VRAM across hotspot operators (`RMSNorm + Residual`, `SwiGLU`, `Online Softmax`, `LayerNorm`).
   - *Verified `sm_89` Example:* `Attention Softmax (3-Pass) accounts for 22.5% of memory-bound execution time, issuing multiple unfused kernel launches and transferring 4.3 GB/s across DRAM while keeping FP32 compute units at only 0.0% capacity.`
3. **What CUDA optimization helps?**
   - **Targeted Kernel Transformation:** Selects and tunes kernel fusion (eliminating intermediate global memory round-trips), 128-bit `float4` vectorized memory coalescing, and register-level `__shfl_xor_sync` warp reductions.
   - *Verified `sm_89` Example:* `Selected Online Softmax — Fused Scalar + Warp Shuffle (Block 128, VecWidth=float1, 40 registers/thread, 128 B shared mem), fusing intermediate memory round-trips into registers and using __shfl_xor_sync warp reductions to achieve a 2.25× measured speedup (4.880 ms → 2.168 ms).`
4. **Is that optimization numerically safe?**
   - **Automated Verification Gate:** Gates every candidate kernel through `compute-sanitizer --tool memcheck`, standard batch numerical parity (`atol/rtol` vs. FP64 CPU reference), and odd unaligned edge-case dimensions (`hidden_dim = 1027`, non-divisible by 4).
   - *Verified `sm_89` Example:* `VERIFIED SAFE: Passed compute-sanitizer memcheck (0 illegal accesses/races), standard batch numerical parity, and odd unaligned dimension (hidden_dim=1027) edge-case verification with max absolute error 2.00e-07 (< 1e-3 threshold).`
5. **How much energy does it save?**
   - **Live Power Integration:** Integrates live GPU power sensors ($E = \int_{t_0}^{t_1} P(t)\,dt$) during execution to report board `Joules / 1K inferences`, PUE-adjusted server `kWh/month` saved, Energy-Delay Product ($\text{EDP}$) improvement, and grid carbon reduction ($\text{kg CO}_2\text{e/month}$).
   - *Verified `sm_89` Example:* `Reduces GPU board energy by 69.7% (42.68 J → 12.95 J per 1K inferences at 6.0 W average board power), improves Energy-Delay Product (EDP) by 7.42×, and saves kWh/month (including PUE).`
6. **Can fewer or cheaper GPUs now serve the workload?**
   - **Pareto & Digital Twin Right-Sizing:** Combines **Multi-Objective Pareto Optimization**, **Digital Twin SKU Simulation**, and **Multi-Workload Bin-Packing** to determine whether throughput gains or VRAM activation savings unlock fewer instances or a cheaper hardware tier.
   - *Verified `sm_89` Example:* `YES: Peak activation VRAM drops by 8 MiB (532 → 524 MiB) and single-GPU capacity rises from 205 to 461 req/s (737,644 → 1,660,670 Requests/GPU-hr). Recommended Digital Twin target: Optimized on 1× NVIDIA L4 24GB (CUDA Kernel Speedup (Fusion + float4)); Pareto Balanced winner: Online Softmax — Fused Scalar + Warp Shuffle (Block 128).`
7. **How much money does the business save monthly?**
   - **Unit & Fleet Economics:** Quantifies unit cost reduction (`$/1M inferences`) and monthly/annual savings (`$/month`, `$/year`) across GPU infrastructure and datacenter electricity under user-configurable tariffs and SLAs.
   - *Verified `sm_89` Example:* `Cuts unit cost by 55.6% ($2.0443 → $0.9077 per 1M inferences) in combined GPU infrastructure and PUE-adjusted electricity costs.`

---

## Architecture: Real-Time Go Daemon + 7 Connected Engines

```text
                      gpuMode Go Daemon (HTTP + SSE Server & CLI)
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
1. CONCURRENT TELEMETRY & ENERGY   2. WORKLOAD & ROOFLINE      3. HARDWARE PHYSICS & COST ENGINE
   - Goroutine Ring-Buffer + SSE      - CUDA Hardware Probe       - Auto-detected GPU FP32/BW/VRAM
   - Live Power P(t), Clocks, Temp    - PyTorch / Workload Trace  - User-Configurable $/GPU-hr & $/kWh
   - E = ∫ P(t) dt (Joules), EDP      - Arithmetic Intensity (AI) - Server PUE & Grid Carbon (CO2e)
       └──────────────────────────────────┼──────────────────────────────────┘
                                          ▼
                        4. CUDA & MEMORY OPTIMIZATION ENGINE
                           - Bottleneck-Driven Kernel Selector
                           - Parameterized CUDA C++ Kernels for the detected NVIDIA architecture
                           - Parallel nvcc Worker Pool + Serialized GPU Queue
                           - Grid Search + Bayesian Autotuner J(θ)=αT+βE+γM
                           - VRAM Activation & Fusion Memory Analyzer
                           - Pluggable KernelProposer Interface
                                          │
                                          ▼
                           5. HARDWARE VERIFICATION ENGINE
                           - nvcc + -Xptxas -v (Registers, Shared Mem, Spills)
                           - compute-sanitizer (memcheck)
                           - Multi-shape Edge-Case Numerical Parity (atol/rtol)
                           - Interleaved CUDA Events + Dynamic 2x L2-Cache Flush + Power Sync
                                          │
                                          ▼
                        6. MULTI-OBJECTIVE & DIGITAL TWIN ENGINE
                           - Pareto Frontier (Latency, Energy, Cost, VRAM)
                           - Modes: Performance | Energy-Saving | Cost-Saving | Balanced
                           - Digital Twin Hardware Capacity & Fleet Right-Sizing
                           - Multi-Workload Bin-Packing Scheduler (J = αC + βE + γL)
                           - Holt Double-Exponential Predictive Autoscaler
                                          │
                                          ▼
                        7. LOCAL WEB DASHBOARD (Live SSE + Interactive UI)
                           - 7-Step Causal Chain Breakdown (CUDA -> Energy -> $/month)
                           - Live Power/Clock/VRAM Telemetry Stream (SSE)
                           - Interactive Roofline Chart & Pareto Frontier Explorer
                           - Custom Rate Controls ($/GPU-hr, $/kWh, PUE, Carbon, SLA, QPS)
                           - Auto-Generated Drop-in PyTorch C++/CUDA Extension
```

---

## Repository Structure

```text
gpuMode/
├── cmd/gpumode/
│   └── main.go                        # CLI & Real-Time HTTP/SSE Daemon entrypoint
├── internal/
│   ├── config/
│   │   └── config.go                  # Daemon, tariff, PUE, carbon, and SLA configuration
│   ├── physics/
│   │   ├── physics.go                 # Live CUDA cudaGetDeviceProperties + nvidia-smi hardware probe
│   │   ├── economics.go               # $/1M inferences, Joules/1K, Requests/GPU-hr, and monthly ROI
│   │   ├── presets.json               # Reference NVIDIA SKU presets for the Digital Twin
│   │   └── physics_test.go
│   ├── telemetry/
│   │   ├── telemetry.go               # Concurrent ring-buffer, SSE pub/sub, E = ∫P(t)dt, & DVFS simulator
│   │   └── telemetry_test.go
│   ├── roofline/
│   │   ├── roofline.go                # Arithmetic Intensity, ridge points, & bottleneck classifier
│   │   └── roofline_test.go
│   ├── harness/
│   │   ├── harness.go                 # Parallel nvcc -Xptxas -v worker pool & serialized GPU runner
│   │   └── harness_test.go
│   ├── optimizer/
│   │   ├── catalog.go                 # Parameterized CUDA C++ kernels (RMSNorm, LayerNorm, Softmax, SwiGLU)
│   │   ├── autotuner.go               # Gaussian-Surrogate Bayesian UCB & Grid Search autotuner
│   │   ├── deepseek_local.go          # Optional hardware-aware local GGUF proposer
│   │   ├── optimizer_test.go
│   │   └── deepseek_fixture_test.go   # Pluggable KernelProposer verification with DeepSeek-9B fixture
│   ├── pareto/
│   │   ├── pareto.go                  # Multi-objective non-dominated Pareto solver (4 business modes)
│   │   └── pareto_test.go
│   ├── twin/
│   │   ├── twin.go                    # Digital Twin hardware simulator, bin-packer, & predictive autoscaler
│   │   └── twin_test.go
│   ├── profiler/
│   │   ├── causal.go                  # 7-Step CausalChainReport pipeline & PyTorch C++ extension generator
│   │   └── causal_test.go
│   ├── store/
│   │   ├── store.go                   # Thread-safe local JSON persistence for optimization history
│   │   └── store_test.go
│   └── server/
│       ├── server.go                  # REST API (/api/state, /api/optimize, /api/recompute) & SSE stream
│       ├── dashboard.go               # Embedded zero-dependency interactive Web Dashboard
│       └── server_test.go
│   └── workload/
│       └── workload.go                # Optional llama-server / Ollama streaming benchmark
├── scripts/
│   └── profile_pytorch.py             # PyTorch operator profiler bridge with standalone fallback
├── run.sh                             # One-command build & launch script
└── go.mod
```

---

## Quick Start

### Prerequisites
- **Go** `1.23+`
- **NVIDIA CUDA Toolkit (`nvcc` & `compute-sanitizer`)** and **`nvidia-smi`**
- **Python 3** (optional `torch` for live PyTorch script profiling; automatically falls back to standalone CUDA trace if `torch` is not installed)

### 1. One-Command Launch (`./run.sh`)
Simply execute:
```bash
./run.sh
```
This automatically compiles `gpumode`, runs the operator profiler trace, and launches the interactive **Web Dashboard** at **`http://127.0.0.1:8080`**.

### 2. CLI Commands
You can also run specific modes directly via `./run.sh` or `./gpumode`:

```bash
# Run the full 7-Step CUDA-to-Economics optimization pipeline in the terminal
./run.sh optimize -op rmsnorm_residual -batch 256 -hidden 4096 -iters 40 -qps 350

# Optimize other LLM hotspot operations (swiglu, softmax, layernorm, rmsnorm)
./run.sh optimize -op swiglu -strategy BAYESIAN_UCB

# Run live optimization using your local DeepSeek-V4-9B GGUF model via llama-cli
./run.sh deepseek -op rmsnorm_residual -batch 256 -hidden 4096 -iters 40 -qps 350

# Benchmark real end-to-end streaming inference on your local DeepSeek-V4-9B / Ollama server (TTFT, TPOT, VRAM, J/1K tokens)
./run.sh workload -llm-reqs 3 -llm-tokens 48

# Print a live GPU hardware & power telemetry snapshot
./run.sh monitor

# Run the complete unit & live GPU hardware verification test suite
./run.sh test
```

### Dynamic NVIDIA/CUDA Configuration

The running GPU is detected from `nvidia-smi` and `cudaGetDeviceProperties`; no GPU-specific values are
used as runtime fallbacks. If hardware probing fails, gpuMode exits instead of reporting fabricated
hardware.

```bash
# Select a GPU and override CUDA tools
./gpumode dashboard \
  -gpu 0 \
  -nvcc /path/to/nvcc \
  -sanitizer /path/to/compute-sanitizer \
  -nvidia-smi /usr/bin/nvidia-smi

# Override common paths with environment variables
GPUMODE_GPU_INDEX=0 \
GPUMODE_NVCC=/path/to/nvcc \
GPUMODE_SANITIZER=/path/to/compute-sanitizer \
GPUMODE_NVIDIA_SMI=/usr/bin/nvidia-smi \
./gpumode monitor
```

The following environment variables are supported:

| Variable | Purpose |
|---|---|
| `GPUMODE_GPU_INDEX` | NVIDIA GPU index passed to CUDA and `nvidia-smi` |
| `GPUMODE_NVCC` | `nvcc` executable override |
| `GPUMODE_SANITIZER` | `compute-sanitizer` executable override |
| `GPUMODE_NVIDIA_SMI` | `nvidia-smi` executable override |
| `GPUMODE_PYTHON` | Python interpreter for PyTorch profiling |
| `GPUMODE_PRESETS_PATH` | External Digital Twin GPU preset JSON |
| `GPUMODE_MODEL_PATH` | GGUF model file override |
| `GPUMODE_MODEL_DIR` | Directory containing a discoverable GGUF model |
| `GPUMODE_NGL` | GPU layers passed to `llama-server`/`llama-cli` |
| `GPUMODE_PASSES_PER_REQ` | Configurable layer passes used in kernel benchmark harnesses |

### Optional End-to-End LLM Workload Benchmark

The main gpuMode dashboard is available on port `8080`. The LLM benchmark is optional and expects a
running OpenAI-compatible inference server or an Ollama server. It does not require a separate dashboard.

When no server is running, gpuMode can start `llama-server` for a configured local GGUF model:

```bash
GPUMODE_MODEL_PATH=/path/to/model.gguf ./gpumode workload \
  -llm-url http://127.0.0.1:8081 \
  -llm-reqs 3 \
  -llm-tokens 48
```

Or start the server separately:

```bash
llama-server -m /path/to/model.gguf -ngl 24 -c 1024 \
  --host 127.0.0.1 --port 8081
```

The blue graph in the dashboard is a live power sparkline: the latest 60 `nvidia-smi` power samples
display recent GPU power changes and idle/load transitions. It is separate from the cumulative trapezoidal
Joules energy measurement and workload economics.

---

## Example Verified Output (on an NVIDIA GPU)

The following values are a historical verification example from an RTX 4050 Laptop GPU. They are
not defaults or runtime fallbacks: gpuMode reports the live GPU name and architecture detected on the
current machine.

```text
================================================================================
 gpuMode — 7-Step CUDA-to-Economics Verification Report (NVIDIA GeForce RTX 4050 Laptop GPU / sm_89)
================================================================================
Headline Business Metrics:
  • $ / 1M Inferences      : $2.0039 -> $0.8506 (-57.5%)
  • Joules / 1K Inferences : 40.26 J -> 16.82 J (-58.2% GPU energy)
  • Requests / GPU-Hour    : 752493 -> 1772700 (2.36x speedup)
  • Monthly Savings        : $1100.79/mo ($13209.48/yr) | 41.3 kWh/mo | 15.9 kg CO2e/mo

1. Why is the GPU underutilized?
   [MEMORY_BOUND] Operation has low Arithmetic Intensity (0.18 FLOPs/Byte vs GPU ridge point 54.8 FLOPs/Byte).

2. Which operations are responsible?
   RMSNorm + Residual Add (Unfused) accounts for 34.5% of memory-bound execution time.

3. What optimization helps?
   Selected RMSNorm + Residual — Fused Warp Shuffle, fusing intermediate memory round-trips into registers
   and using __shfl_xor_sync warp reductions to achieve a 2.36× measured speedup (4.784 ms → 2.031 ms).

4. Is that optimization numerically safe?
   VERIFIED SAFE: Passed compute-sanitizer memcheck (0 illegal accesses/races), standard batch numerical parity,
   and odd unaligned dimension (hidden_dim=1027) edge-case verification with max absolute error 2.00e-07.

5. How much energy does it save?
   Reduces GPU board energy by 58.2% (40.26 J → 16.82 J per 1K inferences), improves Energy-Delay Product (EDP)
   by 5.64×, and saves 41.3 kWh/month (15.9 kg CO2e/month including PUE).

6. Can fewer/cheaper GPUs serve the workload?
   YES: Single-GPU capacity rises from 209 to 492 req/s (752,493 → 1,772,700 Requests/GPU-hr).
   Fleet requirement drops from 2 to 1 GPU(s).

7. How much money does the business save monthly?
   Cuts unit cost by 57.5% ($2.0039 → $0.8506 per 1M inferences), eliminating 1 GPU instance
   and saving $1,100.79/month ($13,209.48/year).
================================================================================
```
