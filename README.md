# gpuMode — Intelligent GPU Infrastructure & CUDA-to-Economics Optimization Platform

> **Don't just make GPU workloads faster. Accomplish the same amount of AI computation using fewer GPUs, less electricity, and less money.**

**gpuMode** is a real-time Go + CUDA systems engineering platform that bridges **low-level CUDA hardware physics** directly to **infrastructure & energy economics** in one closed-loop pipeline:

```text
Observe → Diagnose (Roofline) → Optimize (CUDA + Bayesian) → Verify (Sanitizer + Parity) → Measure Energy → Multi-Objective Pareto → Calculate Savings
```

Unlike standard GPU dashboards that merely report utilization percentages, **gpuMode** compiles, sanitizes, verifies, and benchmarks parameterized CUDA C++ kernels on physical NVIDIA hardware while integrating live GPU power draw $E = \int_{t_0}^{t_1} P(t)\,dt$ to answer seven causal engineering and financial questions.

---

## Headline Business & Systems KPIs

Every optimization and right-sizing run in **gpuMode** is evaluated against three primary efficiency metrics:

| Headline Metric | What It Proves | How gpuMode Computes It |
| :--- | :--- | :--- |
| **`$ / 1M Inferences`** | Financial Efficiency | Combines amortized/cloud GPU hourly cost (`$/GPU-hr`) + PUE-adjusted server electricity cost (`$/kWh`) per 1,000,000 completed requests. |
| **`Joules / 1K Inferences`** | Energy Efficiency | Trapezoidal integration of live GPU board power $E = \int_{t_0}^{t_1} P(t)\,dt$ across CUDA-event timed execution windows, plus Energy-Delay Product ($\text{EDP} = E \times T$) and Performance/Watt. |
| **`Requests / GPU-Hour`** | Hardware Capacity Efficiency | Verified sustained single-GPU throughput capacity ($3600 \times \text{RPS}$) within the target p95 latency SLA. |

---

## The 7-Question CUDA-to-Economics Causal Chain

For every workload, **gpuMode** generates a hardware-verified `CausalChainReport` answering:

1. **Why is the GPU underutilized?**
   Uses the **Roofline Performance Model** ($\text{Arithmetic Intensity} = \text{FLOPs}/\text{Bytes}$ vs. the GPU's hardware ridge point) to classify bottlenecks as `MEMORY_BOUND`, `COMPUTE_BOUND`, or `LAUNCH_BOUND`.
2. **Which operations are responsible?**
   Attributes latency share, DRAM memory bandwidth consumption, kernel launch overhead, and peak activation VRAM across hotspot operators (`RMSNorm + Residual`, `SwiGLU`, `Online Softmax`, `LayerNorm`).
3. **What CUDA optimization helps?**
   Selects and tunes targeted CUDA C++ transformations: kernel fusion (eliminating intermediate global memory round-trips), 128-bit `float4` vectorized memory coalescing, and register-level `__shfl_xor_sync` warp reductions.
4. **Is that optimization numerically safe?**
   Gates every candidate kernel through `compute-sanitizer --tool memcheck`, standard batch numerical parity (`atol/rtol` vs. FP64 CPU reference), and odd unaligned edge-case dimensions (`hidden_dim = 1027`, non-divisible by 4).
5. **How much energy does it save?**
   Integrates live GPU power sensors during execution to report board `Joules / 1K inferences`, PUE-adjusted server `kWh/month` saved, Energy-Delay Product ($\text{EDP}$) improvement, and grid carbon reduction ($\text{kg CO}_2\text{e/month}$).
6. **Can fewer or cheaper GPUs now serve the workload?**
   Combines **Multi-Objective Pareto Optimization**, **Digital Twin SKU Simulation**, and **Multi-Workload Bin-Packing** to determine whether throughput gains or VRAM activation savings unlock fewer instances or a cheaper hardware tier.
7. **How much money does the business save monthly?**
   Quantifies monthly and annual savings (`$/month`, `$/year`) across GPU infrastructure and datacenter electricity under user-configurable tariffs and SLAs.

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
                           - Parameterized CUDA C++ Kernels (sm_89)
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
                           - Interleaved CUDA Events + 32 MiB L2 Flush + Power Sync
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

# Print a live GPU hardware & power telemetry snapshot
./run.sh monitor

# Run the complete unit & live GPU hardware verification test suite
./run.sh test
```

---

## Example Verified Output (NVIDIA GeForce RTX 4050 Laptop GPU / `sm_89`)

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
