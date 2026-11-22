# Architecture & Internals

[← Back to README](../README.md) • [Startup Guide](./STARTUP_GUIDE.md) • [Verification & Proof](./VERIFICATION_AND_PROOF.md)

---

## Closed-Loop Pipeline

```text
Observe (NVML 10ms + eBPF)
   │
   ▼
Diagnose (Roofline Arithmetic Intensity + Host Launch Gap)
   │
   ▼
Optimize (Parameterized CUDA C++ + Bayesian UCB / Local GGUF Proposer)
   │
   ▼
Verify (compute-sanitizer memcheck + Standard & Odd-Shape FP64 Parity)
   │
   ▼
Measure Energy (Trapezoidal Integration E = ∫ P(t) dt + EDP)
   │
   ▼
Right-Size (Multi-Objective Pareto Frontier + Fleet Capacity)
```

---

## System Diagram

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
                           - 8-Op Kernel Catalog (RMSNorm+Res, RMSNorm, LayerNorm, Softmax, SwiGLU, RoPE, QGEMV, GQA)
                           - Parallel nvcc Worker Pool + Serialized GPU Benchmark Queue
                           - Bayesian UCB + Grid Search Autotuner J(θ) = αT + βE + γM
                           - Pluggable KernelProposer (Deterministic + Local GGUF w/ Self-Healing Repair)
                                          │
                                          ▼
                           5. HARDWARE VERIFICATION ENGINE
                           - nvcc -Xptxas -v (Registers, Shared Mem, Spills) + CUDA Occupancy API
                           - compute-sanitizer --tool memcheck
                           - Standard + Odd-Dimension (1027) FP64 Numerical Parity
                           - Live PyTorch 2.15 load_inline Extension Verification + 2x L2 Flush
                                          │
                                          ▼
                   6. PARETO, FLEET SCHEDULER & MODULAR WEB DASHBOARD
                      - Non-Dominated Pareto Frontier (Latency, Energy, VRAM) + $/1M Token Economics
                      - VRAM-Aware Pool Placement, Priority Admission Control (HTTP 429) & Strategy Simulator
                      - Modular frontend/ (components/*.html, css/*.css, js/*.js) + Prometheus /metrics & Perfetto
```

---

## Core Subsystems

### 1. Direct NVML Telemetry (`internal/nvml`, `internal/telemetry`)
- Loads `libnvidia-ml.so.1` at runtime in pure Go (`dlopen`, zero CGo requirement).
- Samples board power, SM/memory clocks, GPU/memory utilization, VRAM, and active compute PIDs at **`10 ms` burst intervals** during benchmark windows (with automatic `50 ms` `nvidia-smi` fallback).
- Integrates live power via the trapezoidal rule $E = \int_{t_0}^{t_1} P(t)\,dt$ across interleaved execution windows to report `Joules / 1K inferences`, active power above idle, Energy-Delay Product ($\text{EDP}$), and measurement uncertainty.

### 2. Host ↔ CUDA eBPF Tracer (`internal/ebpftrace`)
- Attaches Linux uprobes/uretprobes to `libcuda.so.1` (`cuLaunchKernel*`, `cuMemcpyHtoD*`, `cuMemcpyDtoH*`, `cuMemAlloc*`, `cuStreamSynchronize*`, `cuCtxSynchronize*`, with `libcudart.so` fallback) plus the kernel `sched/sched_switch` raw tracepoint.
- Measures kernel launch rates (`launches/tok`), inter-launch CPU dispatch gaps (`µs`), PCIe `H2D/D2H` byte volumes, host synchronization stalls (`ms`), and scheduler off-CPU starvation (`ms`) across the target process tree.

### 3. Roofline & CUDA Autotuner (`internal/roofline`, `internal/optimizer`, `internal/harness`)
- Probes actual GPU hardware limits and empirical sustained DRAM STREAM Triad & L2 cache bandwidth ceilings (`cudaGetDeviceProperties` + startup micro-probe + `nvidia-smi`) with zero fabricated fallbacks.
- Classifies operators as `MEMORY_BOUND`, `COMPUTE_BOUND`, or `LAUNCH_BOUND` against the GPU's hardware ridge point.
- Compiles candidate CUDA C++ kernels (`float1`/`float4` vectorization, kernel fusion, `__shfl_xor_sync` warp shuffles, block sizes across 8 LLM operations) in parallel via `nvcc -Xptxas -v`, queries `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for SM warp occupancy, and serializes GPU execution with dynamic $2\times$ L2 cache flushing.
- Enforces `compute-sanitizer --tool memcheck` and FP64 reference parity (including unaligned `hidden_dim = 1027`), with automated multi-turn self-healing repair (`RepairCandidateWithFeedback`).

### 4. Production Fleet Cost, Placement & Admission Control (`internal/scheduler`, `internal/physics`)
- Computes production unit economics (`CostPer1MTokensUSD` On-Demand & Spot, `CostPer1KRequestsUSD`, `EnergyCostPer1MReqsUSD` from live NVML Joules, and `AnnualFleetSavedUSD`).
- Evaluates model weight + KV-cache VRAM footprints across local and cloud GPU pools (`Local GPU`, `L4 24GB`, `A10G 24GB`, `L40S 48GB`), enforces priority admission control (`CRITICAL_REALTIME`, `STANDARD_INTERACTIVE`, `BATCH_SPOT_INTERRUPTIBLE` with `HTTP 429` backpressure), and simulates fleet scheduling strategies (`ROUND_ROBIN`, `LEAST_LOADED_VRAM_AWARE`, `COST_SLO_WEIGHTED_PARETO`).

### 5. Modular Frontend Architecture (`frontend/`)
- Replaces monolithic inline HTML strings with an embedded modular asset package (`frontend/embed.go`):
  - **7 HTML Component Partials (`frontend/components/`):** `header.html`, `telemetry_strip.html`, `ebpf_boundary.html`, `cuda_optimizer.html`, `model_runner.html`, `fleet_scheduler.html`, `code_panels.html`.
  - **Dedicated Stylesheet (`frontend/css/dashboard.css`)** and **6 Single-Responsibility JS Modules (`frontend/js/`):** `utils.js`, `telemetry.js`, `optimizer.js`, `workload.js`, `scheduler.js`, `app.js` served at `/static/`.

---

## Repository Map

```text
gpuMode/
├── cmd/gpumode/main.go        # CLI & Real-Time HTTP/SSE Daemon entrypoint
├── frontend/                  # Modular Web Dashboard UI (embedded via go:embed)
│   ├── embed.go               # Component assembler & /static/ asset handler
│   ├── css/dashboard.css      # Dashboard theme, layout grid, cards, tables & badges
│   ├── components/            # 7 modular HTML partials (header, telemetry, ebpf, optimizer, runner, scheduler, code)
│   └── js/                    # 6 single-responsibility JS modules (utils, telemetry, optimizer, workload, scheduler, app)
├── internal/
│   ├── config/                # Dynamic CUDA binary, .venv Python, NVML/eBPF, model, and SLA/cost config
│   ├── nvml/                  # Pure-Go libnvidia-ml.so.1 bindings (10ms power/clock/throttle/PCIe/PIDs)
│   ├── models/                # Multi-source model discovery & direct .gguf path resolver
│   ├── ebpftrace/             # Kernel BPF C program + libcuda uprobes, launch geometry, histograms & Perfetto export
│   ├── physics/               # Live cudaGetDeviceProperties + empirical DRAM/L2 probe & unit cost KPIs
│   ├── telemetry/             # Concurrent NVML/SMI sampler, SSE broadcaster, & E = ∫P(t)dt
│   ├── roofline/              # Arithmetic Intensity, hierarchical ceilings, & bottleneck classifier
│   ├── harness/               # Parallel nvcc -Xptxas -v worker pool, occupancy API & GPU verification runner
│   ├── optimizer/             # 8-op CUDA C++ catalog, Bayesian autotuner, GGUF proposer & self-healing repair
│   ├── pareto/                # Multi-objective non-dominated Pareto solver
│   ├── profiler/              # End-to-end causal pipeline & self-contained PyTorch C++/CUDA extension generator
│   ├── scheduler/             # VRAM-aware GPU pool placement, priority admission control & strategy simulator
│   ├── store/                 # Thread-safe local JSON persistence (.gpumode_data)
│   ├── server/                # REST API (/api/state, /api/models, /api/optimize, /api/workload, /api/schedule, /metrics, /api/trace/perfetto)
│   └── workload/              # Concurrent streaming llama-server / Ollama benchmark (Baseline vs Tuned)
├── scripts/profile_pytorch.py # PyTorch 2.15 torch.profiler & load_inline CUDA extension verifier
├── docs/                      # Detailed startup, mode, architecture, dependency, roadmap, and proof docs
└── run.sh                     # Incremental build, setcap, & multi-mode launcher
```
