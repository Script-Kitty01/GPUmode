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

## Core Subsystems

### 1. Direct NVML Telemetry (`internal/nvml`, `internal/telemetry`)
- Loads `libnvidia-ml.so.1` at runtime in pure Go (`dlopen`, zero CGo requirement).
- Samples board power, SM/memory clocks, GPU/memory utilization, VRAM, and active compute PIDs at **`10 ms` burst intervals** during benchmark windows (with automatic `50 ms` `nvidia-smi` fallback).
- Integrates live power via the trapezoidal rule $E = \int_{t_0}^{t_1} P(t)\,dt$ across interleaved execution windows to report `Joules / 1K inferences`, active power above idle, Energy-Delay Product ($\text{EDP}$), and measurement uncertainty.

### 2. Host ↔ CUDA eBPF Tracer (`internal/ebpftrace`)
- Attaches Linux uprobes/uretprobes to `libcuda.so.1` (`cuLaunchKernel*`, `cuMemcpyHtoD*`, `cuMemcpyDtoH*`, `cuMemAlloc*`, `cuStreamSynchronize*`, `cuCtxSynchronize*`, with `libcudart.so` fallback) plus the kernel `sched/sched_switch` raw tracepoint.
- Measures kernel launch rates (`launches/tok`), inter-launch CPU dispatch gaps (`µs`), PCIe `H2D/D2H` byte volumes, host synchronization stalls (`ms`), and scheduler off-CPU starvation (`ms`) across the target process tree.

### 3. Roofline & CUDA Autotuner (`internal/roofline`, `internal/optimizer`, `internal/harness`)
- Probes actual GPU hardware limits (`cudaGetDeviceProperties` + `nvidia-smi`) with zero fabricated fallbacks.
- Classifies operators as `MEMORY_BOUND`, `COMPUTE_BOUND`, or `LAUNCH_BOUND` against the GPU's hardware ridge point.
- Compiles candidate CUDA C++ kernels (`float1`/`float4` vectorization, kernel fusion, `__shfl_xor_sync` warp shuffles, block sizes) in parallel via `nvcc -Xptxas -v` and serializes GPU execution with dynamic $2\times$ L2 cache flushing.
- Enforces `compute-sanitizer --tool memcheck` and FP64 reference parity (including unaligned `hidden_dim = 1027`) before accepting any kernel.

---

## Repository Map

```text
gpuMode/
├── cmd/gpumode/main.go        # CLI & Real-Time HTTP/SSE Daemon entrypoint
├── internal/
│   ├── config/                # Dynamic CUDA binary, NVML/eBPF, model, and SLA config
│   ├── nvml/                  # Pure-Go libnvidia-ml.so.1 bindings (10ms power/clock/PIDs)
│   ├── models/                # Multi-source model discovery & direct .gguf path resolver
│   ├── ebpftrace/             # Kernel BPF C program + libcuda uprobes & sched_switch tracer
│   ├── physics/               # Live cudaGetDeviceProperties probe & capacity/energy KPIs
│   ├── telemetry/             # Concurrent NVML/SMI sampler, SSE broadcaster, & E = ∫P(t)dt
│   ├── roofline/              # Arithmetic Intensity, ridge points, & bottleneck classifier
│   ├── harness/               # Parallel nvcc -Xptxas -v worker pool & GPU verification runner
│   ├── optimizer/             # Parameterized CUDA C++ catalog, Bayesian autotuner, GGUF proposer
│   ├── pareto/                # Multi-objective non-dominated Pareto solver
│   ├── profiler/              # End-to-end causal pipeline & PyTorch C++ extension generator
│   ├── store/                 # Thread-safe local JSON persistence (.gpumode_data)
│   ├── server/                # REST API (/api/state, /api/models, /api/optimize, /api/workload) & UI
│   └── workload/              # Streaming llama-server / Ollama benchmark (Baseline vs Tuned)
├── scripts/profile_pytorch.py # PyTorch operator profiler bridge with standalone fallback
├── docs/                      # Detailed startup, mode, architecture, dependency, and proof docs
└── run.sh                     # Incremental build, setcap, & multi-mode launcher
```
