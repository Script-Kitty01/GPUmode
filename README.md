# gpuMode

[![Go Tests, Static Analysis & eBPF CI](https://github.com/Script-Kitty01/GPUmode/actions/workflows/ci.yml/badge.svg)](https://github.com/Script-Kitty01/GPUmode/actions/workflows/ci.yml)
[![CUDA Kernel Compilation & Hardware Runner CI](https://github.com/Script-Kitty01/GPUmode/actions/workflows/cuda-hardware.yml/badge.svg)](https://github.com/Script-Kitty01/GPUmode/actions/workflows/cuda-hardware.yml)
[![Linux Distribution Installation & Compatibility Matrix](https://github.com/Script-Kitty01/GPUmode/actions/workflows/linux-compat.yml/badge.svg)](https://github.com/Script-Kitty01/GPUmode/actions/workflows/linux-compat.yml)

> **Real-time CUDA hardware optimization, `10 ms` NVML + rootless eBPF observability, and local LLM efficiency platform for Linux & NVIDIA GPUs.**

**gpuMode** closes the loop between **low-level CUDA physics** and **end-to-end AI workload efficiency** — diagnosing GPU bottlenecks, compiling and verifying fused CUDA C++ kernels on your live hardware, integrating real GPU board energy ($E = \int P(t)\,dt$), and benchmarking local `.gguf` / Ollama models **Before vs. After** side-by-side.

```text
Observe (NVML + eBPF) → Diagnose (Roofline) → Optimize (CUDA + Bayesian) → Verify (Sanitizer + Parity) → Measure Energy → Right-Size (Pareto)
```

---

## Key Capabilities

- **Zero-Jitter Hardware Telemetry:** Pure-Go `libnvidia-ml.so.1` (`NVML`) bindings polling GPU power, clocks, throttle reasons, PCIe throughput, utilization, VRAM, and compute PIDs at **`10 ms`**, plus rootless **eBPF** uprobes on `libcuda.so.1` (with launch-geometry extraction, log2 latency histograms, and Perfetto JSON timeline export) & Linux `sched_switch`.
- **Automated CUDA Kernel Tuning & Safety Gate:** Tunes fused CUDA C++ kernels (`RMSNorm + Residual`, `SwiGLU`, `Online Softmax`, `LayerNorm`, `Fused RoPE + QKV`, `Quantized Q8_0 Decode GEMV`, `GQA FlashDecoding Split-K`) via Bayesian UCB or Grid Search with self-healing repair, gated by `compute-sanitizer`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, and odd-shape (`hidden_dim = 1027`) FP64 parity.
- **Local AI Model Runner (Direct Path or Auto-Scan):** Run any `.gguf` path directly from the Web Dashboard (or auto-detect models from `~/AI/Models`, Hugging Face cache, `llama-server`, and `Ollama`) with multi-client concurrency (`1..32`) to compare **Without Project (Baseline)** vs. **With Project (gpuMode)** throughput, VRAM, Watts/Joules, **`$ / 1M tokens` (On-Demand & Spot)**, and eBPF overhead.
- **Production GPU Fleet Cost, VRAM Placement & Admission Control (`internal/scheduler`):** Evaluates On-Demand vs. Spot/Preemptible GPU pools (`Local GPU`, `L4 24GB`, `A10G 24GB`, `L40S 48GB`) for full model + KV-cache VRAM fit, enforces priority admission control (`CRITICAL_REALTIME`, `STANDARD_INTERACTIVE`, `BATCH_SPOT_INTERRUPTIBLE` with `HTTP 429` backpressure), and simulates fleet scheduling strategies (`ROUND_ROBIN`, `LEAST_LOADED_VRAM_AWARE`, `COST_SLO_WEIGHTED_PARETO`).
- **100% Runtime Hardware Detection & Live PyTorch C++/CUDA Extensions:** Probes live GPU specs and empirical DRAM/L2 ceilings (`cudaGetDeviceProperties` + `float4` STREAM Triad + `nvidia-smi`) with zero hardcoded GPU fallbacks, profiles live PyTorch operators (`torch.profiler`), and generates verified drop-in `torch.utils.cpp_extension` C++/CUDA modules (`2.46×` measured 64-layer speedup in PyTorch `2.15.0+cu132`).

---

## Independent Accuracy & Cross-Verification (`gpuMode` vs. `nsys`, `llama-bench`, `nvidia-smi`, & `PyTorch`)

All `gpuMode` measurements on **NVIDIA GeForce RTX 4050 Laptop GPU (`sm_89`)** were cross-checked directly against **NVIDIA Nsight Systems (`nsys` CUPTI SQLite timestamps)**, **`/usr/bin/llama-bench`**, **`/usr/bin/nvidia-smi`**, **`compute-sanitizer`**, and **PyTorch `2.15.0+cu132` (`torch.cuda.Event`)**:

### 1. Multi-Model Local LLM Accuracy (`Qwen3.5-4B` vs. `DeepSeek-V4-Pro-9B`)

| Local `.gguf` Model | `gpuMode` Optimized TPOT | `llama-bench` Optimized TPOT | TPOT Delta (`Δ`) / Error | Throughput (`tok/s`) Error | GPU Power (`W`) Mean Error (`NVML` vs `nvidia-smi`) | **Overall Model Mean Error (MAPE)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.5-4B-Q4_K_M` (`2.82 GiB`, `ngl=24 → 99`)** | `21.06 ms/tok` (`47.48 tok/s`) | `20.35 ms/tok` (`49.13 tok/s`) | `+0.71 ms/tok` (**`3.48%`**) | **`2.38%`** mean (`97.62%` acc) | **`4.78%`** (`95.22%` acc) | **`4.49%` (`95.51%` accuracy)** |
| **`DeepSeek-V4-Pro-9B-Q4_K_M` (`5.38 GiB`, `ngl=20 → 28`)** | `50.64 ms/tok` (`19.75 tok/s`) | `48.92 ms/tok` (`20.44 tok/s`) | `+1.73 ms/tok` (**`3.53%`**) | **`5.98%`** mean (`94.02%` acc) | **`2.78%`** (`97.22%` acc) | **`5.53%` (`94.47%` accuracy)** |
| **Combined 2-Model Mean (18 Measurements)** | — | — | **`3.51%`** Opt TPOT err (`0.05%` diff between models) | **`4.18%`** (`95.82%` acc) | **`3.78%`** (`96.22%` acc) | **`5.01%` (`94.99%` accuracy)** |

### 2. CUDA Kernel Hardware Timing & PyTorch Extension Accuracy (`gpuMode` vs. `nsys` CUPTI & `PyTorch 2.15`)

| Benchmark Verification Suite | `gpuMode` Reading | External Tool (`nsys` CUPTI / `PyTorch` / `compute-sanitizer`) | Delta (`Δ`) | **Measured Accuracy** |
| :--- | :---: | :---: | :---: | :---: |
| **`rmsnorm_residual` Kernel (`Baseline` / `Optimized`)** | `3.2255 ms` / `1.5327 ms` | `3.2234 ms` / `1.5306 ms` (`nsys` `CUPTI_ACTIVITY_KIND_KERNEL` `p50`) | `+2.1 µs` / `+2.1 µs` | **`99.93%` / `99.86%`** |
| **`rope_qkv` Kernel (`Baseline` / `Optimized`)** | `2.7954 ms` / `0.7426 ms` | `2.7915 ms` / `0.7405 ms` (`nsys` `CUPTI_ACTIVITY_KIND_KERNEL` `p50`) | `+3.9 µs` / `+2.1 µs` | **`99.86%` / `99.71%`** |
| **`swiglu` & `softmax` Optimized Kernels** | `4.1040 ms` / `1.7950 ms` | `4.0909 ms` / `1.7915 ms` (`nsys` `CUPTI_ACTIVITY_KIND_KERNEL` `p50`) | `+13.1 µs` / `+3.5 µs` | **`99.68%` / `99.81%`** |
| **Sustained L2 Cache & DRAM Bandwidth Ceilings** | `596.01 GB/s` L2 / `151.15 GB/s` DRAM | `583.41 GB/s` L2-resident / `140.12 GB/s` `cudaMemcpy` D2D | `+12.6` / `+11.0 GB/s` | **`97.89%` / `92.13%`** |
| **Auto-Generated PyTorch C++/CUDA Extension (`64` layers)** | **`12.840 ms`** (`64` fused `float4` launches) | **`31.564 ms`** (`256` unfused `torch.nn` launches in `PyTorch 2.15.0+cu132`) | **`2.46×` speedup** | **`torch.allclose=True` (`maxErr=4.77e-07`)** |
| **Memory Safety (`compute-sanitizer --tool memcheck`)** | `sanitizerPassed = true` | Direct CLI `compute-sanitizer --tool memcheck` (`0 errors`) | `0 errors` | **`100.00%` Exact Match** |

---

## Quick Start

```zsh
# 1. Clone and enter the repository
git clone https://github.com/Script-Kitty01/GPUmode.git
cd GPUmode

# 2. (Optional, one-time) Enable rootless Linux eBPF tracing (libcuda.so.1 uprobes + sched_switch)
./run.sh setcap

# 3. (Optional) Create local .venv with PyTorch for live torch.profiler & load_inline CUDA extension verification
python3 -m venv .venv && ./.venv/bin/pip install torch

# 4. Build (if needed) and launch the live Web Dashboard at http://127.0.0.1:8080
./run.sh
```

### Common CLI Commands

```zsh
# Install 'gpumode' globally to ~/.local/bin/gpumode so it works from any terminal directory
./run.sh install

# Live refreshing terminal GPU & Active AI Model TUI monitor (0 MiB CUDA VRAM overhead — works while models run!)
gpumode top

# Instant rich terminal GPU & Active Model stats card (or single-line snapshot)
gpumode stats
./run.sh monitor

# Closed-loop CUDA kernel autotuning (rmsnorm_residual, rmsnorm, layernorm, softmax, swiglu, rope_qkv, qgemv_decode, gqa_attention)
./run.sh optimize -op rope_qkv -batch 256 -hidden 4096 -iters 50

# CI/CD hardware regression gate (verifies speedup >= threshold, 0 PTXAS register spills, compute-sanitizer, and 1027 FP64 parity)
./run.sh check -op rope_qkv -batch 64 -hidden 2048 -iters 20 -min-speedup 1.15

# End-to-end streaming LLM benchmark (TTFT, TPOT, tok/s, VRAM, Watts/Joules, $/1M tokens, eBPF)
./run.sh workload -deepseek-model ~/AI/Models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf -llm-reqs 4 -llm-concurrency 2

# Production GPU fleet cost, VRAM-aware pool placement, priority admission control & scheduling simulator
./run.sh schedule -deepseek-model ~/AI/Models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf -llm-model Qwen3.5-4B-Q4_K_M

# Run full unit, mocked-NVML (.so dlopen), failure-path, and live CUDA test suite
./run.sh test
```

Requires Linux (`x86_64`), an NVIDIA GPU with drivers (`nvidia-smi`), CUDA Toolkit (`nvcc` & `compute-sanitizer`), Go (`1.24+`), Python 3, and Zsh.

---

## Automated CI/CD & Multi-Distro Verification

| GitHub Actions Workflow | Target Environment | What It Verifies on Every Push |
| :--- | :--- | :--- |
| **[`ci.yml`](./.github/workflows/ci.yml)** | `ubuntu-latest` (`Go` + `Clang/BPF` + `GCC`) | `gofmt`, `go vet ./...`, eBPF C bytecode regeneration (`go generate ./internal/ebpftrace/...`), mocked `libnvidia-ml.so.1` C-ABI `dlopen` tests, failure-path tests, and `go test -v -race ./...` |
| **[`cuda-hardware.yml`](./.github/workflows/cuda-hardware.yml)** | `nvidia/cuda:12.6.3-devel-ubuntu24.04` & `[self-hosted, linux, cuda]` | Compiles all 8 CUDA operations (`24` kernels) with `nvcc -O3 -arch=sm_89 -Xptxas -v` asserting `0` register spills in cloud CI, and runs `./gpumode check` on physical NVIDIA GPU silicon when a self-hosted runner is attached |
| **[`linux-compat.yml`](./.github/workflows/linux-compat.yml)** | **Ubuntu 22.04**, **Ubuntu 24.04**, **Debian 12**, **Fedora 41**, **Arch Linux** | Verifies OS package installation, CGo/`dlopen` linker compatibility, modular `frontend/` embedding, and full unit/mocked-NVML test suites across 5 Linux distributions |

---

## Documentation

| Guide | What's Inside |
| :--- | :--- |
| **[Startup & Web Dashboard Guide](./docs/STARTUP_GUIDE.md)** | Step-by-step setup, using all 3 Web Dashboard sections (CUDA Optimizer, Local Model Runner, Fleet Cost & Scheduler), custom ports, multi-GPU, and environment variables |
| **[`run.sh` Modes & Startup Models](./docs/RUN_SH_MODES.md)** | Complete reference for every `./run.sh` command (`dashboard`, `optimize`, `deepseek`, `workload`, `schedule`, `check`, `monitor`, `ebpf`, `setcap`, `bpf-gen`, `test`) and model flags |
| **[Architecture & Internals](./docs/ARCHITECTURE.md)** | System diagram, NVML + eBPF internals, Roofline/Bayesian/Pareto/Scheduler pipeline, modular `frontend/` components, and repository directory map |
| **[Dependencies & Libraries](./docs/DEPENDENCIES.md)** | Required & optional system packages, `.venv` PyTorch `2.15` setup, Go modules (`go.mod`), shared libraries (`libnvidia-ml.so.1`, `libcuda.so.1`), and Linux install commands |
| **[Verification & Benchmark Proof](./docs/VERIFICATION_AND_PROOF.md)** | Live RTX 4050 (`sm_89`) benchmark results (`2.06×` tok/s, `-21.3%` J/tok), 2-model MAPE (`5.01%`), and cross-checks against `nsys`, `llama-bench`, `nvidia-smi`, and `PyTorch 2.15` |
| **[Engineering Roadmap & Next-Gen Specs](./docs/ROADMAP.md)** | Technical specifications for eBPF launch-geometry & Perfetto tracing, live PyTorch `load_inline` verification, empirical Roofline & SM occupancy, `FP16`/`RoPE`/`QGEMV` kernels, and Bayesian server autotuning |
