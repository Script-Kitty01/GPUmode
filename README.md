# gpuMode

> **Real-time CUDA hardware optimization, `10 ms` NVML + rootless eBPF observability, and local LLM efficiency platform for Linux & NVIDIA GPUs.**

**gpuMode** closes the loop between **low-level CUDA physics** and **end-to-end AI workload efficiency** — diagnosing GPU bottlenecks, compiling and verifying fused CUDA C++ kernels on your live hardware, integrating real GPU board energy ($E = \int P(t)\,dt$), and benchmarking local `.gguf` / Ollama models **Before vs. After** side-by-side.

```text
Observe (NVML + eBPF) → Diagnose (Roofline) → Optimize (CUDA + Bayesian) → Verify (Sanitizer + Parity) → Measure Energy → Right-Size (Pareto)
```

---

## Key Capabilities

- **Zero-Jitter Hardware Telemetry:** Pure-Go `libnvidia-ml.so.1` (`NVML`) bindings polling GPU power, clocks, utilization, VRAM, and compute PIDs at **`10 ms`**, plus rootless **eBPF** uprobes on `libcuda.so.1` & Linux `sched_switch`.
- **Automated CUDA Kernel Tuning & Safety Gate:** Tunes fused CUDA C++ kernels (`RMSNorm + Residual`, `SwiGLU`, `Online Softmax`, `LayerNorm`) via Bayesian UCB or Grid Search, gated by `compute-sanitizer` and odd-shape (`hidden_dim = 1027`) FP64 parity.
- **Local AI Model Runner (Direct Path or Auto-Scan):** Run any `.gguf` path directly from the Web Dashboard (or auto-detect models from `~/AI/Models`, Hugging Face cache, `llama-server`, and `Ollama`) to compare **Without Project (Baseline)** vs. **With Project (gpuMode)** throughput, VRAM, Watts/Joules, and eBPF overhead.
- **100% Runtime Hardware Detection:** Probes live GPU specs (`cudaGetDeviceProperties` + `nvidia-smi`) with zero hardcoded GPU fallbacks and generates drop-in PyTorch C++/CUDA extensions.

---

## Quick Start

```zsh
# Optional (one-time): enable rootless Linux eBPF tracing
./run.sh setcap

# Build (if needed) and launch the live Web Dashboard at http://127.0.0.1:8080
./run.sh
```

Requires Linux, an NVIDIA GPU with drivers (`nvidia-smi`), CUDA Toolkit (`nvcc` & `compute-sanitizer`), Go, Python 3, and Zsh.

---

## Documentation

| Guide | What's Inside |
| :--- | :--- |
| **[Startup & Web Dashboard Guide](./docs/STARTUP_GUIDE.md)** | Step-by-step setup, using the Web Dashboard, Direct Model Path runner, custom ports, multi-GPU, and environment variables |
| **[`run.sh` Modes & Startup Models](./docs/RUN_SH_MODES.md)** | Complete reference for every `./run.sh` command (`dashboard`, `optimize`, `deepseek`, `workload`, `monitor`, `ebpf`, `setcap`, `bpf-gen`, `test`) and model flags |
| **[Architecture & Internals](./docs/ARCHITECTURE.md)** | System diagram, NVML + eBPF internals, Roofline/Bayesian/Pareto pipeline, and repository directory map |
| **[Dependencies & Libraries](./docs/DEPENDENCIES.md)** | Required & optional system packages, Go modules (`go.mod`), shared libraries (`libnvidia-ml.so.1`, `libcuda.so.1`), and Linux install commands |
| **[Verification & Benchmark Proof](./docs/VERIFICATION_AND_PROOF.md)** | Live RTX 4050 (`sm_89`) benchmark results (`2.06×` tok/s, `-21.3%` J/tok) and independent cross-checks against `nsys`, `llama-bench`, `nvidia-smi`, and `nvtop` |
