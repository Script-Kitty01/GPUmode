# Startup & Web Dashboard Guide

[← Back to README](../README.md) • [All `run.sh` Modes & Model Flags](./RUN_SH_MODES.md) • [Dependencies](./DEPENDENCIES.md)

---

## 1. Quick First-Time Setup

```zsh
# 1. Clone and enter the repository
git clone https://github.com/Script-Kitty01/GPUmode.git
cd GPUmode

# 2. (Optional, one-time) Grant rootless Linux eBPF capabilities to ./gpumode
#    Prompts once via sudo so the server never needs to run as root
./run.sh setcap

# 3. Build (if needed) and launch the Web Dashboard
./run.sh
```

Open **`http://127.0.0.1:8080`** in your browser.

### What `./run.sh` Does on Launch
1. Performs an incremental build check (`./gpumode` is rebuilt only when Go or eBPF files change, preserving `setcap` capabilities).
2. Verifies rootless eBPF capabilities (`cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep`), falling back gracefully to `-ebpf=auto` if unprivileged.
3. Runs `scripts/profile_pytorch.py` and pre-warms an initial hardware-verified CUDA report if `.gpumode_data` is empty.
4. Starts the HTTP + Server-Sent Events (SSE) Web Dashboard at `http://127.0.0.1:8080`.

---

## 2. Using the Web Dashboard (`http://127.0.0.1:8080`)

### Section 1 — CUDA Kernel Autotuner, Roofline & Pareto Explorer
1. Choose a target operator (`rmsnorm_residual`, `swiglu`, `softmax`, `layernorm`, or `rmsnorm`), tuning strategy (`BAYESIAN_UCB` or `GRID_SEARCH`), batch size, and hidden dimension.
2. Click **Compile, Verify & Benchmark on GPU**.
3. Inspect the **6-Question Causal Breakdown**, live **Power Sparkline**, **Host ↔ CUDA eBPF** metrics, **Roofline Chart**, **Pareto Frontier**, verified **CUDA C++ kernel**, and auto-generated **PyTorch C++/CUDA extension**.

### Section 2 — Local AI Model Runner (Baseline vs. gpuMode Tuned)
1. **Option A — Direct Model Path (no `/models` folder needed):**
   - Paste any `.gguf` file path (e.g. `~/Downloads/Qwen3.5-4B-Q4_K_M.gguf` or `/abs/path/to/model.gguf`) or model directory into **Direct Model Path**.
   - The **Model Source** dropdown automatically switches to `Direct Local Model Path`.
2. **Option B — Auto-Detected Models:**
   - Select any model auto-discovered from `~/AI/Models`, `~/models`, `./models`, `~/.cache/huggingface/hub`, `llama-server` (`:8081`), or `Ollama` (`:11434`), or specify an optional folder and click **Scan Models**.
3. Click **Run Without Project (Baseline)** to record baseline throughput, TTFT/TPOT, GPU SM %, VRAM, Watts/Joules, and eBPF launch/sync overhead.
4. Click **Run With Project (gpuMode)** to apply VRAM-fitted GPU tuning (`-ngl`, FlashAttention, quantized KV cache, tuned batching) **plus** local-model-guided CUDA kernel optimization, then compare **Before vs. After** side-by-side.

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
