# Dependencies & Libraries

[← Back to README](../README.md) • [Startup Guide](./STARTUP_GUIDE.md) • [Verification & Proof](./VERIFICATION_AND_PROOF.md)

**gpuMode** is designed with minimal external dependencies: a zero-CGo Go backend, an embedded zero-NPM Web Dashboard, and direct runtime discovery of system CUDA/NVML libraries.

---

## 1. Required System Dependencies

| Dependency | Purpose | Resolution |
| :--- | :--- | :--- |
| **Linux (`x86_64`)** | Host OS & eBPF uprobe/tracepoint support | Kernel `5.8+` recommended for ringbuf/BPF features |
| **NVIDIA Driver** | Provides `nvidia-smi`, `libnvidia-ml.so.1`, and `libcuda.so.1` | Loaded dynamically at runtime |
| **CUDA Toolkit (`nvcc` & `compute-sanitizer`)** | Compiles parameterized CUDA C++ kernels (`-Xptxas -v`) and runs `memcheck` verification | Auto-resolved via `GPUMODE_*`, `PATH`, `CUDA_HOME`, `/opt/cuda/bin`, `/usr/local/cuda/bin` |
| **Go Toolchain** | Builds the `gpumode` daemon and CLI (`go.mod` specifies Go `1.26` / `1.23+` compatible) | `go build -o ./gpumode ./cmd/gpumode` |
| **Python 3 (`python3`)** | Executes `scripts/profile_pytorch.py` | Falls back cleanly if `torch` is not installed |
| **Zsh (`zsh`)** | Executes `./run.sh` wrapper script | `/usr/bin/env zsh` |

---

## 2. Go Module Dependencies (`go.mod`)

`gpumode` uses only **two** external Go packages (no CGo toolchain or Node/NPM build step required):

| Module | Version | Role in `gpuMode` |
| :--- | :--- | :--- |
| `github.com/cilium/ebpf` | `v0.22.0` | Loads precompiled eBPF bytecode (`cudatrace_x86_bpfel.o`) and attaches `libcuda.so.1` uprobes/uretprobes and `sched_switch` tracepoints |
| `golang.org/x/sys` | `v0.50.0` | Low-level Linux syscalls and capability/process inspection |

---

## 3. Dynamically Loaded Shared Libraries (Runtime `dlopen` / Uprobes)

| Shared Library | Package | Role |
| :--- | :--- | :--- |
| `libnvidia-ml.so.1` | `internal/nvml` | Direct `10 ms` GPU power, SM/memory clocks, utilization, VRAM, and compute PID polling without spawning subprocesses |
| `libcuda.so.1` | `internal/ebpftrace` | Primary target for eBPF uprobes (`cuLaunchKernel*`, `cuMemcpyHtoD*`, `cuMemcpyDtoH*`, `cuMemAlloc*`, `cuStreamSynchronize*`, `cuCtxSynchronize*`) |
| `libcudart.so` | `internal/ebpftrace` | Fallback CUDA Runtime symbol target when Driver API symbols are not directly invoked |

---

## 4. Optional Runtime & Verification Tools

| Tool | Required For | Notes |
| :--- | :--- | :--- |
| **`setcap` / `getcap` (`libcap`)** | Rootless eBPF (`./run.sh setcap`) | Grants `cap_bpf,cap_perfmon,cap_sys_resource,cap_sys_admin+ep` to `./gpumode` |
| **`llama-server` & `llama-cli` (`llama.cpp` CUDA)** | Local `.gguf` Before/After Runner & `deepseek` Proposer | Used to spawn Baseline vs. gpuMode Tuned `.gguf` inference benchmarks |
| **`ollama`** | Benchmarking local Ollama models | Auto-detected via `http://127.0.0.1:11434/api/tags` |
| **PyTorch (`torch` with CUDA)** | Live `torch.profiler` operator attribution | Optional; `scripts/profile_pytorch.py` uses analytical fallback when absent |
| **`clang` + `llvm`** | `./run.sh bpf-gen` only | Only needed if modifying `internal/ebpftrace/bpf/cuda_trace.bpf.c` |
| **`nvtop`, `nsight-systems` (`nsys`), `llama-bench`** | Independent external verification | Used to cross-check Watts, SM %, tok/s, and CUDA launch counts |

---

## 5. Installing Packages on Linux

### Arch Linux / CachyOS (`pacman`)
```zsh
# Core toolchain & rootless capability helpers
sudo pacman -S go cuda python zsh libcap

# Optional: local GGUF inference & independent hardware profilers
sudo pacman -S nvtop nsight-systems
```

### Ubuntu / Debian (`apt`)
```zsh
sudo apt update
sudo apt install golang-go nvidia-cuda-toolkit python3 zsh libcap2-bin nvtop
```
