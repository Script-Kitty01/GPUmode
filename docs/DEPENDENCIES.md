# Requirements & Setup

[← Back to README](../README.md) • [Startup Guide](./STARTUP_GUIDE.md) • [Real Results & Proof](./VERIFICATION_AND_PROOF.md)

> **Light by design — no Node.js, no heavy install, just your NVIDIA tools and Go.**

**gpuMode** compiles into one small program with the entire Web Dashboard built right inside it. You do not need Node.js, NPM, or Docker to run it.

---

## 1. What You Need (Required)

| Requirement | Why It's Needed |
| :--- | :--- |
| **Linux (`x86_64`)** | Supports live GPU monitoring and optional deep Linux tracking |
| **NVIDIA Graphics Card & Driver** | Provides live GPU stats (`nvidia-smi`) and lets GPU code run |
| **NVIDIA CUDA Toolkit** | Compiles GPU speed-ups (`nvcc`) and checks them for memory bugs (`compute-sanitizer`) |
| **Go** | Builds the `gpumode` app |
| **Python 3** | Runs the optional PyTorch helper script (`scripts/profile_pytorch.py`) |
| **Zsh** | Runs the `./run.sh` helper script |

---

## 2. Optional Extras

You can use gpuMode without these, and add any of them later if you want their features:

| Tool | What It Unlocks |
| :--- | :--- |
| **`llama-server` & `llama-cli`** | Lets you run **Before vs. After** benchmarks on local `.gguf` AI models |
| **`ollama`** | Lets you benchmark models you already have in Ollama |
| **PyTorch (with CUDA)** | Enables live PyTorch speed-up verification inside `.venv` |
| **`libcap` (`setcap` / `getcap`)** | Lets you enable deeper Linux GPU tracking once so the app never needs root access |
| **`nvtop` or NVIDIA Nsight (`nsys`)** | Lets you double-check gpuMode's numbers with independent tools |
| **`clang` & `llvm`** | Only needed if you edit the low-level Linux tracker source code (`./run.sh bpf-gen`) |

---

## 3. Built-In Libraries (Small Footprint)

- **Go packages (`go.mod`):** Only two lightweight packages are used — `github.com/cilium/ebpf` (for Linux GPU tracking) and `golang.org/x/sys` (for Linux system calls).
- **NVIDIA driver libraries:** Loaded automatically from your system when gpuMode starts — no extra driver packages to install.
- **Web Dashboard:** Embedded directly inside the `gpumode` binary — zero frontend build tools required.

---

## 4. Copy-Paste Install Commands

### Ubuntu / Debian
```zsh
sudo apt update
sudo apt install golang-go nvidia-cuda-toolkit python3 zsh libcap2-bin nvtop
```

### Arch Linux / CachyOS
```zsh
# Core tools
sudo pacman -S go cuda python zsh libcap

# Optional: independent GPU monitors
sudo pacman -S nvtop nsight-systems
```

### Fedora
```zsh
sudo dnf install golang python3 zsh libcap
```
*(Install the NVIDIA driver and CUDA Toolkit from NVIDIA's Fedora repository if not already installed.)*

