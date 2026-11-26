# gpuMode

[![Go Tests, Static Analysis & eBPF CI](https://github.com/Script-Kitty01/GPUmode/actions/workflows/ci.yml/badge.svg)](https://github.com/Script-Kitty01/GPUmode/actions/workflows/ci.yml)
[![CUDA Kernel Compilation & Hardware Runner CI](https://github.com/Script-Kitty01/GPUmode/actions/workflows/cuda-hardware.yml/badge.svg)](https://github.com/Script-Kitty01/GPUmode/actions/workflows/cuda-hardware.yml)
[![Linux Distribution Installation & Compatibility Matrix](https://github.com/Script-Kitty01/GPUmode/actions/workflows/linux-compat.yml/badge.svg)](https://github.com/Script-Kitty01/GPUmode/actions/workflows/linux-compat.yml)

> **Run local AI faster, cooler, and cheaper on your NVIDIA graphics card.**

**gpuMode** is a live dashboard and command-line toolkit for Linux and NVIDIA graphics cards. It shows you why your GPU is slowing down, speeds up common AI workloads on your real hardware, checks every change for safety and accuracy, and compares your local AI models **Before vs. After** side by side.

```text
Watch Your GPU  →  Find What's Slow  →  Speed It Up & Verify  →  Compare Before vs. After
```

---

## Why Use gpuMode?

- **See your GPU clearly, without wasting memory**
  Watch live speed, GPU memory, temperature, and power draw in a clean web dashboard or right in your terminal (`gpumode top`) — using zero GPU memory of its own.
- **Make local AI models up to 2× faster**
  Point to any `.gguf` model file or Ollama model to compare default settings against tuned settings side by side, complete with speed, memory, energy, and cost numbers.
- **Speed up core AI steps automatically**
  Tests faster GPU code directly on your graphics card and only keeps improvements that pass strict memory-safety and accuracy checks.
- **Use less total electricity**
  By finishing each answer in roughly half the time, your graphics card spends far less time running hard — lowering total energy per answer.
- **Plan cloud costs with real measurements**
  Estimate how many graphics cards you actually need, which server sizes fit your model, and how much you can save before spending money in the cloud.

---

## Real Results on Real Hardware

Tested on a real **NVIDIA RTX 4050 Laptop GPU** running a local 4B AI model (`Qwen3.5-4B`) and double-checked against official NVIDIA tools (`nvidia-smi`, Nsight Systems, and `llama-bench`):

| What Changed | Before (Default) | After (With gpuMode) | Improvement |
| :--- | :---: | :---: | :---: |
| **Generation Speed** | 20.6 tokens/sec | **42.4 tokens/sec** | **2× faster** |
| **Total Wait Time (4 requests)** | 9.3 seconds | **4.5 seconds** | **51% faster** |
| **Wait for First Word** | 319 ms | **91 ms** | **3.5× faster** |
| **Total Energy Used (per 1,000 tokens)** | 1,909 Joules | **1,502 Joules** | **21% less energy** |
| **Data Shuffled Between CPU & GPU** | 47.5 MB | **9.3 MB** | **5× less traffic** |
| **PyTorch Speed-Up Check (64 layers)** | 31.6 ms | **12.8 ms** | **2.5× faster** |

All measurements match independent tools closely (**95%** agreement on full AI model runs, **99.8%** agreement on individual GPU timing checks, and **0** memory errors). See [Verification & Proof](./docs/VERIFICATION_AND_PROOF.md) for the full breakdown.

---

## Quick Start

> **Get a live GPU dashboard running in under two minutes.**

```zsh
# 1. Download the project
git clone https://github.com/Script-Kitty01/GPUmode.git
cd GPUmode

# 2. (Optional, one-time) Allow deeper Linux GPU tracking without running as root
./run.sh setcap

# 3. Start the live Web Dashboard at http://127.0.0.1:8080
./run.sh
```

Open **`http://127.0.0.1:8080`** in your browser to start monitoring, optimizing, and testing models.

*(Optional)* If you also want live PyTorch speed-up checks, create a local Python environment once:

```zsh
python3 -m venv .venv && ./.venv/bin/pip install torch
```

---

## Everyday Commands

> **Use the web dashboard, or run everything from your terminal.**

```zsh
# Install the 'gpumode' command globally so you can run it from any folder
./run.sh install

# Live updating terminal monitor (safe to run while AI models are active)
gpumode top

# Quick snapshot of GPU health and active AI models
gpumode stats

# Compare a local AI model Before vs. After tuning
./run.sh workload -deepseek-model ~/AI/Models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf

# Automatically tune and verify a core AI step on your GPU
./run.sh optimize -op rope_qkv

# Check cloud server fit, cost, and recommended GPU size for a model
./run.sh schedule -deepseek-model ~/AI/Models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf

# Run an automated speed and safety check (great for CI pipelines)
./run.sh check -op rope_qkv -min-speedup 1.15

# Run the full test suite
./run.sh test
```

**Requirements:** A Linux computer (`x86_64`), an NVIDIA graphics card with drivers and the CUDA Toolkit installed, Go, Python 3, and Zsh.

---

## Automated Quality Checks

Every update to this repository is automatically tested across:

- **Core Code & Tests** ([`ci.yml`](./.github/workflows/ci.yml)) — Checks formatting, builds the Linux tracker, and runs the full test suite.
- **GPU Compiler & Live Hardware Checks** ([`cuda-hardware.yml`](./.github/workflows/cuda-hardware.yml)) — Compiles all 8 AI building blocks and runs live GPU speed and safety checks when a GPU runner is connected.
- **5 Linux Distributions** ([`linux-compat.yml`](./.github/workflows/linux-compat.yml)) — Verifies clean builds and tests on Ubuntu 22.04, Ubuntu 24.04, Debian 12, Fedora 41, and Arch Linux.

---

## Documentation

> **Every guide is short, clear, and written in plain English.**

| Guide | What You'll Learn |
| :--- | :--- |
| **[Startup & Web Dashboard Guide](./docs/STARTUP_GUIDE.md)** | Step-by-step setup and a simple tour of all three dashboard sections |
| **[Command & `run.sh` Guide](./docs/RUN_SH_MODES.md)** | Every command, mode, and option in `./run.sh`, plus how models are found |
| **[How It Works (Architecture)](./docs/ARCHITECTURE.md)** | A plain-English look inside the pieces that power gpuMode |
| **[Requirements & Setup](./docs/DEPENDENCIES.md)** | What you need installed, what's optional, and copy-paste Linux install commands |
| **[Real Results & Proof](./docs/VERIFICATION_AND_PROOF.md)** | Full before-and-after numbers and how they match official NVIDIA tools |
| **[Savings & Honest Trade-Offs](./docs/RESOURCE_SAVINGS_AND_TRADEOFFS.md)** | Where gpuMode saves speed, memory, energy, and money — and what it trades off |
| **[Progress & Roadmap](./docs/ROADMAP.md)** | What is already built today and what we plan to add next |

