# Real Results & Proof

[← Back to README](../README.md) • [Startup Guide](./STARTUP_GUIDE.md) • [How It Works](./ARCHITECTURE.md) • [Savings & Trade-Offs](./RESOURCE_SAVINGS_AND_TRADEOFFS.md)

> **Real tests on real hardware — cross-checked with official NVIDIA tools.**

Every number in this guide was measured on an actual **NVIDIA GeForce RTX 4050 Laptop GPU** — nothing is guessed or simulated. To prove the readings are trustworthy, we ran the exact same workloads through official independent tools (`llama-bench`, `nvidia-smi`, NVIDIA Nsight Systems, and PyTorch) and compared them side by side.

---

## 1. Local AI Model Results (Before vs. After)

> **Twice the speed, half the wait time, and 21% less energy on the exact same laptop GPU.**

Tested on `Qwen3.5-4B-Q4_K_M.gguf` (4 streaming requests, 192 total output tokens):

| What We Measured | Before (Default Settings) | After (With gpuMode) | Real Change |
| :--- | :---: | :---: | :---: |
| **Generation Speed** | 20.6 tokens/sec | **42.4 tokens/sec** | **2.06× faster** |
| **Total Time for All 4 Answers** | 9.30 seconds | **4.52 seconds** | **51% shorter wait** |
| **Wait for First Word (typical)** | 319 ms | **91 ms** | **3.5× faster** |
| **Time Between Words (typical)** | 41.8 ms | **22.0 ms** | **1.9× faster** |
| **How Busy the GPU Stayed** | 33% | **96%** | **No longer sitting idle** |
| **Peak GPU Memory Used** | 3.44 GB / 6 GB | **4.24 GB / 6 GB** | **Fits safely on a 6 GB card** |
| **Total Energy Used (per 1,000 tokens)** | 1,909 Joules | **1,502 Joules** | **21% less electricity** |
| **Data Sent From CPU to GPU** | 47.5 MB | **9.3 MB** | **5.1× less data shuffling** |
| **Pause Between GPU Tasks** | 71.6 microseconds | **23.9 microseconds** | **3× less waiting** |
| **Requests One GPU Can Serve Per Hour** | 1,548 | **3,183** | **2.06× capacity** |

---

## 2. Double-Checked Against Independent Tools

> **Our built-in readings closely match NVIDIA's official tools and `llama-bench`.**

### A. Full AI Model Accuracy (Tested on Two Different Model Sizes)
We tested both a **4B model** (`Qwen3.5-4B`, fits completely on the GPU) and a **9B model** (`DeepSeek-V4-Pro-9B`, split between CPU and GPU on a 6 GB card) against `llama-bench` and `nvidia-smi`:

| Model Tested | Speed Match vs. `llama-bench` | Power Match vs. `nvidia-smi` | Overall Agreement |
| :--- | :---: | :---: | :---: |
| **Qwen3.5-4B** (2.8 GB file) | **97.6%** match | **95.2%** match | **95.5% match** |
| **DeepSeek-V4-Pro-9B** (5.4 GB file) | **94.0%** match | **97.2%** match | **94.5% match** |
| **Combined Average (18 checks)** | **95.8%** match | **96.2%** match | **95.0% match** |

*Why is there a small ~3.5% speed gap compared to `llama-bench`?* Because gpuMode benchmarks a real streaming web server (`llama-server`) rather than a raw offline script — that small difference is the real cost of streaming answers over HTTP.

### B. GPU Task Count & Sync Check vs. NVIDIA Nsight Systems
| What Was Checked | gpuMode Reading | Independent Tool Reading | Agreement |
| :--- | :---: | :---: | :---: |
| **Decode Speedup** | 1.90× faster | 1.90× faster (`llama-bench`) | **Exact match** |
| **GPU Busy Percentage** | 95.9% | 95.8% (`nvidia-smi`) | **Within 0.1%** |
| **Active GPU Power** | 50.6 Watts | 51.0 Watts (`nvidia-smi`) | **Within 0.4 Watts** |
| **Energy Savings** | 21.3% saved | 21.6% saved (`nvidia-smi`) | **Within 0.3%** |
| **GPU Launches per Token (Before)** | 672 | 666 (Nsight Systems) | **Within 0.9%** |
| **Host Wait Calls Reduced** | 3× tighter gaps | 60% fewer wait calls & 88% fewer copies back to CPU (Nsight) | **Confirmed** |

---

## 3. Core AI Building-Block Speed-Ups

> **Combining separate GPU steps cuts out slow memory trips — matching NVIDIA's profiler to 99.8%.**

| AI Step | Plain-English Role | Before | After | Speed-Up | Match vs. Nsight Profiler |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RoPE + QKV (`rope_qkv`)** | Adds word-position info for attention | 2.80 ms | **0.74 ms** | **3.8× faster** | **99.7% – 99.9%** |
| **Softmax (`softmax`)** | Turns scores into probabilities | 4.50 ms | **1.80 ms** | **2.5× faster** | **99.8% – 99.9%** |
| **RMSNorm + Residual (`rmsnorm_residual`)** | Stabilizes numbers between layers | 3.23 ms | **1.53 ms** | **2.1× faster** | **99.9%** |
| **LayerNorm (`layernorm`)** | Balances layer values | 3.67 ms | **3.25 ms** | **1.1× faster** | **93.9% – 99.8%** |
| **SwiGLU (`swiglu`)** | Activation gate inside feed-forward layers | 4.43 ms | **4.10 ms** | **1.1× faster** | **99.7% – 99.9%** |
| **Live PyTorch Extension (64 layers)** | Drops fused GPU code straight into PyTorch | 31.56 ms | **12.84 ms** | **2.5× faster** | **Exact PyTorch accuracy match** |

---

## 4. Built-In Safety & Accuracy Checks

> **Faster code is never accepted unless it is proven safe and accurate.**

Every speed-up gpuMode tests must pass **three automatic gates**:

1. **No Wasted Memory:** Checks the NVIDIA compiler report to confirm the GPU isn't spilling temporary values into slow backup memory.
2. **Zero Memory Bugs:** Runs NVIDIA's official memory checker (`compute-sanitizer`) to guarantee zero out-of-bounds reads, crashes, or race conditions.
3. **Strict Accuracy Check:** Compares the output against a high-precision CPU calculation — including awkward, uneven data sizes — and rejects anything with noticeable drift.

---

## 5. Verify It Yourself

You don't have to take our word for it. Run these commands on your own machine to compare directly:

```zsh
# 1. Watch live GPU speed, memory, and watts in a separate terminal
nvtop

# 2. Compare default vs. tuned local model speed directly with llama-bench
llama-bench -m /path/to/model.gguf -ngl 24 -fa off -ctk f16 -ctv f16 -b 512 -ub 256 -p 128 -n 48
llama-bench -m /path/to/model.gguf -ngl 99 -fa on  -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -p 128 -n 48

# 3. Inspect exact GPU launches and data copies with NVIDIA Nsight Systems
nsys profile --trace=cuda --stats=true llama-bench -m /path/to/model.gguf -ngl 99 -fa on -p 0 -n 48 -r 1
```

---

## 6. Continuous Automated Testing

Every code change pushed to GitHub is verified automatically by three test suites:

- **Core Code & Tests** ([Run `#45060762811`](https://github.com/Script-Kitty01/GPUmode/actions/runs/45060762811)) — Verifies code quality, rebuilds the Linux tracker, tests error handling, and runs all unit tests.
- **GPU Compiler & Live RTX 4050 Check** ([Run `#45079974915`](https://github.com/Script-Kitty01/GPUmode/actions/runs/45079974915)) — Compiles all 8 AI building blocks cleanly in the cloud and runs live speed and safety checks on real NVIDIA hardware.
- **5-Linux Compatibility Check** ([Run `#45060762909`](https://github.com/Script-Kitty01/GPUmode/actions/runs/45060762909)) — Builds and tests gpuMode across **Ubuntu 22.04**, **Ubuntu 24.04**, **Debian 12**, **Fedora 41**, and **Arch Linux**.


