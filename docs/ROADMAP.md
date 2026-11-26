# Progress & Roadmap

[← Back to README](../README.md) • [How It Works](./ARCHITECTURE.md) • [Startup Guide](./STARTUP_GUIDE.md) • [Real Results & Proof](./VERIFICATION_AND_PROOF.md)

> **What is already built and verified today — and where gpuMode is headed next.**

---

## What's Already Live Today

All six major milestones of **gpuMode** are built, verified on real NVIDIA hardware, and ready to use right now:

| Milestone | Status | What It Gives You | Try It |
| :--- | :---: | :--- | :--- |
| **1. Live GPU & CPU-to-GPU Tracking** | **Done** | Fast sensor readings (power, speed, temperature, memory, throttling) plus lightweight Linux tracking with Perfetto timeline export | `./run.sh ebpf` |
| **2. Live PyTorch Speed-Up Checks** | **Done** | Builds and tests fused GPU code directly inside PyTorch and measures real before-and-after speed and accuracy | `./.venv/bin/python3 scripts/profile_pytorch.py --verify-extension` |
| **3. Real Hardware Speed Limits** | **Done** | Measures your exact graphics card's real memory and compute limits at startup instead of relying on guesses | `./run.sh optimize -op rmsnorm_residual` |
| **4. 8 Core AI Building Blocks & Self-Fixing Tuning** | **Done** | Speeds up 8 key AI operations (including attention, rotary embeddings, and quantized decode) with automatic error fixing and safety checks | `./run.sh check -op rope_qkv -min-speedup 1.15` |
| **5. Multi-User Benchmarks & Monitoring** | **Done** | Tests up to 32 simulated users at once, tracks cost per million tokens, exports Prometheus `/metrics`, and includes a live terminal monitor (`gpumode top`) | `./run.sh workload -llm-reqs 8 -llm-concurrency 2` |
| **6. Cloud Cost Planner, Clean UI & 5-Linux CI** | **Done** | Recommends GPU sizes and cloud tiers, simulates traffic priorities, uses a clean modular web dashboard, and tests across 5 Linux distributions | `./run.sh schedule && ./run.sh test` |

---

## Detailed Breakdown of Completed Highlights

### Live Hardware & Activity Visibility
*Know exactly why your GPU is slowing down — without taking away GPU memory.*
- Reads power, clocks, temperature, memory, and slowdown reasons directly from the NVIDIA driver.
- Tracks how your CPU sends tasks and data to the GPU, highlights pause spikes, and exports interactive timeline files for [ui.perfetto.dev](https://ui.perfetto.dev).

### End-to-End PyTorch Verification
*Don't just generate faster code — prove it runs faster inside real PyTorch.*
- Compiles the winning GPU code directly inside PyTorch, checks that the answers match, and reports the real before-and-after speedup (**2.5× faster** across 64 layers on an RTX 4050).

### Real Hardware Measurements (Zero Guesswork)
*Every GPU is different — gpuMode measures yours live.*
- Runs a brief calibration on startup to measure your card's actual memory speed and checks how efficiently each GPU speed-up uses the card's compute cores.

### Broader AI Building Blocks & Automatic Self-Fixing
*Covers the steps that matter most in modern AI models.*
- Supports 8 core AI steps — including normalization, activations, attention, position encoding, and quantized decoding — plus automatic retries that read compiler or memory-checker errors and repair them.

### Multi-User Benchmarks, Terminal Monitor & Cost Planner
*Built for everyday local testing and cloud planning.*
- Simulate multiple users hitting a local AI model at the same time (`-llm-concurrency`), monitor live stats right in your terminal (`gpumode top` / `gpumode stats`), export `/metrics` for Prometheus, and plan the cheapest cloud GPU mix with `./run.sh schedule`.

---

## What's Coming Next

> **Making local and cloud AI even cooler, quieter, and easier to tune automatically.**

1. **Automatic Power-Limit Tuning (Cooler & Quieter GPUs)**
   *Find the sweet spot between full speed and lower fan noise / heat.*
   - Test lower GPU power limits automatically to find settings that keep 95%+ of your speed while running noticeably cooler and using even less electricity.

2. **Full Automatic Search for Local AI Server Settings**
   *Go beyond presets to automatically discover the best settings for any model file.*
   - Automatically test combinations of GPU layers, batch sizes, and conversation cache compression on your exact model — paired with an output-quality check so answers never degrade.

3. **Broader Hardware Support**
   *Bring the same simple experience to more graphics cards over time.*
   - Explore backend adapters beyond Linux + NVIDIA while keeping the core experience fast and lightweight.


