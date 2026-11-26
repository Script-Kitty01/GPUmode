# How It Works (Architecture)

[← Back to README](../README.md) • [Startup Guide](./STARTUP_GUIDE.md) • [Real Results & Proof](./VERIFICATION_AND_PROOF.md) • [Savings & Trade-Offs](./RESOURCE_SAVINGS_AND_TRADEOFFS.md)

> **A plain-English look inside the pieces that power gpuMode.**

---

## The Big Picture

**gpuMode** follows one clear, six-step loop from start to finish:

```text
1. Watch Your GPU
   Read live speed, memory, temperature, power, and CPU-to-GPU traffic.
        │
        ▼
2. Find What's Slow
   See whether your GPU is waiting on memory, waiting on the CPU, or running full speed.
        │
        ▼
3. Try Faster Settings & GPU Code
   Test smarter ways to combine steps and use GPU memory on your actual card.
        │
        ▼
4. Double-Check Safety & Accuracy
   Reject anything with memory bugs or noticeable output differences.
        │
        ▼
5. Measure Real Energy Used
   Track live power draw across the run to see how much electricity was saved.
        │
        ▼
6. Recommend the Best Setup
   Pick the best balance of speed, memory, energy, and cloud server cost.
```

---

## The Five Main Parts of gpuMode

### 1. Live GPU Sensor Reader
*Fast, accurate hardware readings without taking memory away from your AI models.*

- Talks directly to the NVIDIA driver every 10 milliseconds to read power draw, temperature, GPU busy percentage, clock speed, memory usage, and active programs.
- Automatically falls back to `nvidia-smi` if direct driver reading is not available.
- Tracks total energy used across each benchmark run so you can compare Before vs. After honestly.
- **Where it lives:** `internal/nvml/` and `internal/telemetry/`

### 2. CPU-to-GPU Activity Tracker
*Spots hidden pauses and wasted data copies between your CPU and graphics card.*

- Uses lightweight Linux kernel tracking to watch when your CPU sends tasks to the GPU, copies data back and forth, or sits waiting for the GPU to finish.
- Counts how many GPU tasks run per generated word and highlights awkward gaps where the graphics card is sitting idle.
- Exports a clean timeline file you can open in your browser with Perfetto.
- **Where it lives:** `internal/ebpftrace/`

### 3. Automatic Speed-Up & Safety Engine
*Tests faster GPU code on your real card — and proves it is safe before recommending it.*

- Measures your graphics card's real-world speed limits on startup (never guesses or uses hardcoded numbers).
- Tests faster versions of 8 common AI building blocks that combine separate steps into a single pass, cutting down slow memory round-trips.
- Every single candidate must pass automatic memory-safety checks and strict accuracy checks — including awkward, uneven data sizes — before gpuMode will recommend it.
- If a test fails, gpuMode reads the error and automatically tries to fix it.
- **Where it lives:** `internal/roofline/`, `internal/optimizer/`, `internal/harness/`, and `internal/profiler/`

### 4. Local AI Model Runner & Cloud Cost Planner
*Turns hardware measurements into clear answers about speed and cost.*

- Finds your local `.gguf` and Ollama models automatically, or accepts any file path you paste in.
- Runs real side-by-side **Before vs. After** benchmarks with up to 32 simulated users at once.
- Checks which local or cloud graphics cards fit your model cleanly in GPU memory, estimates cost per million tokens (standard and discounted cloud servers), and simulates how to keep urgent traffic fast when servers get busy.
- **Where it lives:** `internal/models/`, `internal/workload/`, `internal/scheduler/`, `internal/physics/`, and `internal/pareto/`

### 5. Built-In Web Dashboard & Terminal Monitor
*Simple to launch — everything is bundled into one self-contained program.*

- The entire Web Dashboard (HTML panels, stylesheet, and scripts) is built directly into the `gpumode` app — no Node.js or extra web server required.
- Also includes a live terminal monitor (`gpumode top`) and instant stats card (`gpumode stats`) that use zero GPU memory.
- Saves past verified reports locally in `.gpumode_data` so results load immediately when you restart.
- **Where it lives:** `frontend/`, `internal/server/`, `internal/tui/`, and `internal/store/`

---

## Project Folder Map

```text
gpuMode/
├── cmd/gpumode/        # Main app entry point (web dashboard + terminal commands)
├── frontend/           # Built-in Web Dashboard (HTML sections, CSS theme, and JS modules)
├── internal/
│   ├── config/         # Finds local tools, models, and reads your settings
│   ├── nvml/           # Direct NVIDIA driver sensor reader
│   ├── telemetry/      # Live sensor loop, energy tracking, and dashboard streaming
│   ├── ebpftrace/      # Lightweight Linux CPU-to-GPU activity tracker
│   ├── models/         # Finds local .gguf, Hugging Face, and Ollama AI models
│   ├── workload/       # Runs Before vs. After local AI model benchmarks
│   ├── physics/        # Measures real GPU speed limits and calculates cost metrics
│   ├── roofline/       # Figures out what is slowing the GPU down
│   ├── optimizer/      # Catalog of 8 AI building blocks + automatic speed-up search
│   ├── harness/        # Compiles, safety-checks, and times GPU code on real hardware
│   ├── profiler/       # Builds the 6-question report and ready-to-use PyTorch code
│   ├── pareto/         # Picks the best balance of speed, energy, and memory
│   ├── scheduler/      # Cloud GPU fit, cost planner, and traffic priority simulator
│   ├── server/         # Web server, API endpoints, and Prometheus /metrics feed
│   ├── store/          # Saves verified reports locally (.gpumode_data)
│   └── tui/            # Live terminal monitor (gpumode top / gpumode stats)
├── scripts/            # Optional PyTorch live speed-up check script
├── docs/               # Plain-English guides and benchmark proof
└── run.sh              # One-command build and launch script
```

