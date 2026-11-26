# Command & `run.sh` Guide

[← Back to README](../README.md) • [Startup Guide](./STARTUP_GUIDE.md) • [Requirements](./DEPENDENCIES.md)

> **One simple script to build, monitor, benchmark, and tune your GPU.**

`./run.sh` is the single launcher for **gpuMode**. You can run it from anywhere — it automatically moves to the project folder, rebuilds only when code changes, checks optional Linux permissions, and launches the mode you asked for.

---

## What `./run.sh` Handles Automatically

- **Smart Rebuilds:** Only recompiles `./gpumode` when source files have changed, so startups stay fast and saved permissions stay intact.
- **Permission Checks:** Checks whether deeper Linux GPU tracking is enabled; if not, it either offers to enable it or continues cleanly without it.
- **No Port Conflicts:** If the Web Dashboard is already running on `http://127.0.0.1:8080`, running `./run.sh` again simply runs a fresh GPU speed-up pass instead of crashing.

---

## Command Cheat Sheet

| Command | Shortcut | What It Does | Example |
| :--- | :--- | :--- | :--- |
| **`dashboard`** *(default)* | `serve` | Starts the live Web Dashboard at `http://127.0.0.1:8080` | `./run.sh` |
| **`top`** | `watch`, `tui` | Opens a live updating terminal monitor (uses 0 GPU memory) | `./run.sh top` |
| **`stats`** | — | Prints a quick, clean GPU & active model summary card | `./run.sh stats` |
| **`install`** | `install-cli` | Installs `gpumode` globally to `~/.local/bin/gpumode` | `./run.sh install` |
| **`workload`** | — | Benchmarks a local AI model (speed, wait time, memory, power, cost) | `./run.sh workload -deepseek-model /path/model.gguf` |
| **`optimize`** | `analyze` | Finds and verifies faster GPU code for a core AI step | `./run.sh optimize -op rmsnorm_residual` |
| **`deepseek`** | — | Uses a local AI model to suggest GPU tuning settings and verifies them | `./run.sh deepseek -deepseek-model /path/model.gguf` |
| **`schedule`** | — | Recommends GPU sizes, cloud tiers, and traffic handling for a model | `./run.sh schedule -deepseek-model /path/model.gguf` |
| **`check`** | — | Automated pass/fail speed and safety check for CI pipelines | `./run.sh check -op rope_qkv -min-speedup 1.15` |
| **`monitor`** | — | Prints a compact one-line snapshot of GPU speed, memory, and power | `./run.sh monitor` |
| **`ebpf`** | `probe` | Checks whether live GPU sensors and Linux tracking are active | `./run.sh ebpf` |
| **`setcap`** | — | One-time permission setup for rootless Linux tracking | `./run.sh setcap` |
| **`bpf-gen`** | — | Rebuilds the Linux tracker code (only needed if you edit tracker C files) | `./run.sh bpf-gen` |
| **`test`** | — | Runs the full automated test suite | `./run.sh test` |

---

## Every Mode Explained

### 1. `dashboard` (Default)
*Open the visual web dashboard to monitor, tune, and compare models in your browser.*

```zsh
# Start on http://127.0.0.1:8080
./run.sh

# Start on a custom port
GPUMODE_ADDR=127.0.0.1:9090 ./run.sh dashboard
```

### 2. `top` & `stats`
*Watch your graphics card right inside your terminal — even while large AI models are loaded.*

```zsh
# Live refreshing terminal screen (press q to quit)
./run.sh top

# One-time summary card (add -json for machine-readable output)
./run.sh stats

# Install globally so you can type 'gpumode top' from any folder
./run.sh install
gpumode top
```

### 3. `workload`
*Measure real-world speed, wait time, GPU memory, energy, and cost on a local AI model.*

```zsh
# Automatically picks a local .gguf model and runs 4 streaming prompts
./run.sh workload -llm-reqs 4 -llm-tokens 48

# Test a specific .gguf file with 2 simulated users at once
./run.sh workload \
  -deepseek-model ~/AI/Models/Qwen3.5-4B-Q4_K_M.gguf \
  -llm-reqs 4 \
  -llm-concurrency 2 \
  -llm-tokens 64

# Test an already-running Ollama model
./run.sh workload \
  -auto-server=false \
  -llm-url http://127.0.0.1:11434 \
  -llm-model qwen2.5:7b \
  -llm-reqs 4
```

> **Tip:** Want an easy visual **Before vs. After** comparison? Run `./run.sh` and use **Section 2 (Local AI Model Runner)** in the Web Dashboard.

### 4. `optimize`
*Automatically test faster GPU code on your real hardware and keep only safe, accurate winners.*

```zsh
# Speed up a fused normalization step (default smart search)
./run.sh optimize -op rmsnorm_residual -batch 256 -hidden 4096 -iters 40

# Try other supported AI building blocks
./run.sh optimize -op rope_qkv
./run.sh optimize -op softmax
./run.sh optimize -op swiglu
./run.sh optimize -op layernorm -json
```

**Supported `-op` names:** `rmsnorm_residual`, `rmsnorm`, `layernorm`, `softmax`, `swiglu`, `rope_qkv`, `qgemv_decode`, `gqa_attention`.

### 5. `deepseek`
*Let a local AI model suggest GPU tuning ideas, then verify them on your real graphics card.*

```zsh
# Uses an auto-detected local .gguf model
./run.sh deepseek -op rmsnorm_residual

# Or point to a specific .gguf file
./run.sh deepseek -deepseek-model ~/AI/Models/Qwen3.5-4B-Q4_K_M.gguf -op swiglu
```

### 6. `schedule`
*Estimate which cloud GPUs fit your model, how much serving costs, and how to handle busy traffic.*

```zsh
./run.sh schedule -deepseek-model ~/AI/Models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf -llm-model Qwen3.5-4B-Q4_K_M

# Machine-readable JSON output
./run.sh schedule -json
```

### 7. `check`
*Pass/fail hardware check to make sure code changes stay fast, memory-safe, and accurate.*

```zsh
./run.sh check -op rope_qkv -batch 64 -hidden 2048 -iters 20 -min-speedup 1.15
```

Exits with an error if the speedup drops below your target, memory errors appear, or accuracy drifts.

### 8. `monitor`, `ebpf`, `setcap`, `bpf-gen`, & `test`
*Quick diagnostics, one-time permissions, and testing.*

```zsh
# One-line GPU health snapshot (no special permissions needed)
./run.sh monitor

# Check sensor and Linux tracking status
./run.sh ebpf

# Grant one-time rootless tracking permissions and verify them
./run.sh setcap

# Rebuild Linux tracker files (only after editing C tracker source)
./run.sh bpf-gen

# Run all automated tests
./run.sh test
```

---

## How gpuMode Finds Your Local AI Models

You can supply local models in any of three ways:

1. **Paste a path in the Web Dashboard:** Drop any `.gguf` file or folder straight into **Direct Model Path** in Section 2.
2. **Automatic discovery:** gpuMode automatically checks `~/AI/Models`, `~/models`, `./models`, your Hugging Face cache (`~/.cache/huggingface/hub`), a running `llama-server` on port `8081`, and `Ollama` on port `11434`.
3. **Pass a flag in the terminal:** Use `-deepseek-model /path/to/model.gguf` or `-llm-model your-model-name`.

---

## Common Command-Line Options

| Flag | Default | What It Does |
| :--- | :--- | :--- |
| `-deepseek-model <path>` | Auto-found | Path to a local `.gguf` model file |
| `-llm-url <url>` | `http://127.0.0.1:8081` | Address of an existing local model server or Ollama |
| `-llm-model <name>` | Auto-found | Name of the model served by Ollama or another server |
| `-llm-reqs <number>` | `4` | How many test requests to run |
| `-llm-concurrency <number>` | `1` | How many users to simulate at the same time (`1` to `32`) |
| `-llm-tokens <number>` | `48` | Maximum tokens generated per test request |
| `-auto-server` | `true` | Automatically start a temporary local server for `.gguf` tests |
| `-min-speedup <number>` | `1.10` | Minimum speedup required for `./run.sh check` to pass |
| `-addr <host:port>` | `127.0.0.1:8080` | Web Dashboard address |
| `-gpu <number>` | `0` | Which NVIDIA GPU to use (on multi-GPU machines) |
| `-ebpf <mode>` | `auto` | Set Linux tracking to `auto`, `on`, or `off` |
| `-json` | `false` | Print results as clean JSON instead of formatted text |

