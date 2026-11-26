# Startup & Web Dashboard Guide

[← Back to README](../README.md) • [Command Guide](./RUN_SH_MODES.md) • [Requirements](./DEPENDENCIES.md)

> **Go from zero to a live GPU dashboard in under two minutes.**

---

## 1. First-Time Setup

### Step 1 — Check That Your Tools Are Ready
Make sure your Linux computer has an NVIDIA graphics card with drivers, the CUDA Toolkit, Go, Python 3, and Zsh installed:

```zsh
nvidia-smi -L
nvcc --version
compute-sanitizer --version
go version
python3 --version
```

Need to install any of these? See the copy-paste install commands in [Requirements & Setup](./DEPENDENCIES.md).

### Step 2 — Download and Launch

```zsh
# 1. Download the project and enter the folder
git clone https://github.com/Script-Kitty01/GPUmode.git
cd GPUmode

# 2. (Optional, one-time) Enable deeper Linux GPU tracking without running as root
#    This asks for your password once so the app itself never needs admin rights
./run.sh setcap

# 3. (Optional) Install PyTorch in a local folder if you want live PyTorch speed-up checks
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip torch

# 4. Build and start the Web Dashboard
./run.sh
```

Open **`http://127.0.0.1:8080`** in your browser.

### What Happens When You Run `./run.sh`
1. **Fast startup** — Only rebuilds the app if code files actually changed.
2. **Automatic hardware check** — Detects your exact NVIDIA graphics card and its live limits.
3. **Optional PyTorch check** — If you created `.venv` with PyTorch, runs a quick check to measure PyTorch speed-ups.
4. **Live dashboard** — Opens the web dashboard on your computer at `http://127.0.0.1:8080` and starts streaming live GPU stats.

---

## 2. Tour of the Web Dashboard

> **Three clear sections: speed up GPU code, test local AI models, and plan cloud costs.**

### Section 1 — GPU Speed-Up Lab
*Test and speed up core AI building blocks on your live graphics card.*

1. Pick one of the **8 AI building blocks** (for example, `rmsnorm_residual`, `rope_qkv`, `softmax`, or `swiglu`), pick a search mode, and set your data size.
2. Click **Compile, Verify & Benchmark on GPU**.
3. Review the results:
   - **Plain-English answers** to six key questions: why the GPU is slow, what is causing it, what fixes it, whether the fix is safe and accurate, how much energy it saves, and whether you can use fewer GPUs.
   - **Live power chart** and **CPU-to-GPU activity summary**, plus quick export buttons for timeline traces and monitoring tools.
   - **Ready-to-use GPU and PyTorch code** for the winning speed-up.

### Section 2 — Local AI Model Runner (Before vs. After)
*Test any local AI model and see the real difference gpuMode makes.*

1. **Pick a model in either of two ways:**
   - **Paste a file path directly:** Drop any `.gguf` file path (like `~/Downloads/Qwen3.5-4B-Q4_K_M.gguf`) or folder into **Direct Model Path** — no need to move your files.
   - **Or choose an auto-detected model:** Pick from models found automatically in `~/AI/Models`, `~/models`, your Hugging Face cache, `llama-server`, or `Ollama`.
2. Choose how many test requests to send, how many users to simulate at once (`1` to `32`), and how long each answer should be.
3. Click **Run Without Project (Baseline)** to record normal performance.
4. Click **Run With Project (gpuMode)** to apply smart memory and speed settings, then compare **Before vs. After** side by side — including tokens per second, wait time for the first word, GPU memory, energy used, and estimated cost per million tokens.

### Section 3 — Cloud Cost & Server Planner
*Find out which graphics cards fit your model and how to spend less in the cloud.*

1. Enter your expected traffic, target response speed, hourly GPU price, and cloud discount percentage, then click **Recompute Fleet Placement & Cost**.
2. Compare the recommended **standard** and **discounted cloud server** options — showing whether your model fits comfortably in GPU memory, expected speed, cost per million tokens, and how many servers you need.
3. Review how traffic priorities protect important live requests when servers get busy, and compare three ways of spreading work across multiple GPUs.

---

## 3. Common Customizations

> **Change ports, pick another GPU, or turn off extra tracking with one line.**

```zsh
# Run the dashboard on a different port
GPUMODE_ADDR=127.0.0.1:9090 ./run.sh

# Use the second NVIDIA graphics card on a multi-GPU computer
GPUMODE_GPU_INDEX=1 ./run.sh

# Use nvidia-smi if direct driver reading is unavailable
GPUMODE_TELEMETRY=smi ./run.sh

# Turn off deeper Linux tracking (useful inside restricted containers)
GPUMODE_EBPF=off ./run.sh
```

To stop the dashboard, press **`Ctrl+C`** in the terminal where `./run.sh` is running.

---

## 4. Settings & Environment Variables

You rarely need to set these — gpuMode finds almost everything automatically — but you can override any setting if needed:

| Setting | Default | What It Controls |
| :--- | :--- | :--- |
| `GPUMODE_ADDR` | `127.0.0.1:8080` | Web address and port for the dashboard |
| `GPUMODE_GPU_INDEX` | `0` | Which NVIDIA GPU to monitor and test |
| `GPUMODE_TELEMETRY` | `auto` | How GPU stats are read (`auto`, `nvml`, or `smi`) |
| `GPUMODE_EBPF` | `auto` | Turn deeper Linux tracking `auto`, `on`, or `off` |
| `GPUMODE_TRACE_PID` | `0` (auto) | Track one specific running process by its process ID |
| `GPUMODE_MODEL_PATH` | Auto-found | Default `.gguf` model file to use |
| `GPUMODE_MODEL_DIR` | — | Extra folder to scan for `.gguf` models |
| `GPUMODE_MODEL_NAME` | Auto-found | Default model name for local server benchmarks |
| `GPUMODE_NGL` | Auto-sized | Manually set how many model layers go on the GPU |
| `GPUMODE_PYTHON` | `python3` | Python command to use for PyTorch checks |
| `GPUMODE_NVCC` | Auto-found | Custom path to the NVIDIA GPU compiler (`nvcc`) |
| `GPUMODE_SANITIZER` | Auto-found | Custom path to the NVIDIA memory checker (`compute-sanitizer`) |
| `GPUMODE_NVIDIA_SMI` | Auto-found | Custom path to `nvidia-smi` |

---

## 5. Dashboard Web Endpoints

If you want to connect your own scripts or monitoring dashboards to gpuMode while it runs at `http://127.0.0.1:8080`:

| Address | Method | What It Gives You |
| :--- | :---: | :--- |
| **`/`** | `GET` | The live Web Dashboard |
| **`/api/state`** | `GET` | Current GPU info, latest results, live stats history, and detected models |
| **`/api/models`** | `GET` | List of found local models (supports `?dir=/your/folder`) |
| **`/api/optimize`** | `POST` | Runs a GPU speed-up and safety check |
| **`/api/recompute`** | `POST` | Recalculates cost and server sizing with new traffic or price inputs |
| **`/api/workload`** | `POST` | Runs a local AI model benchmark (Baseline or Tuned) |
| **`/api/schedule`** | `GET` / `POST` | Runs the GPU cost, fit, and server planner |
| **`/api/telemetry/history`** | `GET` | Recent history of live GPU sensor readings |
| **`/api/stream/telemetry`** | `GET` | Live streaming feed of GPU power, speed, temperature, and memory |
| **`/api/trace/perfetto`** | `GET` | Downloads a timeline file you can view at [ui.perfetto.dev](https://ui.perfetto.dev) |
| **`/metrics`** | `GET` | Live GPU metrics formatted for Prometheus monitoring |

---

## 6. Optional: Running GitHub Checks on Your Own GPU

If you want GitHub Actions to run the live GPU check on your own Linux computer:

```zsh
# 1. Download the GitHub Actions runner in a temporary folder
mkdir -p /tmp/gpumode_gh_runner && cd /tmp/gpumode_gh_runner
curl -fsSL -o actions-runner.tar.gz https://github.com/actions/runner/releases/download/v2.338.2/actions-runner-linux-x64-2.338.2.tar.gz
tar xzf actions-runner.tar.gz

# 2. Register it with your repository
REG_TOKEN=$(gh api -X POST repos/Script-Kitty01/GPUmode/actions/runners/registration-token --jq .token)
./config.sh --url https://github.com/Script-Kitty01/GPUmode --token "$REG_TOKEN" --name "cuda-gpu-runner" --labels "self-hosted,linux,cuda" --ephemeral --unattended --replace

# 3. Trigger the workflow and start the runner
gh workflow run cuda-hardware.yml -f run_self_hosted_gpu=true
./run.sh
```


