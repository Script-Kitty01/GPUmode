# Resource Savings & Honest Trade-Offs

[← Back to README](../README.md) • [How It Works](./ARCHITECTURE.md) • [Real Results & Proof](./VERIFICATION_AND_PROOF.md)

> **Where gpuMode saves time, memory, electricity, and money — and the real trade-offs involved.**

---

## The 30-Second Summary

**gpuMode** cuts waste in **five practical ways**:

1. **Less memory back-and-forth on your graphics card** — Combines separate AI steps into one pass, making core steps **1.1× to 3.8× faster** (and **2.5× faster** in PyTorch) while using **40% less temporary memory**.
2. **Smarter memory settings for local AI models** — Fits more of the model onto your graphics card, trims conversation memory in half, cuts CPU-to-GPU data shuffling by **5×**, and **doubles** generation speed.
3. **Less total electricity per answer** — Finishing each answer in half the time cuts total energy used by **21%** on local AI models and **38%** on fused GPU steps.
4. **Fewer cloud servers needed** — Doubling the work one GPU can handle means you can cut server count in half and fit models onto smaller, cheaper cloud GPUs.
5. **Zero GPU memory spent on monitoring** — Watching your card with gpuMode takes **0 MB** of GPU memory, unlike Python-based profilers that often eat 300–800 MB right away.

---

## Part 1: Where the Savings Come From

### 1. Fewer Trips to Slow GPU Memory
*Do more work while the data is already in hand.*

Many steps inside an AI model spend almost all their time moving numbers back and forth between the graphics card's main memory and its computing cores. By default, each step runs separately: Step 1 reads data, writes a temporary copy back to main memory, and Step 2 immediately reads that copy right back.

**gpuMode** combines neighboring steps into a single pass, reads larger chunks of data at once, and keeps temporary numbers right inside the GPU's fastest on-chip storage. That eliminates unnecessary memory trips and shrinks temporary memory use by **40%**.

### 2. Smarter Local AI Model Settings
*Stop making your CPU and GPU play ping-pong.*

Default local AI settings often leave too much of the model on the CPU and store conversation history at full uncompressed size. That forces constant back-and-forth copies between your CPU and graphics card.

**gpuMode** checks how much free GPU memory you actually have, places the right number of layers onto the GPU, turns on memory-friendly attention, and compresses the conversation cache to half its normal size. On a 4B local model, that cuts CPU-to-GPU data traffic by **5×**, eliminates **60%** of pause/wait calls, and **doubles** token generation speed.

### 3. Why Working Harder Saves Total Electricity
*Sprinting for 4.5 seconds and resting uses less energy than jogging slowly for 9.3 seconds.*

People often ask: *"If gpuMode pushes my GPU from 33% busy up to 96% busy — and power rises from 39 Watts to 64 Watts — how does it save electricity?"*

Think about the difference between **power right now (Watts)** and **total energy used over the whole job (Joules)**:
- Whenever your graphics card wakes up to run AI, it immediately draws a hefty baseline amount of power just to stay awake and keep memory running — even if the cores are mostly waiting around.
- With default settings, the GPU stays awake at ~39 Watts for **9.3 seconds**.
- With gpuMode, the GPU works efficiently at ~64 Watts, finishes the exact same work in **4.5 seconds**, and drops straight back to idle (~13 Watts).
- Moving fewer bytes over slow memory and CPU-to-GPU cables also avoids the most power-hungry operations on the card.

Result: **21% less total energy** per 1,000 tokens on a real local AI model (and **38% less energy** on fused GPU steps).

### 4. Fewer and Cheaper Cloud GPUs
*Fit on smaller cards and serve twice the traffic per machine.*

When one graphics card handles twice as many requests per hour (`1,548 → 3,183`), you need half as many GPUs to serve the same users. And because tighter memory settings keep models smaller, workloads that used to spill onto expensive 48 GB cloud cards can often run comfortably on cheaper 24 GB cards — or shift background batch jobs onto discounted cloud servers.

### 5. Zero GPU Memory Wasted on Monitoring
*Keep 100% of your GPU memory for your actual AI models.*

Traditional Python monitoring scripts grab **300–800 MB** of GPU memory just by starting up — enough to crash a model on a 6 GB or 8 GB laptop card. **gpuMode** reads sensors directly from the driver and watches activity from outside the GPU runtime, using **0 MB** of GPU memory.

---

## Part 2: Honest Trade-Offs (And How We Handle Them)

No speed-up comes completely free. Here are the six real trade-offs gpuMode makes, and how we keep them safe:

### 1. Higher Peak Power and Heat During Busy Bursts
- **The trade-off:** Keeping the GPU 96% busy raises peak power draw (+24 Watts in our test) and creates more heat while an answer is being generated — even though total energy per answer drops.
- **How gpuMode handles it:** Continuously watches temperatures and thermal throttling, and lets you choose cooler, power-friendly options instead of max-speed settings when thermal headroom is tight.

### 2. Uses More GPU Memory to Hold More of the Model
- **The trade-off:** Moving more of an AI model onto the graphics card uses more GPU memory (`3.4 GB → 4.2 GB` on a 4B model), leaving less spare room for huge prompts or other GPU apps.
- **How gpuMode handles it:** Compresses conversation memory in half and checks real free GPU memory before starting. For a larger 9B model on a 6 GB card, it automatically chooses a safe partial fit instead of crashing with an out-of-memory error.

### 3. Takes a Few Seconds Up Front to Test and Verify Speed-Ups
- **The trade-off:** Compiling and safety-checking custom GPU code takes several seconds the first time — great for ongoing work, not worth it for a single quick prompt.
- **How gpuMode handles it:** Compiles candidates in parallel across CPU cores, uses smart search to test fewer options, and saves verified winners on disk in `.gpumode_data` so you don't repeat the work.

### 4. Tiny Rounding Differences (Not Bit-for-Bit Identical)
- **The trade-off:** Combining steps and compressing conversation memory changes the order in which numbers are rounded, so raw numbers can differ by tiny fractions.
- **How gpuMode handles it:** Every speed-up must pass a strict accuracy check against a high-precision CPU calculation (including uneven data sizes) before it is ever accepted.

### 5. Small CPU Cost When Deep Linux Tracking Is On
- **The trade-off:** Watching every CPU-to-GPU call from Linux adds roughly **1% to 3%** extra CPU work during heavy generation and requires a one-time permission step (`./run.sh setcap`).
- **How gpuMode handles it:** Summarizes counts inside Linux to keep overhead tiny, and lets you turn deep tracking off anytime (`-ebpf off` or `GPUMODE_EBPF=off`).

### 6. Built Specifically for Linux and NVIDIA Graphics Cards
- **The trade-off:** Live speed-ups, sensor reading, and Linux tracking are built specifically for Linux (`x86_64`) with NVIDIA GPUs — not AMD, Apple Silicon, or Windows.
- **How gpuMode handles it:** Includes safe fallback stubs and mock tests so the project still builds and passes unit tests on non-GPU machines.

---

## Quick Summary Table

| What You Gain | The Honest Trade-Off | How gpuMode Protects You |
| :--- | :--- | :--- |
| **21%–38% less total energy** | Higher peak power & heat during bursts | Live temperature & throttle watch + cooler energy-saving options |
| **2× faster local AI & 5× less data shuffling** | Uses more GPU memory to hold the model | Half-size conversation cache + automatic free-memory safety cap |
| **1.1×–3.8× faster GPU steps** | Takes a few seconds up front to tune | Parallel builds, smart search, and saved results on disk |
| **Faster combined steps & smaller cache** | Tiny floating-point rounding differences | Strict accuracy gate rejects any noticeable drift |
| **0 MB GPU memory for monitoring** | ~1%–3% CPU work when deep tracking is on | Summarized in Linux + easy off switch (`GPUMODE_EBPF=off`) |
| **99.8% timing accuracy on real hardware** | Linux + NVIDIA GPUs only | Safe stubs for testing on machines without a GPU |

