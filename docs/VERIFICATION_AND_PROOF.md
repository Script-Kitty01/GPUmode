# Hardware Verification & Benchmark Proof

[← Back to README](../README.md) • [Startup Guide](./STARTUP_GUIDE.md) • [Architecture](./ARCHITECTURE.md)

Every optimization and benchmark reported by **gpuMode** is measured on live NVIDIA silicon — never simulated or hardcoded.

---

## 1. Local LLM Before/After Proof (`Qwen3.5-4B-Q4_K_M.gguf` on RTX 4050 `sm_89`)

Measured end-to-end through `POST /api/workload` (`4` streaming requests, `192` output tokens) with `10 ms` direct `NVML` power integration and rootless Linux `eBPF` uprobes attached to `libcuda.so.1`:

| Metric | Without Project (`Baseline`: `ngl=24, fa=off, kv=f16`) | With Project (`gpuMode`: `ngl=99, fa=on, kv=q8_0`) | Measured Change |
| :--- | :--- | :--- | :--- |
| **Sustained Throughput** | `20.65 tok/s` (`0.43 req/s`) | **`42.44 tok/s` (`0.88 req/s`)** | **`2.06×` faster** |
| **Total Wall Time** | `9.30 s` | **`4.52 s`** | **`-51.4%` duration** |
| **TTFT (p50 / p95)** | `318.8 ms` / `496.6 ms` | **`91.1 ms` / `104.3 ms`** | **`3.50×` / `4.76×` faster** |
| **TPOT (p50)** | `41.77 ms/tok` | **`21.97 ms/tok`** | **`1.90×` faster** |
| **GPU SM Utilization (Avg / Peak)** | `33.4%` / `43.0%` | **`95.9%` / `96.0%`** | **`+62.5` pts SM util** |
| **Measured Peak VRAM** | `3,443 MiB` / `6,141 MiB` | `4,242 MiB` / `6,141 MiB` | Fits cleanly in 6 GB VRAM |
| **Board Energy / 1K Tokens (`NVML`)** | `1,909.32 J` (`39.42 W` avg) | **`1,501.82 J`** (`63.74 W` avg) | **`-21.3%` Joules / tok** |
| **eBPF Inter-Launch CPU Gap** | `71.6 µs` | **`23.9 µs`** | **`3.0×` tighter dispatch** |
| **eBPF PCIe `H2D` Runtime Traffic** | `47.53 MiB` | **`9.26 MiB`** | **`5.1×` less PCIe traffic** |
| **Single-GPU Capacity** | `1,548 req/GPU-hr` | **`3,183 req/GPU-hr`** | **`2.06×` capacity** |

---

## 2. Independent Cross-Verification (`gpuMode` vs. `nsys` + `llama-bench` + `nvidia-smi`)

To verify that **gpuMode**'s internal `NVML` and `eBPF` numbers match NVIDIA's official profilers, the exact same `Qwen3.5-4B-Q4_K_M.gguf` Baseline and Tuned profiles were profiled independently using **NVIDIA Nsight Systems (`nsys`)**, **`llama-bench`**, and **`nvidia-smi`** (`50 ms` loop):

| Metric | **gpuMode** (`NVML` + `eBPF`) | **External Tools** (`nsys` / `llama-bench` / `nvidia-smi`) | Agreement |
| :--- | :--- | :--- | :--- |
| **Decode Speedup (`Baseline → Tuned`)** | **`1.90×`** TPOT speedup (`41.77 → 21.97 ms/tok`) | **`1.90×`** `tg48` speedup (`25.47 → 48.48 tok/s` in `llama-bench`) | **Exact `1.90×` match** |
| **Tuned GPU SM Utilization** | **`95.9%`** avg | **`95.8%`** avg (`nvidia-smi` 50 ms) | **Within `0.1%`** |
| **Tuned Active GPU Power** | **`50.62 W`** active | **`51.00 W`** active (`nvidia-smi` 50 ms) | **Within `0.38 W` (`0.7%`)** |
| **Energy Reduction per 1K Tokens** | **`-21.3%`** (`1,909 → 1,502 J`) | **`-21.6%`** (`1,342 → 1,052 J` active) | **Within `0.3%`** |
| **CUDA Kernel Launches / Token (Baseline)** | **`671.9 launches/tok`** (Linux eBPF uprobe) | **`666.0 launches/tok`** (`31,967 / 48` in `nsys` CUPTI) | **Within `0.9%`** |
| **CUDA Kernel Launches / Token (Tuned)** | **`978.6 launches/tok`** (incl. prompt prefill) | **`939.1 launches/tok`** (`45,078 / 48` in `nsys` CUPTI) | **Within `4.0%`** |
| **Host ↔ GPU Stream Sync Reduction** | Inter-launch gap **`71.6 → 23.9 µs`** | `nsys`: `cudaStreamSynchronize` **`3,988 → 1,576` (`-60%`)**; D2H copies **`392 → 49` (`-87.5%`)** | Confirmed |

### Reproduce Independent Verification Yourself

```zsh
# 1. Watch real-time SM %, Memory BW %, Watts, and VRAM in a second terminal
nvtop

# 2. Verify raw Baseline vs Tuned token generation speed via llama-bench
llama-bench -m /path/to/model.gguf -ngl 24 -fa off -ctk f16 -ctv f16 -b 512 -ub 256 -p 128 -n 48
llama-bench -m /path/to/model.gguf -ngl 99 -fa on  -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -p 128 -n 48

# 3. Verify CUDA kernel launch counts, sync calls, and PCIe H2D/D2H bytes via NVIDIA Nsight Systems
nsys profile --trace=cuda --stats=true llama-bench -m /path/to/model.gguf -ngl 99 -fa on -p 0 -n 48 -r 1
```

---

## 2B. Two-Model Delta (`Δ`) & Mean Absolute Percentage Error (`MAPE`) vs. `llama-bench` & `nvidia-smi`

Both locally installed `.gguf` models — **`Qwen3.5-4B-Q4_K_M.gguf` (`2.82 GiB`, full GPU offload `ngl=24 → 99`)** and **`DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-Q4_K_M.gguf` (`5.38 GiB`, partial GPU offload `ngl=20 → 28` on 6 GiB VRAM)** — were benchmarked head-to-head against `/usr/bin/llama-bench` (`-p 64 -n 48 -r 2`) and concurrent `/usr/bin/nvidia-smi` polling:

### Model 1: `Qwen3.5-4B-Q4_K_M.gguf` (`2.82 GiB`)

| Metric | `gpuMode` (`10 ms` NVML + Streaming HTTP) | External Reference (`llama-bench` / `nvidia-smi`) | Exact Delta (`Δ`) | Abs Error (`%`) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Decode TPOT (`ngl=24`)** | `40.65 ms/tok` | `39.99 ms/tok` (`llama-bench`) | `+0.66 ms/tok` | **`1.65%`** |
| **Optimized Decode TPOT (`ngl=99`)** | `21.06 ms/tok` | `20.35 ms/tok` (`llama-bench`) | `+0.71 ms/tok` | **`3.48%`** |
| **Baseline Throughput (`tok/s`)** | `24.60 tok/s` | `25.01 tok/s` (`llama-bench`) | `-0.41 tok/s` | **`1.63%`** |
| **Optimized Throughput (`tok/s`)** | `47.48 tok/s` | `49.13 tok/s` (`llama-bench`) | `-1.65 tok/s` | **`3.36%`** |
| **Decode Speedup Ratio (`Base → Opt`)** | `1.93×` | `1.96×` (`llama-bench`) | `-0.03×` | **`1.76%`** |
| **Baseline Active GPU Power (`W`)** | `34.65 W` | `33.83 W` (`nvidia-smi`) | `+0.82 W` | **`2.41%`** |
| **Optimized Active GPU Power (`W`)** | `53.45 W` | `49.89 W` (`nvidia-smi`) | `+3.56 W` | **`7.14%`** |
| **Baseline GPU SM Utilization (`%`)** | `35.51%` | `33.75%` (`nvidia-smi`) | `+1.76%` | **`5.21%`** |
| **Optimized GPU SM Utilization (`%`)** | `85.34%` | `75.03%` (`nvidia-smi`) | `+10.31%` | **`13.73%`** |
| **Model 1 Mean Error (MAPE)** | **Timing MAPE: `2.38%`** | **Power MAPE: `4.78%`** | — | **`4.49%` (`95.51%` accuracy)** |

### Model 2: `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-Q4_K_M.gguf` (`5.38 GiB`)

| Metric | `gpuMode` (`10 ms` NVML + Streaming HTTP) | External Reference (`llama-bench` / `nvidia-smi`) | Exact Delta (`Δ`) | Abs Error (`%`) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Decode TPOT (`ngl=20`)** | `80.66 ms/tok` | `73.90 ms/tok` (`llama-bench`) | `+6.76 ms/tok` | **`9.15%`** |
| **Optimized Decode TPOT (`ngl=28`)** | `50.64 ms/tok` | `48.92 ms/tok` (`llama-bench`) | `+1.73 ms/tok` | **`3.53%`** |
| **Baseline Throughput (`tok/s`)** | `12.40 tok/s` | `13.53 tok/s` (`llama-bench`) | `-1.13 tok/s` | **`8.38%`** |
| **Optimized Throughput (`tok/s`)** | `19.75 tok/s` | `20.44 tok/s` (`llama-bench`) | `-0.70 tok/s` | **`3.41%`** |
| **Decode Speedup Ratio (`Base → Opt`)** | `1.59×` | `1.51×` (`llama-bench`) | `+0.08×` | **`5.43%`** |
| **Baseline Active GPU Power (`W`)** | `15.90 W` | `16.30 W` (`nvidia-smi`) | `-0.40 W` | **`2.48%`** |
| **Optimized Active GPU Power (`W`)** | `42.45 W` | `41.18 W` (`nvidia-smi`) | `+1.27 W` | **`3.09%`** |
| **Baseline GPU SM Utilization (`%`)** | `29.25%` | `27.40%` (`nvidia-smi`) | `+1.85%` | **`6.75%`** |
| **Optimized GPU SM Utilization (`%`)** | `48.39%` | `44.98%` (`nvidia-smi`) | `+3.41%` | **`7.57%`** |
| **Model 2 Mean Error (MAPE)** | **Timing MAPE: `5.98%`** | **Power MAPE: `2.78%`** | — | **`5.53%` (`94.47%` accuracy)** |

* **Cross-Model Consistency:** In Optimized Mode, both the `4B` and `9B` models exhibit virtually identical TPOT error (**`3.48%`** vs. **`3.53%`** — a `0.05%` difference) and throughput error (**`3.36%`** vs. **`3.41%`**), representing the fixed ~`3.4%` HTTP Server-Sent Events (SSE) & detokenization overhead in `llama-server` vs. raw in-process `llama-bench`. Combined 2-model mean error across all 18 measurements is **`5.01%` (`94.99%` accuracy)**.

---

## 2C. Exact CUDA Kernel Hardware Timestamp Verification (`gpuMode` vs. `nsys` `CUPTI_ACTIVITY_KIND_KERNEL`)

Extracting the generated `.cu` binaries across 5 operations (`Batch=256, Hidden=4096, Iters=50, Passes=64`) and comparing `gpuMode`'s internal `cudaEventElapsedTime` median against `nsys` SQLite `CUPTI_ACTIVITY_KIND_KERNEL` start-to-end window timestamps on the **exact same process execution**:

| Operation | Variant | `gpuMode` `p50` (`ms`) | `nsys` CUPTI Window `p50` (`ms`) | `nsys` Pure Kernel Sum (`ms`) | Timing Delta (`Δ`) | **Hardware Accuracy** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`rmsnorm_residual`** | Baseline Unfused | `3.2255 ms` | `3.2234 ms` | `3.0912 ms` | `+0.0021 ms` (`2.1 µs`) | **`99.93%`** |
| **`rmsnorm_residual`** | Optimized Fused `float4` | `1.5327 ms` | `1.5306 ms` | `1.4875 ms` | `+0.0021 ms` (`2.1 µs`) | **`99.86%`** |
| **`rope_qkv`** | Baseline Unfused | `2.7954 ms` | `2.7915 ms` | `2.6620 ms` | `+0.0039 ms` (`3.9 µs`) | **`99.86%`** |
| **`rope_qkv`** | Optimized Fused `float4` | `0.7426 ms` | `0.7405 ms` | `0.6973 ms` | `+0.0021 ms` (`2.1 µs`) | **`99.71%`** |
| **`swiglu`** | Baseline Unfused | `4.4277 ms` | `4.4245 ms` | `4.2773 ms` | `+0.0032 ms` (`3.2 µs`) | **`99.93%`** |
| **`swiglu`** | Optimized Fused `float4` | `4.1040 ms` | `4.0909 ms` | `3.9948 ms` | `+0.0131 ms` (`13.1 µs`) | **`99.68%`** |
| **`layernorm`** | Baseline Unfused | `3.6677 ms` | `3.6603 ms` | `3.2814 ms` | `+0.0074 ms` (`7.4 µs`) | **`99.80%`** |
| **`layernorm`** | Optimized Fused `float4` | `3.2505 ms` | `3.0640 ms` | `2.7905 ms` | `+0.1865 ms` | **`93.91%`** |
| **`softmax`** | Baseline 3-Pass | `4.5036 ms` | `4.5002 ms` | `4.2497 ms` | `+0.0034 ms` (`3.4 µs`) | **`99.92%`** |
| **`softmax`** | Optimized Online `float4` | `1.7950 ms` | `1.7915 ms` | `1.7127 ms` | `+0.0035 ms` (`3.5 µs`) | **`99.81%`** |

---

## 3. CUDA Kernel Safety & Hardware Verification Proof

Every candidate CUDA C++ kernel generated by `gpuMode` must pass a strict hardware verification gate before being eligible for selection:

1. **Compiler Resource Inspection (`nvcc -Xptxas -v`):** Extracts exact per-thread register allocation, static/dynamic shared memory bytes, and ensures zero unintended local-memory register spills.
2. **Memory Safety (`compute-sanitizer --tool memcheck`):** Verifies `0` out-of-bounds accesses, misaligned `float4` loads, or shared-memory hazards.
3. **Multi-Shape FP64 Numerical Parity:** Validates output tensors against a double-precision (`FP64`) CPU reference across both aligned batches and odd unaligned dimensions (`hidden_dim = 1027`, non-divisible by 4) with max absolute error `< 1e-3` (measured `2.00e-07`).

### Sample Verified CLI Output

```text
================================================================================
 gpuMode — CUDA Hardware & Energy Verification Report (NVIDIA GeForce RTX 4050 Laptop GPU / sm_89)
================================================================================
Observability: Telemetry: NVML | Host-CUDA eBPF: MEASURED_EBPF (76391 launches, 38.8 µs avg gap, 433.38 ms sync stall, 41172.16 ms off-CPU)

1. Why is the GPU underutilized?
   [LAUNCH_BOUND] GPU SMs are starved at the host-device launch boundary: eBPF measured 38.8 µs inter-launch host gap.

2. Which operations are responsible?
   [ESTIMATED] RMSNorm + Residual Add (Unfused) accounts for 34.5% of memory-bound execution time.

3. What optimization helps?
   Selected RMSNorm + Residual — Fused Scalar + Warp Shuffle (Block 256) (40 registers/thread, 128 B shared mem),
   achieving a 2.02× measured speedup (2.268 ms → 1.121 ms).

4. Is that optimization numerically safe?
   VERIFIED SAFE: Passed compute-sanitizer memcheck (0 illegal accesses/races), standard batch numerical parity,
   and odd unaligned dimension (hidden_dim=1027) edge-case verification with max absolute error 2.00e-07 (< 1e-3).

5. How much energy does it save?
   [MEASURED] Reduces GPU board energy by 38.1% (34.66 J → 21.44 J per 1K inferences) and improves EDP by 3.38×.

6. Can fewer/cheaper GPUs serve the workload?
   [MEASURED] Peak activation VRAM drops by 8 MiB (20 → 12 MiB) and single-GPU capacity rises from 441 to 892 req/s
   (1,587,162 → 3,210,559 Requests/GPU-hr).
================================================================================
```

---

## 4. Automated CI, Self-Hosted GPU Runner & Mocked NVML Verification

Every push to `main` is verified across three GitHub Actions workflows:

| Workflow | Verified GitHub Actions Run | Scope & Assertions |
| :--- | :--- | :--- |
| **`ci.yml`** | [Run `#45060762811` (`success`)](https://github.com/Script-Kitty01/GPUmode/actions/runs/45060762811) | `gofmt`, `go vet ./...`, eBPF C bytecode compilation (`go generate ./internal/ebpftrace/...`), mocked `libnvidia-ml.so.1` C-ABI `dlopen` tests, failure-path tests, and `go test -v -race ./...` |
| **`cuda-hardware.yml`** | [Run `#45079974915` (`success` — Cloud + Self-Hosted RTX 4050)](https://github.com/Script-Kitty01/GPUmode/actions/runs/45079974915) | **Job 1 (Cloud):** Compiles all 8 CUDA operations (`24` kernels) with `nvcc -O3 -arch=sm_89 -Xptxas -v` inside `nvidia/cuda:12.6.3-devel-ubuntu24.04` (`0` spills). **Job 2 (Self-Hosted GPU):** Executes `./gpumode monitor`, `./gpumode check -op rmsnorm_residual`, `./gpumode check -op rope_qkv`, and live CUDA/NVML tests on physical RTX 4050 silicon |
| **`linux-compat.yml`** | [Run `#45060762909` (`success`)](https://github.com/Script-Kitty01/GPUmode/actions/runs/45060762909) | 5-distro container matrix (**Ubuntu 22.04 LTS**, **Ubuntu 24.04 LTS**, **Debian 12 Bookworm**, **Fedora 41**, **Arch Linux**) testing CGo/`dlopen` compilation, binary build, and mocked-NVML/failure-path test suites |

