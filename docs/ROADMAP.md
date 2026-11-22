# Engineering Roadmap & Next-Gen Feature Specification

[← Back to README](../README.md) • [Architecture & Internals](./ARCHITECTURE.md) • [Startup Guide](./STARTUP_GUIDE.md) • [Verification & Proof](./VERIFICATION_AND_PROOF.md)

---

## Executive Summary

`gpuMode` already implements a closed-loop hardware optimization pipeline (`Observe → Diagnose → Optimize → Verify → Measure Energy → Right-Size`) powered by `10 ms` pure-Go `libnvidia-ml.so.1` (`NVML`) bindings, rootless Linux eBPF CUDA uprobes, `nvcc -Xptxas -v` + `compute-sanitizer` verification, Bayesian UCB kernel tuning, and a live SSE Web Dashboard.

To elevate `gpuMode` from a strong systems prototype into a production-grade GPU observability and kernel optimization platform, this roadmap specifies **five engineering tracks** that eliminate the remaining analytical approximations and static presets in the codebase:

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                         gpuMode Next-Gen Upgrade Matrix                           │
├───────────────────────────┬───────────────────────────┬───────────────────────────┤
│ Subsystem                 │ Current State             │ Next-Gen Target           │
├───────────────────────────┼───────────────────────────┼───────────────────────────┤
│ PyTorch Extension Loop    │ Generates C++ string;     │ Compiles via load_inline  │
│ (internal/profiler)       │ uses 0.54×/0.55× factors  │ & re-profiles in PyTorch  │
├───────────────────────────┼───────────────────────────┼───────────────────────────┤
│ LLM Workload Runner       │ Hardcoded baseline vs.    │ Bayesian flag autotuner   │
│ (internal/workload)       │ tuned CLI flag presets    │ + concurrent load (c=1..N)│
├───────────────────────────┼───────────────────────────┼───────────────────────────┤
│ Local GGUF Proposer       │ Emits 3 JSON numbers      │ Synthesizes CUDA C++ with │
│ (internal/optimizer)      │ to pick static templates  │ nvcc/sanitizer self-repair│
├───────────────────────────┼───────────────────────────┼───────────────────────────┤
│ CUDA Kernel Catalog       │ FP32 elementwise &        │ FP16/BF16, Fused RoPE,    │
│ (internal/optimizer)      │ reduction ops only        │ Q4_K/Q8_0 GEMV, GQA       │
├───────────────────────────┼───────────────────────────┼───────────────────────────┤
│ Roofline & Occupancy      │ Theoretical clock × bus   │ Empirical DRAM/L2 probe + │
│ (internal/physics)        │ ceilings; no occupancy    │ CUDA Occupancy API        │
├───────────────────────────┼───────────────────────────┼───────────────────────────┤
│ Host ↔ CUDA eBPF & NVML   │ Scalar launch counts &    │ Grid/block extraction,    │
│ (internal/ebpftrace,nvml) │ avg gap; passive NVML     │ histograms, Perfetto, DVFS│
└───────────────────────────┴───────────────────────────┴───────────────────────────┘
```

---

## Track 1: Advanced eBPF & NVML Hardware Observability

**Target Files:**
- [`internal/ebpftrace/bpf/cuda_trace.bpf.c`](../internal/ebpftrace/bpf/cuda_trace.bpf.c)
- [`internal/ebpftrace/tracer_linux.go`](../internal/ebpftrace/tracer_linux.go)
- [`internal/ebpftrace/types.go`](../internal/ebpftrace/types.go)
- [`internal/nvml/nvml_linux.go`](../internal/nvml/nvml_linux.go)
- [`internal/telemetry/telemetry.go`](../internal/telemetry/telemetry.go)

### 1.1 Extract `cuLaunchKernel` Grid/Block Geometries & Shared Memory via eBPF
Currently, `handle_cuda_launch` in [`cuda_trace.bpf.c`](../internal/ebpftrace/bpf/cuda_trace.bpf.c) only increments `kernel_launches` and accumulates `launch_gap_sum_ns`. Because `cuLaunchKernel` follows the standard x86_64 System V ABI:

```c
CUresult CUDAAPI cuLaunchKernel(
    CUfunction f,          // RDI (PT_REGS_PARM1)
    unsigned int gridDimX, // RSI (PT_REGS_PARM2)
    unsigned int gridDimY, // RDX (PT_REGS_PARM3)
    unsigned int gridDimZ, // RCX (PT_REGS_PARM4)
    unsigned int blockDimX,// R8  (PT_REGS_PARM5)
    unsigned int blockDimY,// R9  (PT_REGS_PARM6)
    unsigned int blockDimZ,// stack +8
    unsigned int sharedMemBytes, // stack +16
    CUstream hStream,      // stack +24
    void **kernelParams,
    void **extra
);
```

- **Implementation:**
  - Read `PT_REGS_PARM2` through `PT_REGS_PARM6` and use `bpf_probe_read_user` from `PT_REGS_SP(ctx)` to capture `blockDimZ` and `sharedMemBytes`.
  - Track threads-per-block distribution (`<=128`, `256`, `512`, `1024`), average grid wave size (`gridDimX * gridDimY * gridDimZ`), and dynamic shared-memory usage across closed-source or external processes (`llama-server`, `Ollama`, `vLLM`, `PyTorch`) with **zero application code changes**.

### 1.2 Log2 Latency Histograms & Perfetto (`chrome://tracing`) RingBuffer Export
Averages (`AvgLaunchGapUS`) hide tail latency spikes caused by Linux CPU preemption or page faults.

- **eBPF Log2 Histograms:**
  - Add fixed buckets (`__u64 launch_gap_hist_us[20]`, `__u64 sync_stall_hist_us[20]`) in `struct cuda_tgid_stats` inside [`cuda_trace.bpf.c`](../internal/ebpftrace/bpf/cuda_trace.bpf.c) to compute exact `p50`, `p95`, and `p99` inter-launch gaps and `cuStreamSynchronize` stalls in Go user space.
- **Perfetto Timeline Export (`BPF_MAP_TYPE_RINGBUF`):**
  - Stream timestamped events (`CUDA_LAUNCH`, `CUDA_MEMCPY_H2D`, `CUDA_MEMCPY_D2H`, `CUDA_SYNC_ENTER/EXIT`, `SCHED_OFF_CPU`) through a lockless eBPF ring buffer during benchmark windows.
  - Expose `GET /api/trace/perfetto` in [`internal/server/server.go`](../internal/server/server.go) to download a standard Trace Event JSON file viewable in [ui.perfetto.dev](https://ui.perfetto.dev).

### 1.3 Hook Modern CUDA Driver Symbols (`CUDA 11.8+` & CUDA Graphs)
Extend `DiscoverCUDALibraries` and probe attachment in [`tracer_linux.go`](../internal/ebpftrace/tracer_linux.go) to attach uprobes/uretprobes to:
- `cuLaunchKernelEx` / `cuLaunchKernelEx_ptsz` (used by modern PyTorch 2.x and FlashAttention kernels)
- `cuGraphLaunch` / `cuGraphLaunch_ptsz` (detects whether the target workload uses CUDA Graphs to bypass host launch overhead)
- `cuMemcpyHtoDAsync_v2` & `cuMemcpyDtoHAsync_v2`
- `cuCtxSynchronize` & `cudaDeviceSynchronize`

### 1.4 NVML Clock Throttle Diagnostics & Automated DVFS Power-Cap Sweep
- **Clock Throttle Reason Bitmask:**
  - Resolve `nvmlDeviceGetCurrentClocksEventReasons` (fallback `nvmlDeviceGetCurrentClocksThrottleReasons`) and `nvmlDeviceGetPcieThroughput` via `dlsym` in [`internal/nvml/nvml_linux.go`](../internal/nvml/nvml_linux.go).
  - Decode bitflags (`GpuIdle`, `ApplicationsClocksSetting`, `SwPowerCap`, `HwSlowdown`, `SwThermalSlowdown`, `HwThermalSlowdown`, `HwPowerBrakeSlowdown`) into `telemetry.Sample` so [`internal/telemetry/telemetry.go`](../internal/telemetry/telemetry.go) flags any thermally throttled benchmark run.
- **Automated DVFS Energy-Optimal Power Cap Sweep:**
  - Memory-bound LLM decode operations often achieve `>95%` of peak throughput at `70–75%` of TDP because SMs stall on GDDR6 memory transactions.
  - Add an opt-in privileged tuner (`gpumode dvfs-tune`) using `nvmlDeviceSetPowerManagementLimit` / `nvmlDeviceSetGpuLockedClocks` to sweep power caps and discover the Pareto-optimal Watt limit that minimizes `Joules / 1K tokens` while meeting the user's `p95` latency SLA.

---

## Track 2: End-to-End Live PyTorch `load_inline` Compilation & Re-Profiling

**Target Files:**
- [`scripts/profile_pytorch.py`](../scripts/profile_pytorch.py)
- [`internal/profiler/causal.go`](../internal/profiler/causal.go)
- [`internal/profiler/causal_test.go`](../internal/profiler/causal_test.go)

### 2.1 Current Gap
In [`internal/profiler/causal.go`](../internal/profiler/causal.go):
- `GeneratePyTorchExtension` produces C++/CUDA source code as a UI display artifact.
- `Pipeline.Execute` estimates optimized Roofline traffic using static analytical discounts:
  ```go
  if bestRes.Spec.Fused {
      optProfile.BytesTransferred = baseProfile.BytesTransferred * 0.54
      optProfile.ActivationVRAMMiB = baseProfile.ActivationVRAMMiB * 0.55
  }
  ```

### 2.2 Closed-Loop `torch.utils.cpp_extension.load_inline` Verification
1. **Pass Winning Kernel to Python Bridge:**
   - Extend [`scripts/profile_pytorch.py`](../scripts/profile_pytorch.py) with `--verify-extension-cu <path>` and `--op <category>`.
2. **JIT Compile & Hot-Swap in PyTorch:**
   - Use `torch.utils.cpp_extension.load_inline` to compile the winning fused CUDA kernel against the active PyTorch environment's `libtorch` ABI.
   - Verify tensor parity directly against `torch.nn.functional` in FP32/FP16 (`torch.allclose(..., atol=1e-3)`).
3. **Second `torch.profiler` Pass (`Before` vs. `After`):**
   - Run `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True, profile_memory=True)` on both the unfused baseline PyTorch graph and the fused custom extension.
   - Return measured `beforeLatencyMs`, `afterLatencyMs`, `beforeLaunches`, `afterLaunches`, `beforeCudaMemoryBytes`, and `afterCudaMemoryBytes` in JSON with `provenance: "MEASURED_PYTORCH_EXTENSION"`.

---

## Track 3: Empirical Hierarchical Roofline & CUDA Occupancy Engine

**Target Files:**
- [`internal/physics/physics.go`](../internal/physics/physics.go)
- [`internal/roofline/roofline.go`](../internal/roofline/roofline.go)
- [`internal/harness/harness.go`](../internal/harness/harness.go)
- [`internal/optimizer/catalog.go`](../internal/optimizer/catalog.go)
- [`internal/server/dashboard.go`](../internal/server/dashboard.go)

### 3.1 Startup Empirical Hardware Ceiling Micro-Probe
Theoretical bandwidth ($\text{MemClock} \times 2 \times \text{BusWidth} / 8$) typically overstates achievable GDDR6 bandwidth by `15–25%` due to DRAM refresh, ECC, and memory controller overhead, while L2 cache bandwidth is `3–5×` higher than DRAM.

- **Implementation in [`internal/physics/physics.go`](../internal/physics/physics.go):**
  - Extend `cudaDevQuerySource` with a `<45 ms` calibration suite executed once per GPU session:
    1. **Sustained DRAM STREAM Triad (`A[i] = B[i] + s * C[i]`):** Working set $= 4 \times \text{L2CacheBytes}$ using vectorized `float4` loads/stores → populates `EmpiricalDRAMBandwidthGBps`.
    2. **Sustained L2 Cache Read/Write Kernel:** Working set $= 0.5 \times \text{L2CacheBytes}$ → populates `EmpiricalL2BandwidthGBps`.
    3. **Sustained FP32 FMA & FP16 Tensor Core Loop:** Register-resident FMA chain → populates `EmpiricalFP32TFLOPS` and `EmpiricalTensorFP16TFLOPS`.
  - Plot **Hierarchical Roofline Ceilings** (L2 vs. DRAM bandwidth slopes; FP32 vs. Tensor Core ceilings) in [`internal/roofline/roofline.go`](../internal/roofline/roofline.go) and the Web Dashboard canvas in [`internal/server/dashboard.go`](../internal/server/dashboard.go).

### 3.2 Hardware Warp Occupancy via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
Currently, [`internal/harness/harness.go`](../internal/harness/harness.go) parses `RegistersPerThread`, `SharedMemBytes`, and `SpillStoresBytes` from `nvcc -Xptxas -v`, but does not compute how those resources constrain active warps per SM.

- **Implementation:**
  - In the generated C++ harness templates in [`internal/optimizer/catalog.go`](../internal/optimizer/catalog.go), call:
    ```cpp
    int activeBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, kernel_fn, BLOCK_SIZE, sharedMemBytes);
    ```
  - Compute and emit `activeWarpsPerSM`, `maxWarpsPerSM`, `theoreticalOccupancyPct`, and `occupancyLimiter` (`"REGISTERS"`, `"SHARED_MEMORY"`, `"BLOCK_SIZE"`, or `"MAX_BLOCKS_PER_SM"`) in `HarnessOutput`.
  - Feed `theoreticalOccupancyPct` into the Gaussian Process surrogate feature vector in [`internal/optimizer/autotuner.go`](../internal/optimizer/autotuner.go).

---

## Track 4: Expanded LLM CUDA Kernel Catalog & Self-Healing LLM Codegen

**Target Files:**
- [`internal/optimizer/catalog.go`](../internal/optimizer/catalog.go)
- [`internal/optimizer/deepseek_local.go`](../internal/optimizer/deepseek_local.go)
- [`internal/optimizer/optimizer_test.go`](../internal/optimizer/optimizer_test.go)

### 4.1 Mixed-Precision (`FP16` / `BF16`) Support
Real LLM inference runs in `FP16` (`__half`) or `BF16` (`__nv_bfloat16`) with `FP32` accumulation.
- Add `Precision` (`"fp32"`, `"fp16"`, `"bf16"`) to `harness.KernelSpec`.
- Add 128-bit packed `half2` / `uint4` (8 `fp16` elements per 128-bit load transaction) templates in [`internal/optimizer/catalog.go`](../internal/optimizer/catalog.go), doubling effective element throughput on memory-bound norms and activations while retaining FP32 warp-shuffle accumulation for numerical stability.

### 4.2 High-Impact LLM Operations to Add to `SupportedOperations()`

| Operation ID | LLM Bottleneck Addressed | Baseline Kernel | Optimized Candidate Space |
| :--- | :--- | :--- | :--- |
| **`rope_qkv`** | Separate Q/K bias + Rotary Positional Embedding (`sin`/`cos` table reads + complex rotation) | 2-pass unfused bias + scalar RoPE | Fused single-pass `float4`/`half2` RoPE with vectorized rotary pair loads |
| **`qgemv_decode`** | Single-token autoregressive decode (`batch = 1..8`) is strictly DRAM bandwidth-bound reading quantized weights | Dequantize `Q8_0`/`Q4_K` buffer to global FP32 + GEMV | On-the-fly register dequantization (`__dp4a` / warp-tile dot product) + `__shfl_down_sync` reduction |
| **`gqa_attention`** | Grouped-Query Attention decode with long KV caches underutilizes SMs when `num_kv_heads` is small | Standard single-block-per-head attention | FlashDecoding **Split-K** across KV sequence chunks + online log-sum-exp combine kernel |

### 4.3 Multi-Turn Self-Healing LLM CUDA Synthesis
Currently, `LocalDeepSeekProposer` in [`internal/optimizer/deepseek_local.go`](../internal/optimizer/deepseek_local.go) queries `llama-cli` for `{"blockSize": 256, "vecWidth": 4, "fused": true}` and maps those parameters onto pre-written templates.

- **Upgrade to Verified Code Synthesis & Repair:**
  1. Prompt the local GGUF model with the locked C++ benchmark harness interface, the target GPU's `GPUHardwareSpec` (SM count, L2 cache size, max registers/SM, compute capability), and the baseline `__global__` kernel.
  2. Allow the model to return a complete candidate `__global__` C++ kernel alongside launch configuration metadata.
  3. **Automated 3-Stage Verification & Repair Loop:**
     ```text
     LLM Proposal → nvcc Compile → compute-sanitizer memcheck → Standard + Odd-Shape (1027) FP64 Parity
           ▲              │                    │                                  │
           └──────────────┴────────────────────┴──────────────────────────────────┘
                    On Failure: Feed exact compiler stderr / sanitizer OOB address /
                    maxAbsError coordinate back to LLM (up to N=3 repair turns)
     ```

---

## Track 5: Bayesian Runtime Flag Autotuner, Concurrent Load Generator & Platform Tooling

**Target Files:**
- [`internal/workload/workload.go`](../internal/workload/workload.go)
- [`internal/optimizer/autotuner.go`](../internal/optimizer/autotuner.go)
- [`internal/server/server.go`](../internal/server/server.go)
- [`cmd/gpumode/main.go`](../cmd/gpumode/main.go)

### 5.1 Bayesian Autotuning of Inference Server Runtime Knobs
Currently, `buildLlamaServerArgs` in [`internal/workload/workload.go`](../internal/workload/workload.go) switches between two hardcoded flag lists (`baseline` vs. `optimized`).

- **Dynamic Search Space:**
  - Let the Bayesian UCB optimizer explore multi-dimensional server configurations on the user's actual `.gguf` model and GPU VRAM budget:
    - `gpuLayers` (`-ngl`): dynamic binary/Bayesian search up to VRAM capacity ceiling
    - `flashAttention` (`-fa`): `on` vs. `off`
    - `kvCacheTypeK` / `kvCacheTypeV` (`-ctk`, `-ctv`): `f16`, `q8_0`, `q5_0`, `q4_0`
    - `ubatchSize` (`-ub`): `128`, `256`, `512`, `1024`
- **Output Quality / Perplexity Safety Gate:**
  - Before accepting a quantized KV-cache configuration (`q8_0`, `q4_0`), evaluate deterministic greedy generation (`temperature = 0, seed = 42`) or prompt logprobs against the `f16` baseline and reject configurations whose token agreement / perplexity drift exceeds tolerance.

### 5.2 Concurrent Multi-Client Streaming Benchmark (`Concurrency = 1..32`)
Currently, `RunInferenceBenchmark` in [`internal/workload/workload.go`](../internal/workload/workload.go) executes requests serially in a single `for` loop.

- **Implementation:**
  - Add `Concurrency int` (`-concurrency` CLI flag and Web Dashboard selector) backed by a goroutine worker pool.
  - Measure continuous batching scaling curves:
    - **TTFT (`p50` / `p95` / `p99`)** vs. concurrent users
    - **TPOT (`ms/tok`)** degradation under batch contention
    - **Aggregate Throughput (`tok/s`)** and **Energy Efficiency (`Joules / 1K tokens`)** at saturation (batching often reduces `Joules / 1K tokens` by `2–4×` by amortizing weight loads across requests).

### 5.3 Prometheus `/metrics` Exporter & `gpumode check` CI Regression Gate
- **Prometheus Endpoint (`GET /metrics` in [`internal/server/server.go`](../internal/server/server.go)):**
  - Export real-time gauges and counters in Prometheus exposition format without external library bloat:
    - `gpumode_nvml_power_watts`, `gpumode_nvml_sm_clock_mhz`, `gpumode_nvml_mem_clock_mhz`, `gpumode_nvml_vram_used_mib`, `gpumode_nvml_throttle_reasons_bitmask`
    - `gpumode_ebpf_kernel_launches_total`, `gpumode_ebpf_launch_gap_avg_us`, `gpumode_ebpf_sync_stall_ms_total`, `gpumode_ebpf_off_cpu_ms_total`
- **CI/CD Hardware Regression Gate (`gpumode check` in [`cmd/gpumode/main.go`](../cmd/gpumode/main.go)):**
  - Compare live kernel benchmark results against a pinned baseline report in `.gpumode_data` and exit non-zero if:
    - `p95` latency regresses by `> X%`
    - `Joules / 1K requests` regresses by `> Y%`
    - `nvcc -Xptxas -v` introduces nonzero register spills (`SpillStoresBytes > 0`)
    - `compute-sanitizer` or `hidden_dim = 1027` parity fails.

---

## Phased Implementation Matrix

| Phase | Track | Primary Deliverables | Verification Command |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Track 1 (eBPF & NVML)** | `cuLaunchKernel` grid/block extraction, log2 histograms, Perfetto JSON export, `cuLaunchKernelEx` hooks, NVML throttle bitmasks | `./run.sh bpf-gen && ./run.sh ebpf -json` |
| **Phase 2** | **Track 2 (PyTorch Loop)** | Live `torch.utils.cpp_extension.load_inline` compilation and before/after `torch.profiler` verification in `scripts/profile_pytorch.py` | `python3 scripts/profile_pytorch.py --verify-extension` |
| **Phase 3** | **Track 3 (Roofline & Occupancy)** | `<45 ms` empirical DRAM/L2/TFLOPS startup probe + `cudaOccupancyMaxActiveBlocksPerMultiprocessor` warp occupancy metrics | `./run.sh optimize -op rmsnorm_residual -json` |
| **Phase 4** | **Track 4 (Kernels & Codegen)** | `FP16`/`BF16`, `rope_qkv`, `qgemv_decode`, `gqa_attention` kernels + multi-turn compiler/sanitizer LLM repair loop | `./run.sh deepseek -op rope_qkv -strategy BAYESIAN_UCB` |
| **Phase 5** | **Track 5 (Workload & Platform)** | Bayesian `llama-server` flag autotuner, concurrent load tester (`-concurrency`), `/metrics` endpoint, `gpumode check` CLI | `./run.sh workload -llm-reqs 16 && go test ./...` |
