#!/usr/bin/env python3
"""
gpuMode PyTorch Profiler Bridge
Uses torch.profiler to measure operator-level latency, CUDA kernel launch counts,
tensor shapes, and memory allocation when PyTorch is installed.
Explicitly labels all output with provenance ("MEASURED" vs "ESTIMATED") so
synthetic fallback traces are never misrepresented as hardware measurements.
"""
import argparse
import json
import sys


def _extract_attr(evt, *names, default=0):
    for name in names:
        if hasattr(evt, name):
            val = getattr(evt, name)
            if val is not None:
                return val
    return default


def profile_with_torch(batch: int, hidden: int, iters: int):
    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile, record_function

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    activities = [ProfilerActivity.CPU]
    if has_cuda:
        activities.append(ProfilerActivity.CUDA)

    x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    res = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    w = torch.ones(hidden, device=device, dtype=torch.float32)
    b = torch.zeros(hidden, device=device, dtype=torch.float32)
    gate = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    up = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    attn_scores = torch.randn(8, batch, min(hidden, 512), device=device, dtype=torch.float32)

    for _ in range(3):
        h = x + res
        _ = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * w
        _ = F.silu(gate) * up
        _ = torch.softmax(attn_scores, dim=-1)
        _ = F.layer_norm(h, (hidden,), w, b, 1e-5)
    if has_cuda:
        torch.cuda.synchronize()

    op_specs = [
        ("gpumode::rmsnorm_residual", "RMSNorm + Residual Add (Unfused)", "rmsnorm_residual"),
        ("gpumode::swiglu", "SwiGLU FFN Activation (SiLU * Gate)", "swiglu"),
        ("gpumode::softmax", "Attention Softmax", "softmax"),
        ("gpumode::layernorm", "Post-Attention LayerNorm", "layernorm"),
    ]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(iters):
            with record_function("gpumode::rmsnorm_residual"):
                h = x + res
                var = h.pow(2).mean(-1, keepdim=True)
                _ = h * torch.rsqrt(var + 1e-5) * w
            with record_function("gpumode::swiglu"):
                _ = F.silu(gate) * up
            with record_function("gpumode::softmax"):
                _ = torch.softmax(attn_scores, dim=-1)
            with record_function("gpumode::layernorm"):
                _ = F.layer_norm(h, (hidden,), w, b, 1e-5)
        if has_cuda:
            torch.cuda.synchronize()

    avgs = prof.key_averages()
    by_key = {evt.key: evt for evt in avgs}

    raw_ops = []
    total_time_us = 0.0
    for rec_key, label, category in op_specs:
        evt = by_key.get(rec_key)
        if evt is None:
            continue
        cuda_us = float(_extract_attr(evt, "device_time_total", "cuda_time_total", default=0.0))
        cpu_us = float(_extract_attr(evt, "cpu_time_total", default=0.0))
        effective_us = cuda_us if (has_cuda and cuda_us > 0) else cpu_us
        total_time_us += effective_us

        shapes = []
        raw_shapes = _extract_attr(evt, "input_shapes", default=[])
        if raw_shapes:
            shapes = [list(s) for s in raw_shapes[:3] if s]

        mem_bytes = int(_extract_attr(evt, "device_memory_usage", "cuda_memory_usage", "cpu_memory_usage", default=0))
        launches = int(_extract_attr(evt, "count", default=iters))

        raw_ops.append({
            "name": label,
            "category": category,
            "measuredLatencyMs": round((effective_us / max(1, iters)) / 1000.0, 4),
            "cudaTimeTotalUs": round(cuda_us, 2),
            "cpuTimeTotalUs": round(cpu_us, 2),
            "launches": launches,
            "cudaMemoryBytes": mem_bytes,
            "inputShapes": shapes,
            "effectiveUs": effective_us,
            "provenance": "MEASURED",
        })

    if total_time_us <= 0:
        raise RuntimeError("torch.profiler recorded zero total operator time")

    hotspots = []
    for item in raw_ops:
        share = round((item["effectiveUs"] / total_time_us) * 100.0, 2)
        del item["effectiveUs"]
        item["sharePct"] = share
        hotspots.append(item)

    hotspots.sort(key=lambda h: h["sharePct"], reverse=True)

    return {
        "torchAvailable": True,
        "torchVersion": str(torch.__version__),
        "device": device if not has_cuda else f"cuda:0 ({torch.cuda.get_device_name(0)})",
        "batchSize": batch,
        "hiddenDim": hidden,
        "iterations": iters,
        "provenance": "MEASURED",
        "sourceNote": f"Measured via torch.profiler ({'CUDA+CPU' if has_cuda else 'CPU'} activities, record_shapes=True, profile_memory=True).",
        "hotspots": hotspots,
    }


def fallback_trace(batch: int, hidden: int, reason: str):
    return {
        "torchAvailable": False,
        "device": "no-torch-fallback (analytical trace)",
        "batchSize": batch,
        "hiddenDim": hidden,
        "iterations": 0,
        "provenance": "ESTIMATED",
        "sourceNote": f"ESTIMATED trace ({reason}); install PyTorch with CUDA to enable live torch.profiler operator attribution.",
        "hotspots": [
            {
                "name": "RMSNorm + Residual Add (Unfused)",
                "category": "rmsnorm_residual",
                "sharePct": 34.5,
                "launches": 4,
                "measuredLatencyMs": 0.0,
                "cudaMemoryBytes": batch * hidden * 4 * 3,
                "inputShapes": [[batch, hidden], [batch, hidden]],
                "provenance": "ESTIMATED",
            },
            {
                "name": "SwiGLU FFN Activation (SiLU * Gate)",
                "category": "swiglu",
                "sharePct": 24.0,
                "launches": 3,
                "measuredLatencyMs": 0.0,
                "cudaMemoryBytes": batch * hidden * 4 * 3,
                "inputShapes": [[batch, hidden], [batch, hidden]],
                "provenance": "ESTIMATED",
            },
            {
                "name": "Attention Softmax (3-Pass)",
                "category": "softmax",
                "sharePct": 22.5,
                "launches": 3,
                "measuredLatencyMs": 0.0,
                "cudaMemoryBytes": batch * hidden * 4 * 2,
                "inputShapes": [[8, batch, min(hidden, 512)]],
                "provenance": "ESTIMATED",
            },
            {
                "name": "LayerNorm (Unfused Mean/Var)",
                "category": "layernorm",
                "sharePct": 19.0,
                "launches": 4,
                "measuredLatencyMs": 0.0,
                "cudaMemoryBytes": batch * hidden * 4 * 3,
                "inputShapes": [[batch, hidden]],
                "provenance": "ESTIMATED",
            },
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gpuMode PyTorch Operator Profiler")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    try:
        data = profile_with_torch(args.batch, args.hidden, args.iters)
    except Exception as exc:
        data = fallback_trace(args.batch, args.hidden, str(exc))
    sys.stdout.write(json.dumps(data) + "\n")
