package server

const dashboardHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>gpuMode — Intelligent GPU Infrastructure & CUDA-to-Economics Platform</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131a27;
    --panel-alt: #182132;
    --border: #25324a;
    --text: #e6edf7;
    --muted: #93a4c3;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
    padding: 18px 24px 48px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
  }
  .brand h1 {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .brand h1 span { color: var(--accent); }
  .brand p { color: var(--muted); font-size: 0.85rem; }
  .hw-badge {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    color: var(--text);
  }
  .hw-badge strong { color: var(--green); }

  /* Live Telemetry Bar */
  .telemetry-strip {
    display: grid;
    grid-template-columns: repeat(6, 1fr) 220px;
    gap: 10px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
    align-items: center;
  }
  .ebpf-strip {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 16px;
  }
  .ebpf-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.82rem;
  }
  .ebpf-hint {
    font-size: 0.76rem;
    color: var(--muted);
  }
  .ebpf-hint code {
    background: #0d131f;
    border: 1px solid var(--border);
    padding: 1px 5px;
    border-radius: 4px;
    color: var(--accent);
  }
  .ebpf-metrics {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }
  .t-item { display: flex; flex-direction: column; }
  .t-label { font-size: 0.72rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.04em; }
  .t-val { font-size: 1.1rem; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
  #sparkCanvas { width: 210px; height: 42px; background: #0d131f; border-radius: 6px; border: 1px solid var(--border); }

  /* Control Bar */
  .controls-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 14px;
    margin-bottom: 18px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 15px 16px;
  }
  .card h2 {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 10px;
    color: var(--text);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .form-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
    margin-bottom: 10px;
  }
  label { display: block; font-size: 0.74rem; color: var(--muted); margin-bottom: 4px; }
  select, input[type="number"], input[type="text"] {
    width: 100%;
    background: #0d131f;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 7px 9px;
    border-radius: 6px;
    font-size: 0.84rem;
  }
  .btn-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 6px; }
  button {
    background: var(--accent);
    color: #06101e;
    font-weight: 700;
    border: none;
    padding: 8px 15px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.84rem;
    transition: opacity 0.15s;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-secondary {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-green {
    background: var(--green);
    color: #05140b;
  }
  .delta-good { color: #4ade80; font-weight: 700; }
  .delta-warn { color: #fbbf24; font-weight: 700; }
  .status-msg { font-size: 0.8rem; color: var(--amber); }

  /* 7-Question Causal Chain */
  .causal-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
    gap: 10px;
    margin-top: 8px;
  }
  .q-card {
    background: var(--panel-alt);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 10px 12px;
  }
  .q-card.safe { border-left-color: var(--green); }
  .q-card.money { border-left-color: var(--purple); }
  .q-title { font-size: 0.78rem; font-weight: 700; color: var(--accent); margin-bottom: 4px; }
  .q-body { font-size: 0.81rem; color: var(--text); }

  /* Charts Row */
  .charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin: 18px 0;
  }
  canvas.plot {
    width: 100%;
    height: 250px;
    background: #0d131f;
    border: 1px solid var(--border);
    border-radius: 8px;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.81rem;
    margin-top: 8px;
  }
  th, td {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
  }
  th { color: var(--muted); font-weight: 600; font-size: 0.74rem; text-transform: uppercase; }
  .badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
  }
  .badge-green { background: rgba(34,197,94,0.18); color: #4ade80; }
  .badge-blue { background: rgba(56,189,248,0.18); color: #38bdf8; }
  .badge-amber { background: rgba(245,158,11,0.18); color: #fbbf24; }
  .badge-red { background: rgba(239,68,68,0.18); color: #f87171; }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 18px;
  }
  pre.code-box {
    background: #090d14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.75rem;
    overflow-x: auto;
    max-height: 290px;
    color: #cbd5e1;
  }
  @media (max-width: 1024px) {
    .telemetry-strip, .ebpf-metrics, .controls-grid, .charts-grid, .two-col {
      grid-template-columns: 1fr;
    }
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>gpu<span>Mode</span> — GPU Resource Intelligence &amp; CUDA Optimization Engine</h1>
    <p>Closed-Loop Pipeline: Observe &rarr; Diagnose (Roofline) &rarr; Optimize (Bayesian/CUDA) &rarr; Verify (Sanitizer) &rarr; Measure Energy &rarr; Right-Size</p>
  </div>
  <div class="hw-badge" id="hwBadge">
    Detecting GPU Hardware Physics...
  </div>
</header>

<!-- Live SSE Telemetry Strip -->
<section class="telemetry-strip">
  <div class="t-item">
    <span class="t-label">Live GPU Power <span class="badge badge-blue" id="telemetrySourceBadge">PROBING</span></span>
    <span class="t-val" id="tPower">-- W</span>
  </div>
  <div class="t-item">
    <span class="t-label">SM / Mem Clock</span>
    <span class="t-val" id="tClock">-- / -- MHz</span>
  </div>
  <div class="t-item">
    <span class="t-label">Temperature</span>
    <span class="t-val" id="tTemp">-- &deg;C</span>
  </div>
  <div class="t-item">
    <span class="t-label">SM / Mem Util</span>
    <span class="t-val" id="tUtil">--% / --%</span>
  </div>
  <div class="t-item">
    <span class="t-label">VRAM Used</span>
    <span class="t-val" id="tVram">-- / -- MiB</span>
  </div>
  <div class="t-item">
    <span class="t-label">Ridge Point</span>
    <span class="t-val" id="tRidge">-- FLOP/B</span>
  </div>
  <canvas id="sparkCanvas" width="210" height="42"></canvas>
</section>

<!-- Host <-> CUDA Boundary (eBPF + Scheduler) Readout -->
<section class="ebpf-strip" id="ebpfStrip">
  <div class="ebpf-head">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <strong>Host &harr; CUDA Boundary (eBPF + Scheduler)</strong>
      <span class="badge badge-blue" id="ebpfProvBadge">PROBING</span>
      <a href="/api/trace/perfetto" target="_blank" class="badge badge-green" style="text-decoration:none;">Export Perfetto JSON</a>
      <a href="/metrics" target="_blank" class="badge badge-blue" style="text-decoration:none;">Prometheus /metrics</a>
    </div>
    <span class="ebpf-hint" id="ebpfHint">Checking Linux eBPF uprobe &amp; scheduler capabilities...</span>
  </div>
  <div class="ebpf-metrics" id="ebpfMetrics" style="display:none;">
    <div class="t-item">
      <span class="t-label">Kernel Launches</span>
      <span class="t-val" id="eLaunches">--</span>
    </div>
    <div class="t-item">
      <span class="t-label">Inter-Launch CPU Gap</span>
      <span class="t-val" id="eLaunchGap">-- &micro;s</span>
    </div>
    <div class="t-item">
      <span class="t-label">PCIe H2D / D2H</span>
      <span class="t-val" id="eMemcpy">-- / -- MiB</span>
    </div>
    <div class="t-item">
      <span class="t-label">CUDA Sync Stall</span>
      <span class="t-val" id="eSyncStall">-- ms</span>
    </div>
    <div class="t-item">
      <span class="t-label">Scheduler Off-CPU</span>
      <span class="t-val" id="eOffCpu">-- ms</span>
    </div>
  </div>
</section>

<!-- Controls: Live GPU Optimization Trigger -->
<section class="controls-grid">
  <div class="card">
    <h2>1. Live CUDA Hardware Optimization &amp; Verification</h2>
    <div class="form-row">
      <div>
        <label for="opSelect">Target Hotspot Operation</label>
        <select id="opSelect">
          <option value="rmsnorm_residual">RMSNorm + Residual Add (LLM Block)</option>
          <option value="rmsnorm">RMSNorm (Unfused vs float4)</option>
          <option value="layernorm">LayerNorm (Welford Warp-Shuffle)</option>
          <option value="softmax">Attention Softmax (3-Pass vs Online)</option>
          <option value="swiglu">SwiGLU Activation (SiLU * Gate)</option>
          <option value="rope_qkv">Rotary Positional Embedding (Fused RoPE + QKV Bias)</option>
          <option value="qgemv_decode">Quantized Decode GEMV (Q8_0 Register Dequant + Dot)</option>
          <option value="gqa_attention">Grouped-Query Attention (FlashDecoding Split-K)</option>
        </select>
      </div>
      <div>
        <label for="strategySelect">Autotuning Strategy</label>
        <select id="strategySelect">
          <option value="BAYESIAN_UCB">Bayesian Optimization (GP-LCB)</option>
          <option value="GRID_SEARCH">Exhaustive Grid Search</option>
        </select>
      </div>
      <div>
        <label for="batchInput">Batch &times; Tokens</label>
        <input type="number" id="batchInput" value="256" min="16" max="2048" step="16">
      </div>
      <div>
        <label for="hiddenInput">Hidden Dim</label>
        <input type="number" id="hiddenInput" value="4096" min="512" max="8192" step="256">
      </div>
      <div>
        <label for="iterInput">Timed Runs</label>
        <input type="number" id="iterInput" value="50" min="20" max="300" step="10">
      </div>
    </div>
    <div class="btn-row">
      <button id="runOptBtn" onclick="triggerOptimize()">Compile, Verify &amp; Benchmark on GPU</button>
      <span class="status-msg" id="jobStatus"></span>
    </div>
  </div>
</section>

<!-- Local AI Model Selector & Without-Project vs With-Project Benchmark -->
<section class="card" style="margin-bottom:18px;">
  <h2>2. Local AI Model Runner &amp; Before/After GPU Usage Comparison <span class="badge badge-blue" id="workloadProvBadge">MEASURED END-TO-END</span></h2>
  <div class="form-row" style="grid-template-columns: 1.2fr 1fr auto; align-items: end;">
    <div>
      <label for="directModelPathInput">Direct Model Path (.gguf file or folder — runs directly without /models)</label>
      <input type="text" id="directModelPathInput" placeholder="e.g. ~/Downloads/model.gguf or /abs/path/to/model.gguf" oninput="onDirectPathInput()">
    </div>
    <div>
      <label for="modelDirInput">Optional Scan Folder / Repo Path (for auto-discovery)</label>
      <input type="text" id="modelDirInput" placeholder="Optional folder to scan (e.g. ~/AI/Models or /path/to/repo)">
    </div>
    <div>
      <button id="scanModelsBtn" class="btn-secondary" onclick="scanLocalModels()">Scan Models</button>
    </div>
  </div>
  <div class="form-row">
    <div style="grid-column: span 2;">
      <label for="modelSelect">Model Source / Detected Local AI Model</label>
      <select id="modelSelect" onchange="onModelSelectionChanged()"></select>
    </div>
    <div>
      <label for="llmUrl">Inference Server URL (when using external server)</label>
      <input type="text" id="llmUrl" value="http://127.0.0.1:8081">
    </div>
    <div>
      <label for="llmReqs">Streaming Requests</label>
      <input type="number" id="llmReqs" value="3" min="1" max="16">
    </div>
    <div>
      <label for="llmTokens">Max Output Tokens / Req</label>
      <input type="number" id="llmTokens" value="48" min="16" max="256">
    </div>
  </div>
  <div id="modelScanMeta" style="margin-bottom:8px;font-size:0.78rem;color:var(--muted);">
    Scanning local AI models...
  </div>
  <div class="btn-row">
    <button id="runBaselineBtn" class="btn-secondary" onclick="triggerWorkload('baseline')">Run Without Project (Baseline)</button>
    <button id="runOptimizedBtn" class="btn-green" onclick="triggerWorkload('optimized')">Run With Project (gpuMode)</button>
    <span class="status-msg" id="workloadStatus"></span>
  </div>
  <div id="workloadResultBox" style="margin-top:12px;font-size:0.82rem;color:var(--muted);">
    Select a detected local AI model above, run it <strong>Without Project</strong> to record baseline GPU usage, then run it <strong>With Project</strong> to apply gpuMode runtime &amp; CUDA kernel optimizations and compare results side-by-side.
  </div>
</section>

<!-- Verified CUDA C++ Kernel Comparison & Drop-In PyTorch Extension -->
<section class="two-col">
  <div class="card">
    <h2 id="optCodeHeading">Verified Winning CUDA C++ Kernel</h2>
    <pre class="code-box" id="optCodeBox">// Run optimization to view verified CUDA C++ kernel</pre>
  </div>
  <div class="card">
    <h2>Auto-Generated Drop-In PyTorch C++/CUDA Extension</h2>
    <pre class="code-box" id="torchExtBox">// Run optimization to generate drop-in torch.utils.cpp_extension binding</pre>
  </div>
</section>

<script>
let powerHistory = [];
let currentState = null;
let discoveredModels = [];
let comparisonsByModel = {};
let directPathKeyMap = {};
let latestCausalForWorkload = null;

function fmtNum(v, d = 2) {
  if (v === undefined || v === null || isNaN(v)) return "--";
  return Number(v).toFixed(d);
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadInitialState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    currentState = data;
    renderHardware(data.hardware);
    if (data.telemetryBackend) {
      updateTelemetryBadge(data.telemetryBackend);
    }
    if (data.ebpfStatus) {
      renderEBPFTrace(null, data.ebpfStatus);
    }
    if (data.telemetryHistory && data.telemetryHistory.length) {
      data.telemetryHistory.forEach(s => updateTelemetryUI(s, false));
      drawSparkline();
    }
    if (data.customModelDir) {
      document.getElementById("modelDirInput").value = data.customModelDir;
    }
    if (data.directModelPath) {
      document.getElementById("directModelPathInput").value = data.directModelPath;
    }
    if (data.comparisonsByModel) {
      comparisonsByModel = data.comparisonsByModel;
    }
    if (data.discoveredModels) {
      renderModelOptions(data.discoveredModels);
      if (data.directModelPath) {
        document.getElementById("modelSelect").value = "__direct_path__";
        onModelSelectionChanged();
      }
    }
    if (data.hasReport && data.latestReport) {
      latestCausalForWorkload = data.latestReport;
      renderReport(data.latestReport);
    } else {
      // Automatically run an initial hardware-verified report if store is empty
      triggerOptimize();
    }
    renderWorkloadComparison();
  } catch (e) {
    console.error("Failed to load initial state:", e);
  }
}

function renderModelOptions(disc) {
  const sel = document.getElementById("modelSelect");
  const meta = document.getElementById("modelScanMeta");
  if (!sel || !disc) return;

  const prevVal = sel.value;
  discoveredModels = disc.models || [];
  sel.innerHTML = "";

  discoveredModels.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    const parts = [m.name, "[" + (m.sourceLabel || m.source) + "]"];
    if (m.sizeHuman) parts.push(m.sizeHuman);
    if (m.quantization) parts.push(m.quantization);
    if (m.running) parts.push("RUNNING");
    opt.textContent = parts.join(" • ");
    sel.appendChild(opt);
  });

  const directOpt = document.createElement("option");
  directOpt.value = "__direct_path__";
  directOpt.textContent = "Direct Local Model Path (use Direct Model Path field above)";
  sel.appendChild(directOpt);

  const customOpt = document.createElement("option");
  customOpt.value = "__custom_url__";
  customOpt.textContent = "Custom External Inference URL (use Server URL field)";
  sel.appendChild(customOpt);

  if (prevVal && Array.from(sel.options).some(o => o.value === prevVal)) {
    sel.value = prevVal;
  } else if (discoveredModels.length > 0) {
    sel.value = discoveredModels[0].id;
  } else {
    sel.value = "__direct_path__";
  }

  const dirs = (disc.scannedDirs || []).join(", ") || "none";
  const notes = (disc.notes && disc.notes.length) ? (" • " + disc.notes.join("; ")) : "";
  meta.innerHTML =
    "Detected <strong>" + discoveredModels.length + "</strong> model(s) • Scanned: <code>" + escapeHtml(dirs) + "</code>" +
    " • Or enter any <code>.gguf</code> path directly above" + escapeHtml(notes);
  onModelSelectionChanged();
}

function getSelectedModel() {
  const sel = document.getElementById("modelSelect");
  if (!sel) return null;
  const id = sel.value;
  return discoveredModels.find(m => m.id === id) || null;
}

function onDirectPathInput() {
  const directVal = document.getElementById("directModelPathInput").value.trim();
  const sel = document.getElementById("modelSelect");
  if (directVal && sel && sel.value !== "__direct_path__") {
    sel.value = "__direct_path__";
  }
  onModelSelectionChanged();
}

function onModelSelectionChanged() {
  const sel = document.getElementById("modelSelect");
  const selVal = sel ? sel.value : "";
  const m = getSelectedModel();
  const urlInput = document.getElementById("llmUrl");
  if (selVal === "__direct_path__") {
    urlInput.disabled = true;
  } else if (m && m.endpointUrl) {
    urlInput.value = m.endpointUrl;
    urlInput.disabled = true;
  } else if (m && m.path) {
    urlInput.disabled = true;
  } else {
    urlInput.disabled = false;
  }
  renderWorkloadComparison();
}

async function scanLocalModels() {
  const btn = document.getElementById("scanModelsBtn");
  const dir = document.getElementById("modelDirInput").value.trim();
  const llmUrl = document.getElementById("llmUrl").value.trim();
  btn.disabled = true;
  try {
    const params = new URLSearchParams();
    params.set("dir", dir);
    if (llmUrl) params.set("llmUrl", llmUrl);
    const res = await fetch("/api/models?" + params.toString());
    if (!res.ok) throw new Error(await res.text());
    const disc = await res.json();
    renderModelOptions(disc);
  } catch (e) {
    document.getElementById("modelScanMeta").textContent = "Scan error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function updateTelemetryBadge(source) {
  const el = document.getElementById("telemetrySourceBadge");
  if (!el || !source) return;
  const upper = String(source).toUpperCase();
  el.textContent = upper;
  el.className = "badge " + (upper === "NVML" ? "badge-green" : "badge-blue");
}

function renderEBPFTrace(trace, fallbackStatus) {
  const badge = document.getElementById("ebpfProvBadge");
  const hint = document.getElementById("ebpfHint");
  const metrics = document.getElementById("ebpfMetrics");
  if (!badge || !hint || !metrics) return;

  let t = (trace && trace.provenance) ? trace : fallbackStatus;
  if (fallbackStatus && fallbackStatus.active && (!trace || !trace.active)) {
    t = fallbackStatus;
  }
  if (!t) return;

  const prov = t.provenance || "DISABLED";
  badge.textContent = prov;
  if (prov === "MEASURED_EBPF") {
    badge.className = "badge badge-green";
  } else if (prov === "UNAVAILABLE_UNPRIVILEGED") {
    badge.className = "badge badge-amber";
  } else {
    badge.className = "badge badge-blue";
  }

  if (t.active) {
    metrics.style.display = "grid";
    const launchRate = t.launchesPerSec > 0 ? " (" + fmtNum(t.launchesPerSec, 0) + "/s)" : "";
    document.getElementById("eLaunches").textContent = (t.kernelLaunches || 0) + launchRate;
    document.getElementById("eLaunchGap").textContent = fmtNum(t.avgLaunchGapUs, 1) + " µs";
    document.getElementById("eMemcpy").textContent =
      fmtNum(t.memcpyH2dMib, 2) + " / " + fmtNum(t.memcpyD2hMib, 2) + " MiB";
    document.getElementById("eSyncStall").textContent =
      fmtNum(t.syncStallMs, 2) + " ms (" + fmtNum(t.syncStallPct, 1) + "%)";
    document.getElementById("eOffCpu").textContent =
      fmtNum(t.offCpuMs, 2) + " ms (" + fmtNum(t.hostStarvationPct, 1) + "%)";
    hint.textContent = t.statusDetail || "Live libcuda.so.1 uprobes + sched_switch tracepoint active";
  } else {
    metrics.style.display = "none";
    if (prov === "UNAVAILABLE_UNPRIVILEGED") {
      hint.innerHTML =
        "Rootless eBPF capabilities not granted &mdash; run <code>./run.sh setcap</code> or execute with <code>sudo</code> to attach <code>libcuda.so.1</code> uprobes &amp; <code>sched_switch</code>.";
    } else {
      hint.textContent = t.statusDetail || "eBPF host-to-CUDA tracer disabled";
    }
  }
}

function renderHardware(hw) {
  if (!hw) return;
  document.getElementById("hwBadge").innerHTML =
    "<strong>" + hw.name + " (" + hw.archFlag + ")</strong> &bull; " +
    hw.smCount + " SMs &bull; " +
    fmtNum(hw.peakFp32Tflops, 1) + " TFLOPS FP32 &bull; " +
    fmtNum(hw.peakMemBandwidthGbps, 0) + " GB/s BW &bull; " +
    fmtNum(hw.totalVramMib, 0) + " MiB VRAM";
  document.getElementById("tRidge").textContent = fmtNum(hw.ridgePointFlopsPerByte, 1) + " FLOP/B";
  const heading = document.getElementById("optCodeHeading");
  if (heading && hw.archFlag) {
    heading.textContent = "Verified Winning CUDA C++ Kernel (" + hw.archFlag + ")";
  }
}

function updateTelemetryUI(s, redraw = true) {
  if (!s) return;
  if (s.source) {
    updateTelemetryBadge(s.source);
  }
  document.getElementById("tPower").textContent = fmtNum(s.powerWatts, 2) + " W";
  document.getElementById("tClock").textContent = fmtNum(s.smClockMhz, 0) + " / " + fmtNum(s.memClockMhz, 0) + " MHz";
  document.getElementById("tTemp").textContent = fmtNum(s.temperatureC, 1) + " °C";
  document.getElementById("tUtil").textContent = fmtNum(s.gpuUtilPct, 0) + "% / " + fmtNum(s.memUtilPct, 0) + "%";
  document.getElementById("tVram").textContent = fmtNum(s.vramUsedMib, 0) + " / " + fmtNum(s.vramTotalMib, 0) + " MiB";

  if (s.powerWatts > 0) {
    powerHistory.push(s.powerWatts);
    if (powerHistory.length > 60) powerHistory.shift();
    if (redraw) drawSparkline();
  }
}

function drawSparkline() {
  const canvas = document.getElementById("sparkCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (powerHistory.length < 2) return;

  const minP = Math.max(0, Math.min(...powerHistory) - 2);
  const maxP = Math.max(...powerHistory, minP + 10) + 2;

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  powerHistory.forEach((p, idx) => {
    const x = (idx / (powerHistory.length - 1)) * (w - 8) + 4;
    const y = h - 4 - ((p - minP) / (maxP - minP)) * (h - 8);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function connectSSE() {
  const es = new EventSource("/api/stream/telemetry");
  es.onmessage = (ev) => {
    try {
      const sample = JSON.parse(ev.data);
      updateTelemetryUI(sample, true);
    } catch (_) {}
  };
}

async function triggerOptimize() {
  const btn = document.getElementById("runOptBtn");
  const status = document.getElementById("jobStatus");
  btn.disabled = true;
  status.textContent = "Compiling CUDA candidates with nvcc -Xptxas -v & running energy-timed GPU benchmarks...";

  try {
    const payload = {
      opCategory: document.getElementById("opSelect").value,
      strategy: document.getElementById("strategySelect").value,
      batchSize: parseInt(document.getElementById("batchInput").value) || 256,
      hiddenDim: parseInt(document.getElementById("hiddenInput").value) || 4096,
      iterations: parseInt(document.getElementById("iterInput").value) || 50
    };
    const res = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const report = await res.json();
    renderReport(report);
    status.textContent = "Verified on GPU (" + fmtNum(report.economicsComparison.speedupFactor, 2) + "x speedup)";
  } catch (err) {
    status.textContent = "Error: " + err.message;
  } finally {
    btn.disabled = false;
  }
}

function renderReport(rep) {
  if (!rep) return;
  if (rep.telemetryBackend) {
    updateTelemetryBadge(rep.telemetryBackend);
  }
  renderEBPFTrace(rep.hostCudaTrace, currentState && currentState.ebpfStatus);

  // Code boxes
  if (rep.autotune && rep.autotune.bestResult && rep.autotune.bestResult.spec) {
    document.getElementById("optCodeBox").textContent = rep.autotune.bestResult.spec.displayKernel;
  }
  document.getElementById("torchExtBox").textContent = rep.pytorchExtensionCode || "";
}

async function triggerWorkload(mode) {
  const baseBtn = document.getElementById("runBaselineBtn");
  const optBtn = document.getElementById("runOptimizedBtn");
  const status = document.getElementById("workloadStatus");
  baseBtn.disabled = true;
  optBtn.disabled = true;

  const selVal = document.getElementById("modelSelect").value;
  const directPath = document.getElementById("directModelPathInput").value.trim();
  const selModel = getSelectedModel();

  if (selVal === "__direct_path__" && !directPath) {
    status.textContent = "Enter a direct .gguf file or model folder path first.";
    baseBtn.disabled = false;
    optBtn.disabled = false;
    document.getElementById("directModelPathInput").focus();
    return;
  }

  const modelLabel = selVal === "__direct_path__"
    ? directPath
    : (selModel ? selModel.name : document.getElementById("llmUrl").value);
  status.textContent = mode === "optimized"
    ? ("Running " + modelLabel + " WITH gpuMode (tuned runtime + CUDA kernel optimization)...")
    : ("Running " + modelLabel + " WITHOUT gpuMode (baseline inference)...");

  try {
    const payload = {
      mode: mode,
      customDir: document.getElementById("modelDirInput").value.trim(),
      serverUrl: document.getElementById("llmUrl").value.trim(),
      numRequests: parseInt(document.getElementById("llmReqs").value) || 3,
      maxTokens: parseInt(document.getElementById("llmTokens").value) || 48,
      autoServer: true,
      opCategory: document.getElementById("opSelect").value,
      strategy: document.getElementById("strategySelect").value,
      batchSize: parseInt(document.getElementById("batchInput").value) || 256,
      hiddenDim: parseInt(document.getElementById("hiddenInput").value) || 4096,
      iterations: parseInt(document.getElementById("iterInput").value) || 50
    };
    if (selVal === "__direct_path__") {
      payload.modelId = "__direct_path__";
      payload.directModelPath = directPath;
      payload.forceLocalSpawn = true;
    } else if (selVal === "__custom_url__") {
      payload.modelId = "__custom_url__";
      payload.autoServer = false;
    } else if (selModel) {
      payload.modelId = selModel.id;
      payload.modelName = selModel.name;
      payload.modelSource = selModel.source;
      if (selModel.path) {
        payload.modelPath = selModel.path;
        payload.forceLocalSpawn = true;
      }
      if (selModel.endpointUrl) {
        payload.serverUrl = selModel.endpointUrl;
        payload.autoServer = false;
      }
    }

    const res = await fetch("/api/workload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const w = data.workload;

    if (data.modelKey) {
      if (selVal === "__direct_path__" && directPath) {
        directPathKeyMap[directPath] = data.modelKey;
      }
      if (!comparisonsByModel[data.modelKey]) comparisonsByModel[data.modelKey] = {};
      if (data.baseline) comparisonsByModel[data.modelKey].baseline = data.baseline;
      if (data.optimized) comparisonsByModel[data.modelKey].optimized = data.optimized;
    }
    if (data.causalReport) {
      latestCausalForWorkload = data.causalReport;
      renderReport(data.causalReport);
    }
    if (w && w.hostCudaTrace && w.hostCudaTrace.active) {
      renderEBPFTrace(w.hostCudaTrace, currentState && currentState.ebpfStatus);
    }
    renderWorkloadComparison(data.modelKey);
    status.textContent = (mode === "optimized" ? "With Project" : "Without Project") +
      " measured: " + fmtNum(w.throughputTokensPerS, 1) + " tok/s, " +
      fmtNum(w.avgGpuUtilPct, 0) + "% avg GPU SM, " +
      fmtNum(w.measuredPeakVramMib, 0) + " MiB VRAM";
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    baseBtn.disabled = false;
    optBtn.disabled = false;
  }
}

function fmtDelta(baseVal, optVal, unit, lowerIsBetter) {
  if (baseVal === undefined || optVal === undefined || baseVal === null || optVal === null || baseVal === 0) {
    return "--";
  }
  const diff = optVal - baseVal;
  const pct = (diff / baseVal) * 100.0;
  const sign = diff > 0 ? "+" : "";
  const improved = lowerIsBetter ? (diff < -1e-6) : (diff > 1e-6);
  const cls = improved ? "delta-good" : (Math.abs(pct) < 1.0 ? "" : "delta-warn");
  return "<span class='" + cls + "'>" + sign + fmtNum(diff, 2) + (unit ? " " + unit : "") + " (" + sign + fmtNum(pct, 1) + "%)</span>";
}

function renderWorkloadComparison(explicitKey) {
  const box = document.getElementById("workloadResultBox");
  if (!box) return;

  const selEl = document.getElementById("modelSelect");
  const selVal = selEl ? selEl.value : "";
  const directPath = (document.getElementById("directModelPathInput") || {}).value
    ? document.getElementById("directModelPathInput").value.trim()
    : "";
  const sel = getSelectedModel();
  let key = explicitKey || "";
  if (!key) {
    if (selVal === "__direct_path__" && directPath && directPathKeyMap[directPath]) {
      key = directPathKeyMap[directPath];
    } else if (sel) {
      key = sel.id;
    }
  }
  let pair = key ? comparisonsByModel[key] : null;
  if (!pair && selVal !== "__direct_path__") {
    const keys = Object.keys(comparisonsByModel);
    if (keys.length > 0 && !sel) {
      pair = comparisonsByModel[keys[keys.length - 1]];
    }
  }
  const b = pair && pair.baseline ? pair.baseline : null;
  const o = pair && pair.optimized ? pair.optimized : null;

  if (!b && !o) {
    box.innerHTML = "Select a detected local AI model above, run it <strong>Without Project</strong> to record baseline GPU usage, then run it <strong>With Project</strong> to apply gpuMode runtime &amp; CUDA kernel optimizations and compare results side-by-side.";
    return;
  }

  const cell = (r, fn) => r ? fn(r) : "<span style='color:var(--muted);'>Not run yet</span>";
  const delta = (fn, unit, lowerBetter) => (b && o) ? fmtDelta(fn(b), fn(o), unit, lowerBetter) : "--";

  const rows = [
    ["Runtime Profile", cell(b, r => escapeHtml(r.runtimeProfile || "Baseline")), cell(o, r => escapeHtml(r.runtimeProfile || "gpuMode")), "--"],
    ["GPU SM Util (Avg / Peak)", cell(b, r => fmtNum(r.avgGpuUtilPct, 1) + "% / " + fmtNum(r.peakGpuUtilPct, 0) + "%"), cell(o, r => fmtNum(r.avgGpuUtilPct, 1) + "% / " + fmtNum(r.peakGpuUtilPct, 0) + "%"), delta(r => r.avgGpuUtilPct, "%", false)],
    ["Memory Ctrl Util (Avg / Peak)", cell(b, r => fmtNum(r.avgMemUtilPct, 1) + "% / " + fmtNum(r.peakMemUtilPct, 0) + "%"), cell(o, r => fmtNum(r.avgMemUtilPct, 1) + "% / " + fmtNum(r.peakMemUtilPct, 0) + "%"), delta(r => r.avgMemUtilPct, "%", false)],
    ["Peak VRAM Used", cell(b, r => fmtNum(r.measuredPeakVramMib, 0) + " MiB"), cell(o, r => fmtNum(r.measuredPeakVramMib, 0) + " MiB"), delta(r => r.measuredPeakVramMib, "MiB", true)],
    ["Power (Avg / Active)", cell(b, r => fmtNum(r.energy.avgPowerWatts, 1) + " W / " + fmtNum(r.energy.activePowerWatts, 1) + " W"), cell(o, r => fmtNum(r.energy.avgPowerWatts, 1) + " W / " + fmtNum(r.energy.activePowerWatts, 1) + " W"), delta(r => r.energy.avgPowerWatts, "W", true)],
    ["Throughput", cell(b, r => fmtNum(r.throughputTokensPerS, 2) + " tok/s"), cell(o, r => fmtNum(r.throughputTokensPerS, 2) + " tok/s"), delta(r => r.throughputTokensPerS, "tok/s", false)],
    ["Latency (TTFT / TPOT)", cell(b, r => fmtNum(r.medianTtftMs, 1) + " ms / " + fmtNum(r.medianTpotMs, 1) + " ms"), cell(o, r => fmtNum(r.medianTtftMs, 1) + " ms / " + fmtNum(r.medianTpotMs, 1) + " ms"), delta(r => r.medianTpotMs, "ms/tok", true)],
    ["Energy / 1K Tokens", cell(b, r => fmtNum(r.joulesPer1kTokens, 1) + " J"), cell(o, r => fmtNum(r.joulesPer1kTokens, 1) + " J"), delta(r => r.joulesPer1kTokens, "J", true)],
    ["eBPF Launches / Tok", cell(b, r => (r.hostCudaTrace && r.hostCudaTrace.active) ? fmtNum(r.launchesPerOutputToken, 1) : (r.hostCudaTrace ? r.hostCudaTrace.provenance : "--")), cell(o, r => (r.hostCudaTrace && r.hostCudaTrace.active) ? fmtNum(r.launchesPerOutputToken, 1) : (r.hostCudaTrace ? r.hostCudaTrace.provenance : "--")), (b && o && b.hostCudaTrace && b.hostCudaTrace.active && o.hostCudaTrace && o.hostCudaTrace.active) ? fmtDelta(b.launchesPerOutputToken, o.launchesPerOutputToken, "", true) : "--"],
    ["eBPF Sync Stall", cell(b, r => (r.hostCudaTrace && r.hostCudaTrace.active) ? fmtNum(r.hostCudaTrace.syncStallMs, 1) + " ms" : "--"), cell(o, r => (r.hostCudaTrace && r.hostCudaTrace.active) ? fmtNum(r.hostCudaTrace.syncStallMs, 1) + " ms" : "--"), (b && o && b.hostCudaTrace && b.hostCudaTrace.active && o.hostCudaTrace && o.hostCudaTrace.active) ? fmtDelta(b.hostCudaTrace.syncStallMs, o.hostCudaTrace.syncStallMs, "ms", true) : "--"]
  ];

  let html = "<table><thead><tr><th>Metric</th><th>Without Project (Baseline)</th><th>With Project (gpuMode)</th><th>Change</th></tr></thead><tbody>";
  rows.forEach(row => {
    html += "<tr><td><strong>" + row[0] + "</strong></td><td>" + row[1] + "</td><td>" + row[2] + "</td><td>" + row[3] + "</td></tr>";
  });
  html += "</tbody></table>";

  if (o && latestCausalForWorkload && latestCausalForWorkload.autotune && latestCausalForWorkload.economicsComparison) {
    const ec = latestCausalForWorkload.economicsComparison;
    const best = latestCausalForWorkload.autotune.bestResult || {};
    const spec = best.spec || {};
    html += "<div style='margin-top:10px;padding:10px;background:var(--panel-alt);border-left:3px solid var(--green);border-radius:6px;'>" +
      "<strong>gpuMode CUDA Kernel Optimization Applied (" + escapeHtml(latestCausalForWorkload.opCategory) + "):</strong> " +
      escapeHtml(spec.name || "Fused CUDA Kernel") + " &bull; " +
      "<strong>Speedup:</strong> " + fmtNum(ec.speedupFactor, 2) + "&times; &bull; " +
      "<strong>Energy Saved:</strong> " + fmtNum(ec.energySavedPct, 1) + "% &bull; " +
      "<strong>Kernel Activation VRAM:</strong> -" + fmtNum(ec.vramReductionMib, 0) + " MiB &bull; " +
      "<strong>Safety:</strong> " + (best.parityPassed && best.edgeCasePassed ? "PASS (Parity + Edge-Case" + (best.sanitizerPassed ? " + Sanitizer" : "") + ")" : "CHECK") +
      "</div>";
  }
  box.innerHTML = html;
}

window.addEventListener("DOMContentLoaded", () => {
  loadInitialState();
  connectSSE();
});
</script>
</body>
</html>`
