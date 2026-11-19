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
    margin-bottom: 16px;
    align-items: center;
  }
  .t-item { display: flex; flex-direction: column; }
  .t-label { font-size: 0.72rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.04em; }
  .t-val { font-size: 1.1rem; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
  #sparkCanvas { width: 210px; height: 42px; background: #0d131f; border-radius: 6px; border: 1px solid var(--border); }

  /* Control Bar */
  .controls-grid {
    display: grid;
    grid-template-columns: 1.25fr 1.75fr;
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
  select, input[type="number"] {
    width: 100%;
    background: #0d131f;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 7px 9px;
    border-radius: 6px;
    font-size: 0.84rem;
  }
  .btn-row { display: flex; gap: 10px; align-items: center; margin-top: 6px; }
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
  .status-msg { font-size: 0.8rem; color: var(--amber); }

  /* Headline KPIs */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 18px;
  }
  .kpi-card {
    background: linear-gradient(145deg, #131a27, #172135);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .kpi-title { font-size: 0.76rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi-main { font-size: 1.55rem; font-weight: 800; margin: 4px 0; color: var(--green); font-variant-numeric: tabular-nums; }
  .kpi-sub { font-size: 0.78rem; color: var(--muted); }

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
    .telemetry-strip, .controls-grid, .kpi-grid, .charts-grid, .two-col {
      grid-template-columns: 1fr;
    }
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>gpu<span>Mode</span> — GPU Resource Intelligence & CUDA-to-Economics Engine</h1>
    <p>Closed-Loop Pipeline: Observe &rarr; Diagnose (Roofline) &rarr; Optimize (Bayesian/CUDA) &rarr; Verify (Sanitizer) &rarr; Measure Energy &rarr; Right-Size &rarr; Calculate Savings</p>
  </div>
  <div class="hw-badge" id="hwBadge">
    Detecting GPU Hardware Physics...
  </div>
</header>

<!-- Live SSE Telemetry Strip -->
<section class="telemetry-strip">
  <div class="t-item">
    <span class="t-label">Live GPU Power</span>
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

<!-- Controls: Live GPU Optimization Trigger + Custom Tariff & SLA Controls -->
<section class="controls-grid">
  <div class="card">
    <h2>1. Live CUDA Hardware Optimization & Verification</h2>
    <div class="form-row">
      <div>
        <label for="opSelect">Target Hotspot Operation</label>
        <select id="opSelect">
          <option value="rmsnorm_residual">RMSNorm + Residual Add (LLM Block)</option>
          <option value="rmsnorm">RMSNorm (Unfused vs float4)</option>
          <option value="layernorm">LayerNorm (Welford Warp-Shuffle)</option>
          <option value="softmax">Attention Softmax (3-Pass vs Online)</option>
          <option value="swiglu">SwiGLU Activation (SiLU * Gate)</option>
        </select>
      </div>
      <div>
        <label for="strategySelect">Autotuning Strategy</label>
        <select id="strategySelect">
          <option value="BAYESIAN_UCB">Bayesian Optimization (GP-LCB)</option>
          <option value="GRID_SEARCH">Exhaustive Grid Search</option>
        </select>
      </div>
    </div>
    <div class="form-row">
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

  <div class="card">
    <h2>2. Custom Hardware Tariff, Energy, Carbon &amp; SLA Parameters</h2>
    <div class="form-row">
      <div>
        <label for="rateGpu">GPU Rate ($/GPU-hr)</label>
        <input type="number" id="rateGpu" value="1.50" step="0.10" min="0.10">
      </div>
      <div>
        <label for="rateKwh">Electricity ($/kWh)</label>
        <input type="number" id="rateKwh" value="0.14" step="0.01" min="0.01">
      </div>
      <div>
        <label for="ratePue">Datacenter PUE</label>
        <input type="number" id="ratePue" value="1.30" step="0.05" min="1.0">
      </div>
      <div>
        <label for="rateCarbon">Carbon (gCO2e/kWh)</label>
        <input type="number" id="rateCarbon" value="385" step="10" min="10">
      </div>
      <div>
        <label for="targetQps">Target Traffic (QPS)</label>
        <input type="number" id="targetQps" value="350" step="25" min="10">
      </div>
      <div>
        <label for="slaMs">Latency SLA (ms)</label>
        <input type="number" id="slaMs" value="25.0" step="1.0" min="0.5">
      </div>
    </div>
    <div class="btn-row">
      <button class="btn-secondary" onclick="triggerRecompute()">Recalculate Economics &amp; Pareto Frontier</button>
      <span style="font-size:0.77rem;color:var(--muted);">Instantly updates $/1M inferences, Joules/1K, Pareto winners &amp; Digital Twin</span>
    </div>
  </div>
</section>

<!-- Headline Business Metrics -->
<section class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-title">Financial Efficiency ($ / 1M Inferences)</div>
    <div class="kpi-main" id="kpiCost">--</div>
    <div class="kpi-sub" id="kpiCostSub">Baseline: --</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Energy Efficiency (Joules / 1K Inferences)</div>
    <div class="kpi-main" id="kpiJoules">--</div>
    <div class="kpi-sub" id="kpiJoulesSub">Baseline: --</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Hardware Efficiency (Requests / GPU-Hour)</div>
    <div class="kpi-main" id="kpiReqHr">--</div>
    <div class="kpi-sub" id="kpiReqHrSub">Speedup: --</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Verified Monthly Infrastructure Savings</div>
    <div class="kpi-main" id="kpiMonthlySave">--</div>
    <div class="kpi-sub" id="kpiMonthlySub">--</div>
  </div>
</section>

<!-- 7-Step Causal Chain Breakdown -->
<section class="card" style="margin-bottom:18px;">
  <h2>
    <span>The 7-Step CUDA-to-Economics Causal Chain</span>
    <span class="badge badge-blue" id="reportMetaBadge">Hardware-Verified on sm_89</span>
  </h2>
  <div class="causal-grid">
    <div class="q-card">
      <div class="q-title">1. Why is the GPU underutilized?</div>
      <div class="q-body" id="q1Text">--</div>
    </div>
    <div class="q-card">
      <div class="q-title">2. Which operations are responsible?</div>
      <div class="q-body" id="q2Text">--</div>
    </div>
    <div class="q-card">
      <div class="q-title">3. What CUDA optimization helps?</div>
      <div class="q-body" id="q3Text">--</div>
    </div>
    <div class="q-card safe">
      <div class="q-title">4. Is that optimization numerically safe?</div>
      <div class="q-body" id="q4Text">--</div>
    </div>
    <div class="q-card">
      <div class="q-title">5. How much energy does it save?</div>
      <div class="q-body" id="q5Text">--</div>
    </div>
    <div class="q-card">
      <div class="q-title">6. Can fewer/cheaper GPUs serve the workload?</div>
      <div class="q-body" id="q6Text">--</div>
    </div>
    <div class="q-card money">
      <div class="q-title">7. How much money does the business save monthly?</div>
      <div class="q-body" id="q7Text">--</div>
    </div>
  </div>
</section>

<!-- Visual Roofline Model & Pareto Frontier Plots -->
<section class="charts-grid">
  <div class="card">
    <h2>
      <span>Roofline Performance Model (Log-Log AI vs TFLOPS)</span>
      <span class="badge badge-amber" id="rooflineBadge">MEMORY_BOUND</span>
    </h2>
    <canvas id="rooflineCanvas" class="plot" width="560" height="250"></canvas>
  </div>
  <div class="card">
    <h2>
      <span>Multi-Objective Pareto Frontier (Latency ms vs Joules / 1K Req)</span>
      <span class="badge badge-green" id="tuningBadge">Bayesian UCB</span>
    </h2>
    <canvas id="paretoCanvas" class="plot" width="560" height="250"></canvas>
  </div>
</section>

<!-- Multi-Objective Pareto Operating Modes & Autotuner Trials -->
<section class="two-col">
  <div class="card">
    <h2>Multi-Objective Pareto Recommendations (4 Business Modes)</h2>
    <table>
      <thead>
        <tr>
          <th>Operating Mode</th>
          <th>Winning Kernel Configuration</th>
          <th>Latency (p50)</th>
          <th>Joules / 1K</th>
          <th>$ / 1M Req</th>
        </tr>
      </thead>
      <tbody id="paretoModesBody"></tbody>
    </table>
  </div>

  <div class="card">
    <h2>Bayesian / Grid Autotuner Candidate Verification Log</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Kernel Variant</th>
          <th>Latency</th>
          <th>BW (GB/s)</th>
          <th>Regs / SMem</th>
          <th>Parity &amp; Edge (1027)</th>
        </tr>
      </thead>
      <tbody id="trialsBody"></tbody>
    </table>
  </div>
</section>

<!-- Digital Twin SKU Right-Sizing & DVFS Power-Clock Policies -->
<section class="two-col">
  <div class="card">
    <h2>GPU Digital Twin &amp; Fleet Right-Sizing Simulator</h2>
    <table>
      <thead>
        <tr>
          <th>Scenario</th>
          <th>Latency</th>
          <th>Fleet Cap (RPS)</th>
          <th>VRAM Fit</th>
          <th>Monthly Cost</th>
          <th>Unlocked By</th>
          <th>Decision</th>
        </tr>
      </thead>
      <tbody id="twinBody"></tbody>
    </table>
  </div>

  <div class="card">
    <h2>Power-Aware DVFS Execution &amp; Workload Bin-Packing Scheduler</h2>
    <table>
      <thead>
        <tr>
          <th>DVFS Clock Policy</th>
          <th>Target Clock</th>
          <th>Est. Power</th>
          <th>Throughput</th>
          <th>Joules / 1K</th>
          <th>SLA</th>
        </tr>
      </thead>
      <tbody id="dvfsBody"></tbody>
    </table>
    <div id="scheduleBox" style="margin-top:12px;padding:10px;background:var(--panel-alt);border-radius:6px;font-size:0.8rem;"></div>
  </div>
</section>

<!-- Verified CUDA C++ Kernel Comparison & Drop-In PyTorch Extension -->
<section class="two-col">
  <div class="card">
    <h2>Verified Winning CUDA C++ Kernel (sm_89)</h2>
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

function fmtNum(v, d = 2) {
  if (v === undefined || v === null || isNaN(v)) return "--";
  return Number(v).toFixed(d);
}

async function loadInitialState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    currentState = data;
    renderHardware(data.hardware);
    if (data.economics) {
      document.getElementById("rateGpu").value = data.economics.gpuHourlyRateUsd;
      document.getElementById("rateKwh").value = data.economics.electricityRateKwh;
      document.getElementById("ratePue").value = data.economics.datacenterPue;
      document.getElementById("rateCarbon").value = data.economics.carbonIntensityGco2;
      document.getElementById("targetQps").value = data.economics.targetQps;
      document.getElementById("slaMs").value = data.economics.latencySlaMs;
    }
    if (data.telemetryHistory && data.telemetryHistory.length) {
      data.telemetryHistory.forEach(s => updateTelemetryUI(s, false));
      drawSparkline();
    }
    if (data.hasReport && data.latestReport) {
      renderReport(data.latestReport);
    } else {
      // Automatically run an initial hardware-verified report if store is empty
      triggerOptimize();
    }
  } catch (e) {
    console.error("Failed to load initial state:", e);
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
}

function updateTelemetryUI(s, redraw = true) {
  if (!s) return;
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

function readEconomicsForm() {
  return {
    gpuHourlyRateUsd: parseFloat(document.getElementById("rateGpu").value) || 1.50,
    electricityRateKwh: parseFloat(document.getElementById("rateKwh").value) || 0.14,
    datacenterPue: parseFloat(document.getElementById("ratePue").value) || 1.30,
    serverOverheadWatts: 35.0,
    carbonIntensityGco2: parseFloat(document.getElementById("rateCarbon").value) || 385.0,
    targetQps: parseFloat(document.getElementById("targetQps").value) || 350.0,
    latencySlaMs: parseFloat(document.getElementById("slaMs").value) || 25.0,
    monthlyOperatingHrs: 730.0,
    alphaCostWeight: 0.45,
    betaEnergyWeight: 0.35,
    gammaMemoryWeight: 0.20
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
      iterations: parseInt(document.getElementById("iterInput").value) || 50,
      economics: readEconomicsForm()
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

async function triggerRecompute() {
  const status = document.getElementById("jobStatus");
  try {
    const res = await fetch("/api/recompute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readEconomicsForm())
    });
    if (res.ok) {
      const rep = await res.json();
      if (rep && rep.id) {
        renderReport(rep);
        status.textContent = "Economics & Pareto frontier updated!";
      }
    }
  } catch (e) {
    status.textContent = "Recompute error: " + e.message;
  }
}

function renderReport(rep) {
  if (!rep || !rep.economicsComparison) return;
  const ec = rep.economicsComparison;

  // Headline KPIs
  document.getElementById("kpiCost").textContent = "$" + fmtNum(ec.optimized.costPer1mInferencesUsd, 4);
  document.getElementById("kpiCostSub").textContent =
    "Baseline: $" + fmtNum(ec.baseline.costPer1mInferencesUsd, 4) +
    " (-" + fmtNum(ec.costPer1mReductionPct, 1) + "%)";

  document.getElementById("kpiJoules").textContent = fmtNum(ec.optimized.joulesPer1kInferences, 1) + " J";
  document.getElementById("kpiJoulesSub").textContent =
    "Baseline: " + fmtNum(ec.baseline.joulesPer1kInferences, 1) +
    " J (-" + fmtNum(ec.energySavedPct, 1) + "% GPU energy)";

  document.getElementById("kpiReqHr").textContent = Number(Math.round(ec.optimized.requestsPerGpuHour)).toLocaleString();
  document.getElementById("kpiReqHrSub").textContent =
    "Baseline: " + Number(Math.round(ec.baseline.requestsPerGpuHour)).toLocaleString() +
    " (" + fmtNum(ec.speedupFactor, 2) + "x faster)";

  document.getElementById("kpiMonthlySave").textContent = "$" + fmtNum(ec.monthlySavingsUsd, 2) + "/mo";
  document.getElementById("kpiMonthlySub").textContent =
    "$" + fmtNum(ec.annualSavingsUsd, 0) + "/yr • " +
    fmtNum(ec.monthlyKwhSaved, 1) + " kWh & " +
    fmtNum(ec.monthlyCarbonSavedKgCo2e, 1) + " kg CO2e saved";

  // 7 Questions
  const sa = rep.sevenAnswers || {};
  document.getElementById("q1Text").textContent = sa.q1WhyUnderutilized || "--";
  document.getElementById("q2Text").textContent = sa.q2WhichOpsResponsible || "--";
  document.getElementById("q3Text").textContent = sa.q3WhatOptimizationHelps || "--";
  document.getElementById("q4Text").textContent = sa.q4IsItNumericallySafe || "--";
  document.getElementById("q5Text").textContent = sa.q5HowMuchEnergySaved || "--";
  document.getElementById("q6Text").textContent = sa.q6CanCheaperGpuServe || "--";
  document.getElementById("q7Text").textContent = sa.q7MonthlyMoneySaved || "--";

  document.getElementById("rooflineBadge").textContent = rep.baselineRoofline.classification;
  document.getElementById("tuningBadge").textContent =
    rep.autotune.strategy + " (" + rep.autotune.bayesianEvaluations + "/" + rep.autotune.gridTotalConfigurations + " evals, -" +
    fmtNum(rep.autotune.tuningTimeSavedPct, 0) + "% tuning time)";

  // Pareto Modes Table
  const pr = rep.autotune.paretoRecommendations;
  const modes = [
    ["Performance Mode", pr.performance, "badge-blue"],
    ["Energy-Saving Mode", pr.energySaving, "badge-green"],
    ["Cost-Saving Mode", pr.costSaving, "badge-amber"],
    ["Balanced (J = αT+βE+γM)", pr.balanced, "badge-blue"]
  ];
  const pBody = document.getElementById("paretoModesBody");
  pBody.innerHTML = modes.map(([label, c, bcls]) => {
    if (!c) return "";
    return "<tr>" +
      "<td><span class='badge " + bcls + "'>" + label + "</span></td>" +
      "<td>" + c.name + "</td>" +
      "<td>" + fmtNum(c.latencyMs, 3) + " ms</td>" +
      "<td>" + fmtNum(c.joulesPer1kRequests, 1) + " J</td>" +
      "<td>$" + fmtNum(c.costPer1mInferencesUsd, 4) + "</td>" +
      "</tr>";
  }).join("");

  // Trials Table
  const tBody = document.getElementById("trialsBody");
  const baseR = rep.autotune.baselineResult;
  let rowsHTML = "";
  if (baseR && baseR.spec) {
    rowsHTML += "<tr>" +
      "<td>Base</td>" +
      "<td>" + baseR.spec.name + "</td>" +
      "<td>" + fmtNum(baseR.medianLatencyMs, 3) + " ms</td>" +
      "<td>" + fmtNum(baseR.effectiveBwGbps, 1) + "</td>" +
      "<td>" + baseR.ptxas.registersPerThread + "r / " + baseR.ptxas.sharedMemBytes + "B</td>" +
      "<td><span class='badge badge-amber'>Baseline</span></td>" +
      "</tr>";
  }
  (rep.autotune.trials || []).forEach(tr => {
    const r = tr.result;
    const ok = r.compiledOk && r.parityPassed && r.edgeCasePassed;
    const isBest = r.spec.id === rep.autotune.bestResult.spec.id;
    rowsHTML += "<tr>" +
      "<td>#" + tr.iteration + "</td>" +
      "<td>" + r.spec.name + (isBest ? " <span class='badge badge-green'>WINNER</span>" : "") + "</td>" +
      "<td>" + fmtNum(r.medianLatencyMs, 3) + " ms</td>" +
      "<td>" + fmtNum(r.effectiveBwGbps, 1) + "</td>" +
      "<td>" + r.ptxas.registersPerThread + "r / " + r.ptxas.sharedMemBytes + "B</td>" +
      "<td><span class='badge " + (ok ? "badge-green" : "badge-red") + "'>" +
      (ok ? "PASS (err " + Number(r.maxAbsError).toExponential(1) + ")" : "FAIL") + "</span></td>" +
      "</tr>";
  });
  tBody.innerHTML = rowsHTML;

  // Digital Twin Table
  const twinBody = document.getElementById("twinBody");
  twinBody.innerHTML = (rep.digitalTwinScenarios || []).map(sc => {
    let bcls = "badge-blue";
    if (sc.decision === "RECOMMENDED") bcls = "badge-green";
    else if (sc.decision === "BASELINE") bcls = "badge-amber";
    else if (sc.decision.includes("EXCEEDED") || sc.decision.includes("INSUFFICIENT")) bcls = "badge-red";
    return "<tr>" +
      "<td><strong>" + sc.scenarioName + "</strong></td>" +
      "<td>" + fmtNum(sc.estimatedLatencyMs, 2) + " ms</td>" +
      "<td>" + fmtNum(sc.estimatedThroughputRps, 0) + " req/s</td>" +
      "<td>" + fmtNum(sc.requiredVramMib, 0) + " / " + fmtNum(sc.availableVramMib, 0) + " MiB</td>" +
      "<td>$" + fmtNum(sc.monthlyTotalCostUsd, 0) + "/mo</td>" +
      "<td>" + sc.unlockedByOptimization + "</td>" +
      "<td><span class='badge " + bcls + "'>" + sc.decision + "</span></td>" +
      "</tr>";
  }).join("");

  // DVFS Table
  const dvfsBody = document.getElementById("dvfsBody");
  dvfsBody.innerHTML = (rep.dvfsPolicies || []).map(p => {
    return "<tr>" +
      "<td>" + p.policyName + "</td>" +
      "<td>" + fmtNum(p.targetSmClockMhz, 0) + " MHz</td>" +
      "<td>" + fmtNum(p.estimatedPowerWatts, 1) + " W (" + fmtNum(p.powerChangePct, 1) + "%)</td>" +
      "<td>" + fmtNum(p.estimatedThroughputRps, 0) + " req/s</td>" +
      "<td>" + fmtNum(p.joulesPer1kRequests, 1) + " J</td>" +
      "<td><span class='badge " + (p.meetsLatencySla ? "badge-green" : "badge-red") + "'>" +
      (p.meetsLatencySla ? "SLA OK" : "SLA VIOLATION") + "</span></td>" +
      "</tr>";
  }).join("");

  // Workload Bin-Packing & Predictive Autoscaler summary
  const sp = rep.workloadSchedule;
  const af = rep.autoscaleForecast;
  if (sp && af) {
    document.getElementById("scheduleBox").innerHTML =
      "<strong>Multi-Workload Bin-Packing Scheduler (J = &alpha;C + &beta;E + &gamma;L = " + fmtNum(sp.objectiveScoreJ, 2) + "):</strong> " +
      "Consolidates " + sp.separateGpusBefore + " separate inference workloads onto <strong>" + sp.consolidatedGpus + " GPU(s)</strong>, saving <strong>$" +
      fmtNum(sp.monthlyPackingSave, 0) + "/mo</strong>.<br>" +
      "<strong>Predictive Autoscaler (15-min Holt Forecast):</strong> Current " + fmtNum(af.currentTrafficRps, 0) +
      " req/s &rarr; Predicted " + fmtNum(af.predictedTraffic15Min, 0) + " req/s &bull; <em>" + af.action + "</em>";
  }

  // Code boxes
  if (rep.autotune.bestResult && rep.autotune.bestResult.spec) {
    document.getElementById("optCodeBox").textContent = rep.autotune.bestResult.spec.displayKernel;
  }
  document.getElementById("torchExtBox").textContent = rep.pytorchExtensionCode || "";

  drawRooflineChart(rep);
  drawParetoChart(rep);
}

function drawRooflineChart(rep) {
  const canvas = document.getElementById("rooflineCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 52, padR = 24, padT = 22, padB = 34;
  const pw = w - padL - padR, ph = h - padT - padB;

  const hw = rep.hardware;
  const peakFP32 = Math.max(1.0, hw.peakFp32Tflops || 12.0);
  const peakBW = Math.max(10.0, hw.peakMemBandwidthGbps || 192.0);

  // Log10 domain: AI from 0.05 to 250 FLOP/Byte; TFLOPS from 0.005 to peakFP32*1.4
  const minLogX = Math.log10(0.05), maxLogX = Math.log10(250);
  const minLogY = Math.log10(0.005), maxLogY = Math.log10(peakFP32 * 1.5);

  const toX = (ai) => padL + ((Math.log10(Math.max(0.05, ai)) - minLogX) / (maxLogX - minLogX)) * pw;
  const toY = (tf) => padT + ph - ((Math.log10(Math.max(0.005, tf)) - minLogY) / (maxLogY - minLogY)) * ph;

  // Axes
  ctx.strokeStyle = "#25324a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + ph);
  ctx.lineTo(padL + pw, padT + ph);
  ctx.stroke();

  // Draw Roofline Curve
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let i = 0; i <= 100; i++) {
    const logAI = minLogX + (i / 100) * (maxLogX - minLogX);
    const ai = Math.pow(10, logAI);
    const ceil = Math.min(peakFP32, (ai * peakBW) / 1000.0);
    const x = toX(ai), y = toY(ceil);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Ridge point vertical dashed guide
  const ridge = hw.ridgePointFlopsPerByte || (peakFP32 * 1000.0 / peakBW);
  const rx = toX(ridge);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#64748b";
  ctx.beginPath();
  ctx.moveTo(rx, padT);
  ctx.lineTo(rx, padT + ph);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#93a4c3";
  ctx.font = "11px sans-serif";
  ctx.fillText("Ridge (" + fmtNum(ridge, 1) + " FLOP/B)", Math.min(w - 130, rx + 5), padT + 14);
  ctx.fillText("Arithmetic Intensity (FLOPs / Byte)", padL + pw / 2 - 80, h - 8);

  // Plot Baseline and Optimized points
  (rep.rooflinePoints || []).forEach(pt => {
    const x = toX(pt.arithmeticIntensity);
    const y = toY(Math.max(0.006, pt.achievedTflops));
    ctx.fillStyle = pt.isOptimized ? "#22c55e" : "#f59e0b";
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e6edf7";
    ctx.fillText(pt.label + " (" + fmtNum(pt.achievedTflops * 1000, 1) + " GFLOPS)", x + 9, y + 4);
  });
}

function drawParetoChart(rep) {
  const canvas = document.getElementById("paretoCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const cands = (rep.autotune && rep.autotune.paretoCandidates) || [];
  if (!cands.length) return;

  const padL = 52, padR = 24, padT = 22, padB = 34;
  const pw = w - padL - padR, ph = h - padT - padB;

  const lats = cands.map(c => c.latencyMs);
  const joules = cands.map(c => c.joulesPer1kRequests);
  const minL = Math.min(...lats) * 0.85, maxL = Math.max(...lats) * 1.1;
  const minJ = Math.min(...joules) * 0.85, maxJ = Math.max(...joules) * 1.1;

  const toX = (l) => padL + ((l - minL) / Math.max(1e-6, maxL - minL)) * pw;
  const toY = (j) => padT + ph - ((j - minJ) / Math.max(1e-6, maxJ - minJ)) * ph;

  ctx.strokeStyle = "#25324a";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + ph);
  ctx.lineTo(padL + pw, padT + ph);
  ctx.stroke();

  ctx.fillStyle = "#93a4c3";
  ctx.font = "11px sans-serif";
  ctx.fillText("Latency (ms) — lower is better", padL + pw / 2 - 75, h - 8);

  cands.forEach(c => {
    const x = toX(c.latencyMs), y = toY(c.joulesPer1kRequests);
    ctx.fillStyle = c.isParetoOptimal ? "#22c55e" : "#64748b";
    ctx.beginPath();
    ctx.arc(x, y, c.isParetoOptimal ? 6 : 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = c.isParetoOptimal ? "#e6edf7" : "#93a4c3";
    const shortLabel = c.variant || c.name;
    ctx.fillText(shortLabel, Math.min(w - 115, x + 8), y + 4);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  loadInitialState();
  connectSSE();
});
</script>
</body>
</html>`
