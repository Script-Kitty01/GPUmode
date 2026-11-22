// Section 2: Local AI model discovery, direct .gguf runner, and Before/After workload comparison

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
  if (typeof loadFleetSchedule === "function") {
    loadFleetSchedule(false);
  }
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
      concurrency: parseInt((document.getElementById("llmConcurrency") || {}).value) || 1,
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
    if (typeof loadFleetSchedule === "function") {
      loadFleetSchedule(false);
    }
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
    ["Cost / 1M Tokens (On-Demand)", cell(b, r => "$" + fmtNum(r.costPer1mTokensUsd, 4)), cell(o, r => "$" + fmtNum(r.costPer1mTokensUsd, 4)), delta(r => r.costPer1mTokensUsd, "USD", true)],
    ["Cost / 1M Tokens (Spot -65%)", cell(b, r => "$" + fmtNum(r.spotCostPer1mTokensUsd, 4)), cell(o, r => "$" + fmtNum(r.spotCostPer1mTokensUsd, 4)), delta(r => r.spotCostPer1mTokensUsd, "USD", true)],
    ["Cost / 1K Requests", cell(b, r => "$" + fmtNum(r.costPer1kRequestsUsd, 4)), cell(o, r => "$" + fmtNum(r.costPer1kRequestsUsd, 4)), delta(r => r.costPer1kRequestsUsd, "USD", true)],
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
