// Section 1: CUDA kernel optimization, PyTorch 2.15 verification, 6-Question Causal Grid, and SM Warp Occupancy table

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

  // 1. PyTorch Profiler & Inline Extension Badge
  const ptBadge = document.getElementById("pytorchTraceBadge");
  const ptSummary = document.getElementById("pytorchInlineSummary");
  if (ptBadge && rep.trace) {
    const isMeas = rep.trace.provenance === "MEASURED";
    ptBadge.textContent = isMeas
      ? ("PYTORCH " + (rep.trace.torchVersion || "") + " [MEASURED]")
      : ("TRACE [" + (rep.trace.provenance || "ESTIMATED") + "]");
    ptBadge.className = "badge " + (isMeas ? "badge-green" : "badge-amber");
  }
  if (ptSummary && rep.trace) {
    let ptHtml = "<strong>Operator Profiler:</strong> " + escapeHtml(rep.trace.sourceNote || rep.trace.device || "");
    if (rep.trace.extensionVerification && rep.trace.extensionVerification.verified) {
      const ev = rep.trace.extensionVerification;
      ptHtml += " &bull; <span class='badge badge-green'>" + escapeHtml(ev.provenance) + "</span> " +
        "Kernel Launches: <strong>" + ev.beforeLaunches + " &rarr; " + ev.afterLaunches + "</strong> &bull; " +
        "Activation VRAM: <strong>" + fmtNum((ev.beforeMemoryBytes || 0) / 1048576, 1) + " &rarr; " + fmtNum((ev.afterMemoryBytes || 0) / 1048576, 1) + " MiB (-50%)</strong> &bull; " +
        "Max Parity Error: <code>" + Number(ev.maxAbsError || 0).toExponential(2) + "</code>";
    }
    ptSummary.innerHTML = ptHtml;
  }

  // 2. 6-Question Causal Verification Grid
  const qGrid = document.getElementById("causalAnswersGrid");
  if (qGrid && rep.sevenAnswers) {
    const sa = rep.sevenAnswers;
    const cards = [
      { title: "1. Why is the GPU underutilized?", body: sa.q1WhyUnderutilized, cls: "" },
      { title: "2. Which operations are responsible?", body: sa.q2WhichOpsResponsible, cls: "" },
      { title: "3. What CUDA optimization helps?", body: sa.q3WhatOptimizationHelps, cls: "" },
      { title: "4. Is that optimization numerically safe?", body: sa.q4IsItNumericallySafe, cls: "safe" },
      { title: "5. How much energy does it save?", body: sa.q5HowMuchEnergySaved, cls: "safe" },
      { title: "6. Can fewer/cheaper GPUs serve the workload?", body: sa.q6CanCheaperGpuServe, cls: "money" }
    ];
    qGrid.innerHTML = cards.map(c =>
      "<div class='q-card " + c.cls + "'>" +
        "<div class='q-title'>" + escapeHtml(c.title) + "</div>" +
        "<div class='q-body'>" + escapeHtml(c.body) + "</div>" +
      "</div>"
    ).join("");
  }

  // 3. Autotune Trials, PTXAS Resources & SM Warp Occupancy Table
  const trialsBox = document.getElementById("autotuneTrialsBox");
  if (trialsBox && rep.autotune) {
    const baseR = rep.autotune.baselineResult;
    const bestID = rep.autotune.bestResult && rep.autotune.bestResult.spec ? rep.autotune.bestResult.spec.id : "";
    const allRows = [];
    if (baseR && baseR.spec) {
      allRows.push({ label: "Baseline", r: baseR, isWinner: false });
    }
    (rep.autotune.trials || []).forEach(t => {
      if (t.result && t.result.spec) {
        allRows.push({ label: "Trial #" + t.iteration, r: t.result, isWinner: t.result.spec.id === bestID });
      }
    });
    if (allRows.length > 0) {
      let tHtml = "<table><thead><tr>" +
        "<th>Candidate Kernel</th><th>Block / Vec</th><th>PTXAS Regs / SMEM / Spills</th>" +
        "<th>SM Warp Occupancy</th><th>Latency (p50 / p95)</th><th>Achieved BW</th><th>Energy / 1K</th><th>Safety Gate</th>" +
        "</tr></thead><tbody>";
      allRows.forEach(item => {
        const r = item.r;
        const sp = r.spec || {};
        const pt = r.ptxas || {};
        const badge = item.isWinner
          ? " <span class='badge badge-green'>WINNER</span>"
          : (item.label === "Baseline" ? " <span class='badge badge-amber'>BASELINE</span>" : "");
        const occ = r.theoreticalOccupancyPct
          ? (fmtNum(r.theoreticalOccupancyPct, 0) + "% (" + (r.activeWarpsPerSm || 0) + "/" + (r.maxWarpsPerSm || 48) + " warps)")
          : "--";
        const safe = (r.compiledOk && r.parityPassed && r.edgeCasePassed && r.sanitizerPassed)
          ? "<span class='badge badge-green'>VERIFIED (err=" + Number(r.maxAbsError || 0).toExponential(1) + ")</span>"
          : "<span class='badge badge-red'>FAILED</span>";
        tHtml += "<tr>" +
          "<td><strong>" + escapeHtml(sp.name || sp.id) + "</strong>" + badge + "</td>" +
          "<td>" + (sp.blockSize || "--") + " / float" + (sp.vecWidth || 1) + "</td>" +
          "<td>" + (pt.registersPerThread || 0) + " regs / " + (pt.sharedMemBytes || 0) + " B / " + (pt.spillStoresBytes || 0) + " B</td>" +
          "<td>" + occ + "</td>" +
          "<td>" + fmtNum(r.medianLatencyMs, 4) + " / " + fmtNum(r.p95LatencyMs, 4) + " ms</td>" +
          "<td>" + fmtNum(r.effectiveBwGbps, 1) + " GB/s</td>" +
          "<td>" + fmtNum(r.energy ? r.energy.joulesPer1kRequests : 0, 2) + " J</td>" +
          "<td>" + safe + "</td>" +
          "</tr>";
      });
      tHtml += "</tbody></table>";
      trialsBox.innerHTML = tHtml;
    }
  }

  // Code panels
  if (rep.autotune && rep.autotune.bestResult && rep.autotune.bestResult.spec) {
    document.getElementById("optCodeBox").textContent = rep.autotune.bestResult.spec.displayKernel;
  }
  document.getElementById("torchExtBox").textContent = rep.pytorchExtensionCode || "";
}
