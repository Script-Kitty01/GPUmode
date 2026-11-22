// Live NVML SSE telemetry, sparkline canvas, hardware physics badge, and eBPF boundary UI

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
    const graphCnt = t.graphLaunches ? " • " + t.graphLaunches + " graphs" : "";
    document.getElementById("eLaunches").textContent = (t.kernelLaunches || 0) + launchRate + graphCnt;
    const p95Gap = t.p95LaunchGapUs ? " / " + fmtNum(t.p95LaunchGapUs, 1) + " µs" : "";
    document.getElementById("eLaunchGap").textContent = fmtNum(t.avgLaunchGapUs, 1) + p95Gap;
    const geomEl = document.getElementById("eGeometry");
    if (geomEl) {
      geomEl.textContent = (t.avgGridBlocks ? fmtNum(t.avgGridBlocks, 0) : "--") + " blk × " +
        (t.avgBlockThreads ? fmtNum(t.avgBlockThreads, 0) : "--") + " thr";
    }
    document.getElementById("eMemcpy").textContent =
      fmtNum(t.memcpyH2dMib, 2) + " / " + fmtNum(t.memcpyD2hMib, 2) + " MiB";
    document.getElementById("eSyncStall").textContent =
      fmtNum(t.syncStallMs, 2) + " / " + fmtNum(t.offCpuMs, 2) + " ms";
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
  const empDram = hw.empiricalDramBandwidthGbps
    ? " (Emp DRAM: " + fmtNum(hw.empiricalDramBandwidthGbps, 1) + " GB/s, L2: " + fmtNum(hw.empiricalL2BandwidthGbps, 0) + " GB/s)"
    : "";
  document.getElementById("hwBadge").innerHTML =
    "<strong>" + escapeHtml(hw.name) + " (" + escapeHtml(hw.archFlag) + ")</strong> &bull; " +
    hw.smCount + " SMs &bull; " +
    fmtNum(hw.peakFp32Tflops, 1) + " TFLOPS FP32 &bull; " +
    fmtNum(hw.peakMemBandwidthGbps, 0) + " GB/s Peak BW" + escapeHtml(empDram) + " &bull; " +
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
  if (s.throttleReasons && s.throttleReasons.length > 0 && currentState && currentState.hardware) {
    document.getElementById("tRidge").textContent =
      fmtNum(currentState.hardware.ridgePointFlopsPerByte, 1) + " (" + s.throttleReasons[0] + ")";
  }

  if (s.powerWatts > 0) {
    powerHistory.push(s.powerWatts);
    if (powerHistory.length > 60) powerHistory.shift();
    if (redraw) drawSparkline();
  }
}

function drawSparkline() {
  const canvas = document.getElementById("sparkCanvas");
  if (!canvas) return;
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
