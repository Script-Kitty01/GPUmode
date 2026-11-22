// Main application bootstrap & initial state hydration

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
      triggerOptimize();
    }
    if (data.economics) {
      if (data.economics.targetQps) document.getElementById("schedQpsInput").value = data.economics.targetQps;
      if (data.economics.latencySlaMs) document.getElementById("schedSloInput").value = data.economics.latencySlaMs;
      if (data.economics.gpuHourlyCostUsd) document.getElementById("schedHourlyUsdInput").value = data.economics.gpuHourlyCostUsd;
      if (data.economics.spotDiscountPct) document.getElementById("schedSpotDiscInput").value = data.economics.spotDiscountPct;
    }
    renderWorkloadComparison();
    loadFleetSchedule(false);
  } catch (e) {
    console.error("Failed to load initial state:", e);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  loadInitialState();
  connectSSE();
});
