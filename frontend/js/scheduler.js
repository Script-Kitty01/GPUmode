// Section 3: Production GPU Fleet Cost, VRAM Placement, Admission Control & Multi-Strategy Simulator

async function loadFleetSchedule(updateEconomics) {
  const box = document.getElementById("fleetScheduleBox");
  if (!box) return;
  try {
    if (updateEconomics) {
      const econPayload = {
        targetQps: parseFloat(document.getElementById("schedQpsInput").value) || 120,
        latencySlaMs: parseFloat(document.getElementById("schedSloInput").value) || 25,
        gpuHourlyCostUsd: parseFloat(document.getElementById("schedHourlyUsdInput").value) || 0.75,
        spotDiscountPct: parseFloat(document.getElementById("schedSpotDiscInput").value) || 65,
        alphaCostWeight: 0.45,
        betaEnergyWeight: 0.35,
        gammaMemoryWeight: 0.20,
        electricityKwhUsd: 0.14
      };
      const recRes = await fetch("/api/recompute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(econPayload)
      });
      if (recRes.ok) {
        const updatedRep = await recRes.json();
        if (updatedRep && updatedRep.sevenAnswers) {
          renderReport(updatedRep);
        }
      }
    }

    const sel = typeof getSelectedModel === "function" ? getSelectedModel() : null;
    let schedReq = {};
    if (sel) {
      schedReq.modelName = sel.name;
      if (sel.sizeBytes > 0) {
        schedReq.modelSizeMib = sel.sizeBytes / (1024 * 1024);
      }
    }
    const res = await fetch("/api/schedule", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(schedReq)
    });
    if (!res.ok) throw new Error(await res.text());
    const rep = await res.json();
    renderFleetSchedule(rep);
  } catch (e) {
    box.textContent = "Schedule error: " + e.message;
  }
}

function renderFleetSchedule(rep) {
  const box = document.getElementById("fleetScheduleBox");
  if (!box || !rep) return;

  const od = rep.recommendedOnDemandPool || {};
  const sp = rep.recommendedSpotPool || {};
  const mp = rep.modelProfile || {};

  let html = "<div class='causal-grid' style='margin-bottom:12px;'>" +
    "<div class='q-card safe'>" +
      "<div class='q-title'>Recommended On-Demand Pool (" + escapeHtml(mp.modelName || "Model") + ")</div>" +
      "<div class='q-body'><strong>" + escapeHtml(od.poolName || "--") + "</strong> &bull; " +
      "VRAM Fit: <strong>" + (od.fitsFullVram ? "YES" : "PARTIAL") + " (" + fmtNum(od.vramUtilizationPct, 1) + "%)</strong> &bull; " +
      "Est TPOT: <strong>" + fmtNum(od.estimatedTpotMs, 1) + " ms/tok (" + fmtNum(od.estimatedTokensPerSec, 1) + " tok/s)</strong> &bull; " +
      "Cost: <strong>$" + fmtNum(od.costPer1mTokensUsd, 4) + " / 1M tok</strong> (" + (od.gpusRequiredForQps || 1) + " GPUs for target QPS)<br>" +
      "<span style='color:var(--muted);font-size:0.76rem;'>Tuning: " + escapeHtml(od.recommendedQuantization || "") + " | " + escapeHtml(od.recommendedBatchFlags || "") + "</span></div>" +
    "</div>" +
    "<div class='q-card money'>" +
      "<div class='q-title'>Recommended Spot / Preemptible Pool (Interruptible Batch)</div>" +
      "<div class='q-body'><strong>" + escapeHtml(sp.poolName || "--") + "</strong> &bull; " +
      "Spot Cost: <strong class='delta-good'>$" + fmtNum(sp.costPer1mTokensUsd, 4) + " / 1M tok</strong> &bull; " +
      "Cost / 1K Reqs: <strong>$" + fmtNum(sp.costPer1kRequestsUsd, 4) + "</strong><br>" +
      "<span style='color:var(--muted);font-size:0.76rem;'>Offloads BATCH_SPOT_INTERRUPTIBLE traffic onto Spot instances while keeping CRITICAL_REALTIME on primary On-Demand capacity.</span></div>" +
    "</div>" +
  "</div>";

  // Priority Admission Control & Backpressure Table
  html += "<div style='margin-top:8px;'><strong>Priority Admission Control &amp; Backpressure Gate:</strong></div>";
  html += "<table><thead><tr><th>Workload Priority</th><th>Admission Action</th><th>Assigned Pool</th><th>Queue Depth Gate</th><th>Est p95 vs SLO</th><th>Micro-Batch (-ub)</th><th>Policy Reason</th></tr></thead><tbody>";
  (rep.admissionPolicies || []).forEach(adm => {
    const actCls = adm.action === "ADMIT_ON_DEMAND" ? "badge-green" : (adm.action === "ROUTE_TO_SPOT_POOL" ? "badge-blue" : "badge-red");
    html += "<tr>" +
      "<td><strong>" + escapeHtml(adm.priority) + "</strong></td>" +
      "<td><span class='badge " + actCls + "'>" + escapeHtml(adm.action) + "</span></td>" +
      "<td><code>" + escapeHtml(adm.assignedPoolId) + "</code></td>" +
      "<td>" + adm.currentQueueDepth + " / " + adm.maxAllowedQueueDepth + "</td>" +
      "<td>" + fmtNum(adm.estimatedTotalP95Ms, 1) + " ms / " + fmtNum(adm.latencySloMs, 1) + " ms</td>" +
      "<td>" + (adm.dynamicMicroBatchSize || "--") + "</td>" +
      "<td>" + escapeHtml(adm.reason) + "</td>" +
      "</tr>";
  });
  html += "</tbody></table>";

  // Multi-Strategy Scheduling Simulator Table
  html += "<div style='margin-top:12px;'><strong>Multi-Strategy Fleet Scheduling Simulator (1,000-Request Trace):</strong></div>";
  html += "<table><thead><tr><th>Scheduling Strategy</th><th>SLO Compliance</th><th>End-to-End Latency (p50 / p95)</th><th>Spot Traffic Share</th><th>Backpressure Shed (HTTP 429)</th><th>Cost / 1M Tokens</th><th>Cost / 1K Reqs</th><th>Savings vs Naive</th></tr></thead><tbody>";
  (rep.strategyComparison || []).forEach(sim => {
    const isBest = sim.strategy === "COST_SLO_WEIGHTED_PARETO";
    html += "<tr>" +
      "<td><strong>" + escapeHtml(sim.strategy) + "</strong>" + (isBest ? " <span class='badge badge-green'>OPTIMAL</span>" : "") + "</td>" +
      "<td><strong>" + fmtNum(sim.sloCompliancePct, 1) + "%</strong></td>" +
      "<td>" + fmtNum(sim.p50EndToEndLatencyMs, 1) + " / " + fmtNum(sim.p95EndToEndLatencyMs, 1) + " ms</td>" +
      "<td>" + fmtNum(sim.spotTrafficSharePct, 1) + "% (" + sim.routedToSpotRequests + " reqs)</td>" +
      "<td>" + sim.backpressureShed429 + " reqs</td>" +
      "<td><strong>$" + fmtNum(sim.costPer1mTokensUsd, 4) + "</strong></td>" +
      "<td>$" + fmtNum(sim.costPer1kRequestsUsd, 4) + "</td>" +
      "<td><span class='" + (sim.costSavedVsNaivePct > 0 ? "delta-good" : "") + "'>-" + fmtNum(sim.costSavedVsNaivePct, 1) + "%</span></td>" +
      "</tr>";
  });
  html += "</tbody></table>";

  box.innerHTML = html;
}
