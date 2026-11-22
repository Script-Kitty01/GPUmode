// Shared dashboard state & formatting utilities
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
