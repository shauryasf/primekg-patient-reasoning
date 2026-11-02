#!/usr/bin/env node
/* Compares the browser engine with the pipeline's saved out-of-fold outputs for every patient. */
const fs = require("fs");
const path = require("path");
const E = require("./engine.js");
const here = __dirname;
const M = E.prepare(JSON.parse(fs.readFileSync(path.join(here, "model_data.json"), "utf8")));
const csv = (file) => {
  const lines = fs.readFileSync(path.join(here, "..", "outputs", "neurosymbolic", file), "utf8").trim().split("\n");
  const head = lines[0].split(",");
  return new Map(lines.slice(1).map((line) => {
    const cells = line.split(",");
    return [cells[0], Object.fromEntries(head.map((name, i) => [name, cells[i]]))];
  }));
};
const oof = csv("fusion/oof_risk_all.csv");
const logic = csv("logic/oof_risk.csv");
const preds = csv("fusion/patient_predictions.csv");
const worst = { risk: 0, rule: 0, p12: 0, p24: 0, p36: 0, p60: 0, month: 0, low: 0, high: 0, percentile: 0, pathRule: 0 };
let paths = 0, pathMismatch = 0;
const bump = (key, a, b) => { const d = Math.abs(a - b); if (d > worst[key]) worst[key] = d; };
for (const p of M.patients) {
  const r = E.predictCohort(M, p);
  bump("risk", r.lp, Number(oof.get(p.id).neurosymbolic_cox));
  bump("rule", r.rule, Number(logic.get(p.id).rl_mln));
  const row = preds.get(p.id);
  ["12", "24", "36", "60"].forEach((h, i) => bump("p" + h, r.progress[i], Number(row[`p_progress_${h}m`])));
  bump("month", r.month, Number(row.predicted_month));
  bump("low", r.low, Number(row.predicted_low));
  bump("high", r.high, Number(row.predicted_high));
  bump("percentile", r.percentile, Number(row.risk_percentile));
  for (const [index, , score] of p.top) {
    const rule = E.pathRule(M, index, r.values, p.f);
    paths += 1;
    if (Math.abs(rule.score - score) > 1e-4) pathMismatch += 1;
    bump("pathRule", rule.score, score);
  }
}
console.log("largest absolute difference from the pipeline over", M.patients.length, "patients:");
console.log(Object.fromEntries(Object.entries(worst).map(([k, v]) => [k, Number(v.toPrecision(3))])));
console.log(`path rule scores: ${paths - pathMismatch} of ${paths} match`);
