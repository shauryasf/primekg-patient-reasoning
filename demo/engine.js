/* Browser inference for the neuro-symbolic model. Works from demo/model_data.json alone.
   demo/check_engine.js proves this file reproduces the pipeline's out-of-fold predictions. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ReasonerEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const KIND = { mut: 0, amp: 1, del: 2 };
  const KIND_NAME = ["mut", "amp", "del"];
  const HORIZONS = [12, 24, 36, 60];

  function prepare(M) {
    M.atomsByNode = new Map();
    M.atomsByEntrez = new Map();
    M.atoms.forEach((atom, index) => {
      atom.nodes.forEach((node) => {
        if (!M.atomsByNode.has(node)) M.atomsByNode.set(node, new Set());
        M.atomsByNode.get(node).add(index);
      });
      atom.pathGenes.forEach((gene) => {
        if (!M.atomsByEntrez.has(gene)) M.atomsByEntrez.set(gene, new Set());
        M.atomsByEntrez.get(gene).add(index);
      });
    });
    M.geneIndex = new Map(M.genes.map((gene, i) => [gene, i]));
    M.pathsByGene = new Map();
    M.paths.forEach((path, index) => {
      if (!M.pathsByGene.has(path.g)) M.pathsByGene.set(path.g, []);
      M.pathsByGene.get(path.g).push(index);
    });
    return M;
  }

  function atomValues(M, input) {
    const altered = new Set();
    const codes = new Set(input.alt);
    input.alt.forEach((code) => altered.add(Math.floor(code / 3)));
    return M.atoms.map((atom) => {
      const test = atom.test;
      const value = input.c[test.col];
      switch (test.t) {
        case "eq": return String(value) === test.v ? 1 : 0;
        case "lt": return value !== null && value < test.v ? 1 : 0;
        case "ge": return value !== null && value >= test.v ? 1 : 0;
        case "le": return value !== null && value <= test.v ? 1 : 0;
        case "gene": return codes.has(test.gene * 3 + KIND[test.kind]) ? 1 : 0;
        default: return test.genes.reduce((sum, gene) => sum + (altered.has(gene) ? 1 : 0), 0);
      }
    });
  }

  function firedFormulas(formulas, values) {
    return formulas.filter((f) => f.atoms.every((a) => values[a] > 0));
  }

  function ruleRisk(formulas, values) {
    let risk = 0;
    for (const f of formulas) {
      let count = 1;
      for (const a of f.atoms) count *= values[a];
      risk += (f.w * (Math.log1p(count) - f.mean)) / f.std;
    }
    return risk;
  }

  function clinicalVector(M, input) {
    const vector = [input.age19, input.sex];
    const hot = (levels, value) => levels.forEach((level) => vector.push(level === value ? 1 : 0));
    hot(M.levels.sub, input.sub19);
    hot(M.levels.stage, input.st19);
    hot(M.levels.m, input.m19);
    return vector;
  }

  function designValue(M, input, column) {
    if (M.columns.numeric.includes(column)) {
      const raw = input.c[column];
      const value = raw === null || raw === undefined ? M.columns.median[column] : raw;
      return column === "tmb_nonsynonymous" ? Math.log1p(Math.max(value, 0)) : value;
    }
    for (const name of M.columns.nominal) {
      if (column.startsWith(name + "_")) return String(input.c[name]) === column.slice(name.length + 1) ? 1 : 0;
    }
    return 0;
  }

  /* One fold's prediction. gnn === null holds the GNN term at that fold's training mean. */
  function predictFold(M, input, values, k, gnn) {
    const fold = M.folds[k];
    const rule = ruleRisk(fold.formulas, values);
    const clinical = clinicalVector(M, input);
    const lookup = (column, mean) => {
      if (column === "gnn_risk") return gnn === null ? mean : gnn;
      if (column === "rule_risk") return rule;
      if (column.startsWith("clinical_")) return clinical[Number(column.slice(9))];
      return designValue(M, input, column);
    };
    let lp = 0;
    fold.cox.cols.forEach((column, i) => {
      const z = (lookup(column, fold.cox.mean[i]) - fold.cox.mean[i]) / fold.cox.std[i];
      lp += fold.cox.coef[i] * (z - fold.cox.norm[i]);
    });
    const hazard = Math.exp(lp);
    const survival = fold.h0.map((h) => Math.exp(-h * hazard));
    let mu = fold.aft.intercept;
    fold.aft.cols.forEach((column, i) => {
      mu += (fold.aft.coef[i] * (lookup(column, fold.aft.mean[i]) - fold.aft.mean[i])) / fold.aft.std[i];
    });
    const scale = Math.exp(mu);
    const quantile = (p) => scale * Math.pow(1 / p - 1, 1 / fold.aft.rho);
    return { rule, lp, survival, month: scale, low: quantile(0.75), high: quantile(0.25) };
  }

  function percentile(M, risk) {
    const q = M.riskQuantiles;
    if (risk <= q[0]) return 0;
    if (risk >= q[q.length - 1]) return 1;
    let lo = 0, hi = q.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (q[mid] <= risk) lo = mid; else hi = mid;
    }
    const part = q[hi] > q[lo] ? (risk - q[lo]) / (q[hi] - q[lo]) : 0;
    return (lo + part) / (q.length - 1);
  }

  function summarise(M, prediction) {
    const at = (t) => 1 - prediction.survival[M.grid.indexOf(t)];
    const code = (m) => (m <= 12 ? "A" : m <= 24 ? "B" : m <= 36 ? "C" : "D");
    return Object.assign(prediction, {
      risk: prediction.lp,
      percentile: percentile(M, prediction.lp),
      progress: HORIZONS.map(at),
      windowCode: code(prediction.month),
    });
  }

  /* The held-out fold model, which is what the reports use for a cohort patient. */
  function predictCohort(M, patient) {
    const values = atomValues(M, patient);
    const result = summarise(M, predictFold(M, patient, values, patient.f, patient.gnn));
    result.values = values;
    result.formulas = firedFormulas(M.folds[patient.f].formulas, values).map((f) => ({ text: f.text, w: f.w, source: f.source, support: f.support, folds: 1 }));
    return result;
  }

  /* A patient no fold has seen: the five fold models are averaged. */
  function predictNew(M, input) {
    const values = atomValues(M, input);
    const parts = M.folds.map((_, k) => predictFold(M, input, values, k, null));
    const mean = (key) => parts.reduce((sum, part) => sum + part[key], 0) / parts.length;
    const survival = M.grid.map((_, i) => parts.reduce((sum, part) => sum + part.survival[i], 0) / parts.length);
    const result = summarise(M, { rule: mean("rule"), lp: mean("lp"), survival, month: mean("month"), low: mean("low"), high: mean("high") });
    result.values = values;
    const merged = new Map();
    M.folds.forEach((fold) => {
      firedFormulas(fold.formulas, values).forEach((f) => {
        const item = merged.get(f.text) || { text: f.text, w: 0, source: f.source, support: f.support, folds: 0 };
        item.w += f.w / M.folds.length;
        item.folds += 1;
        merged.set(f.text, item);
      });
    });
    result.formulas = [...merged.values()];
    return result;
  }

  function relevantAtoms(M, path) {
    const relevant = new Set(M.atomsByNode.get(path.end) || []);
    (M.atomsByNode.get(M.genes[path.g]) || []).forEach((a) => relevant.add(a));
    path.n.forEach((code) => {
      const [id, type] = M.nodes[code];
      if (type === "gene") (M.atomsByEntrez.get(id.split(":").pop()) || []).forEach((a) => relevant.add(a));
    });
    return relevant;
  }

  function ruleForPath(path, fired, relevant) {
    let score = 0, count = 0, strongest = null;
    for (const f of fired) {
      if (f.atoms.some((a) => relevant.has(a))) {
        score += f.w;
        count += 1;
        if (!strongest || Math.abs(f.w) > Math.abs(strongest.w)) strongest = f;
      }
    }
    return { score, count, strongest };
  }

  /* Rule part of a stored path for one fold's formulas. */
  function pathRule(M, pathIndex, values, k) {
    const path = M.paths[pathIndex];
    return ruleForPath(path, firedFormulas(M.folds[k].formulas, values), relevantAtoms(M, path));
  }

  /* Top-K paths for a new patient, chosen as paths.select_top_paths chooses them. Rule weights are averaged over the
     folds, and the neural part is the mean GNNExplainer percentile the same path earned across the cohort. */
  function topPathsNew(M, input, values, kMax = 5, kMin = 3, perGene = 2) {
    const fired = M.folds.map((fold) => firedFormulas(fold.formulas, values));
    const genes = new Set(input.alt.map((code) => Math.floor(code / 3)));
    const scored = [];
    genes.forEach((gene) => {
      (M.pathsByGene.get(gene) || []).forEach((index) => {
        const path = M.paths[index];
        const relevant = relevantAtoms(M, path);
        let score = 0, count = 0, strongest = null;
        fired.forEach((items) => {
          const part = ruleForPath(path, items, relevant);
          score += part.score / fired.length;
          count = Math.max(count, part.count);
          if (part.strongest && (!strongest || Math.abs(part.strongest.w) > Math.abs(strongest.w))) strongest = part.strongest;
        });
        const symbolic = Math.min(Math.abs(score) / M.ruleScale, 1);
        scored.push({ index, neural: path.neural, ruleScore: score, ruleCount: count, symbolic, ns: 0.6 * path.neural + 0.4 * symbolic, strongest });
      });
    });
    scored.sort((a, b) => b.ns - a.ns);
    const chosen = [], perGeneCount = new Map(), ends = new Set(), routes = new Set();
    for (const item of scored) {
      const path = M.paths[item.index];
      if ((perGeneCount.get(path.g) || 0) >= perGene || ends.has(path.end) || routes.has(path.route)) continue;
      chosen.push(item);
      perGeneCount.set(path.g, (perGeneCount.get(path.g) || 0) + 1);
      ends.add(path.end);
      routes.add(path.route);
      if (chosen.length >= kMax) break;
    }
    chosen.forEach((item) => (item.prominent = item.ns >= M.cutoff));
    const prominent = chosen.filter((item) => item.prominent).length;
    return chosen.slice(0, Math.max(Math.min(kMax, prominent), Math.min(kMin, chosen.length)));
  }

  /* "TP53, MYC amp, RB1 del" becomes alteration codes, plus any symbols outside the 500-gene panel. */
  function parseGenes(M, text) {
    const codes = [], unknown = [];
    const seen = new Set();
    text.split(/[,;\n]+/).map((item) => item.trim()).filter(Boolean).forEach((item) => {
      const [symbol, kindWord] = item.split(/[\s:]+/);
      const gene = M.geneIndex.get(symbol.toUpperCase());
      if (gene === undefined) { unknown.push(symbol); return; }
      const word = (kindWord || "mut").toLowerCase();
      const kind = word.startsWith("amp") ? 1 : word.startsWith("del") ? 2 : 0;
      const code = gene * 3 + kind;
      if (!seen.has(code)) { seen.add(code); codes.push(code); }
    });
    return { codes, unknown };
  }

  return { prepare, atomValues, predictCohort, predictNew, pathRule, topPathsNew, parseGenes, percentile, KIND_NAME, HORIZONS };
});
