#!/usr/bin/env python
"""Export the fitted neuro-symbolic model to demo/model_data.json for the browser demo.

Nothing here is retrained differently from the pipeline. The fold Cox and AFT models are refitted with the
same inputs and penalties as ``stage_fusion`` so their parameters can be written out, and ``demo/check_engine.js``
then proves that the JavaScript engine reproduces the pipeline's out-of-fold predictions patient by patient.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, LogLogisticAFTFitter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_fusion import M_STAGES, STAGES, SUBTYPES, PrimeSlice, clinical_block  # noqa: E402
from src.neurosymbolic.features import NOMINAL_CLINICAL, NUMERIC_CLINICAL, cox_design_matrix  # noqa: E402
from src.neurosymbolic.report_pdf import load_context  # noqa: E402
from src.neurosymbolic.timing import AFT_PENALIZER  # noqa: E402

OUT = ROOT / "outputs" / "neurosymbolic"
DATASET = ROOT / "brca_tcga_pan_can_atlas_2018"
TARGET = ROOT / "demo" / "model_data.json"
GRID = np.arange(0.0, 241.0, 1.0)
COX_PENALIZER = 0.1


def rounded(values, digits: int = 6) -> list:
    return [None if value is None or not np.isfinite(value) else float(f"{value:.{digits}g}") for value in values]


def clinical_19(patient_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(DATASET / "data_clinical_patient.txt", sep="\t", comment="#", dtype=str, keep_default_na=False).set_index("PATIENT_ID")
    rows = np.vstack([clinical_block(frame.loc[pid].to_dict()) for pid in patient_ids]).astype(float)
    table = pd.DataFrame(rows, index=patient_ids, columns=[f"clinical_{i}" for i in range(rows.shape[1])])
    return table.fillna(table.mean())


def main() -> int:
    ctx = load_context(OUT)
    logic = pd.read_csv(OUT / "logic" / "oof_risk.csv")
    patient_ids = logic["patient_id"].tolist()
    position = {pid: i for i, pid in enumerate(patient_ids)}
    fold_id = logic["fold"].to_numpy()
    oof = pd.read_csv(OUT / "fusion" / "oof_risk_all.csv").set_index("patient_id").loc[patient_ids]
    best_gnn = ctx.timing["best_gnn"]
    gnn = oof[best_gnn].to_numpy()
    rule = logic[ctx.logic_engine].to_numpy()
    time, event = oof["time"].to_numpy(float), oof["event"].to_numpy(int)
    timed = np.isfinite(time) & (time > 0)

    clinical = ctx.clinical.loc[patient_ids]
    c19 = clinical_19(patient_ids)
    design = cox_design_matrix(clinical)
    design.index = patient_ids
    numeric_median = clinical[list(NUMERIC_CLINICAL)].astype(float).median()

    atoms = ctx.atom_frame
    values = np.load(OUT / "logic" / "atom_values.npy")
    weights = pd.read_csv(OUT / "logic" / f"{ctx.logic_engine}_weights.csv")
    used = sorted({name for row in weights["atoms"] for name in str(row).split("|")}, key=atoms["atom"].tolist().index)
    atom_index = {name: i for i, name in enumerate(atoms["atom"])}
    used_index = {name: i for i, name in enumerate(used)}

    alterations = ctx.alterations
    genes = sorted(alterations["gene"].unique())
    gene_position = {gene: i for i, gene in enumerate(genes)}
    entrez = alterations.drop_duplicates("gene").set_index("gene")["entrez"].astype(str).to_dict()
    driver = alterations.drop_duplicates("gene").set_index("gene")["driver"].astype(str).eq("True").to_dict()
    entrez_to_gene = {value: key for key, value in entrez.items()}
    prime = ctx.prime
    disease_genes: dict[str, set[str]] = {}
    drug_genes: dict[str, set[str]] = {}
    for gene, items in prime.gene_diseases.items():
        for item in items:
            disease_genes.setdefault(item, set()).add(gene)
    for gene, items in prime.gene_drugs.items():
        for item in items:
            drug_genes.setdefault(item, set()).add(gene)
    node_genes = {"pathway": prime.pathway_genes, "disease": disease_genes, "drug": drug_genes}

    def gene_indices(entrez_ids) -> list[int]:
        return sorted(gene_position[entrez_to_gene[e]] for e in entrez_ids if e in entrez_to_gene)

    quartiles = {column: (float(np.nanquantile(clinical[column].astype(float), 0.75)), float(np.nanquantile(clinical[column].astype(float), 0.25))) for column in ("aneuploidy_score", "tmb_nonsynonymous")}
    atom_rows = []
    for name in used:
        row = atoms.iloc[atom_index[name]]
        members = [m for m in str(row["members"]).split("|") if m]
        entry = {"name": name, "kind": row["kind"], "label": row["label"], "formula": row["formula"], "nodes": members, "pathGenes": []}
        if row["kind"] == "clinical":
            if name == "age<40":
                entry["test"] = {"t": "lt", "col": "age", "v": 40}
            elif name == "age>=65":
                entry["test"] = {"t": "ge", "col": "age", "v": 65}
            elif name.endswith("_high"):
                column = name[:-5]
                entry["test"] = {"t": "ge", "col": column, "v": quartiles[column][0]}
            elif name.endswith("_low"):
                column = name[:-4]
                entry["test"] = {"t": "le", "col": column, "v": quartiles[column][1]}
            else:
                column, level = name.split("=", 1)
                entry["test"] = {"t": "eq", "col": column, "v": level}
        elif row["kind"] == "genomic":
            kind, gene = name.split("_", 1)
            entry["test"] = {"t": "gene", "kind": kind, "gene": gene_position[gene]}
        else:
            prefix, node = name[3:].split(":", 1)
            prefix = prefix.replace("kg_", "")
            entry["test"] = {"t": "kg", "genes": gene_indices(node_genes[prefix][node])}
            found: set[str] = set()
            for member in members:
                member_prefix, _, member_node = member.partition(":")
                found |= node_genes.get(member_prefix, {}).get(member_node, set())
            entry["pathGenes"] = sorted(found)
        atom_rows.append(entry)

    folds = []
    for fold in range(5):
        train = np.where((fold_id != fold) & timed)[0]
        rows = []
        for item in weights[weights["fold"] == fold].itertuples(index=False):
            names = str(item.atoms).split("|")
            count = np.ones(len(patient_ids))
            for name in names:
                count = count * values[:, atom_index[name]]
            counts = np.log1p(count)[train]
            std = counts.std()
            rows.append({"atoms": [used_index[n] for n in names], "w": float(item.weight_per_sd), "mean": float(counts.mean()), "std": float(std if std > 0 else 1.0), "text": item.formula, "source": item.source, "support": int(item.support)})
        x = c19.assign(gnn_risk=gnn, rule_risk=rule).iloc[train]
        keep = x.columns[x.std(ddof=0) > 0]
        mean, std = x[keep].mean(), x[keep].std(ddof=0)
        frame = ((x[keep] - mean) / std).assign(T=time[train], E=event[train])
        cox = CoxPHFitter(penalizer=COX_PENALIZER).fit(frame, duration_col="T", event_col="E")
        hazard = cox.baseline_cumulative_hazard_.iloc[:, 0]
        h0 = np.interp(GRID, hazard.index.to_numpy(), hazard.to_numpy(), left=0.0)
        # lifelines steps the baseline, so read it as a step function rather than interpolating it.
        h0 = hazard.reindex(hazard.index.union(GRID)).ffill().fillna(0.0).loc[GRID].to_numpy()

        x_aft = design.assign(gnn_risk=gnn, rule_risk=rule)
        progressed = np.where((fold_id != fold) & (event == 1) & np.isfinite(time))[0]
        subset = x_aft.iloc[progressed]
        keep_aft = subset.columns[subset.std(ddof=0) > 0]
        mean_aft, std_aft = subset[keep_aft].mean(), subset[keep_aft].std(ddof=0)
        aft_frame = ((subset[keep_aft] - mean_aft) / std_aft).assign(T=time[progressed], E=1)
        aft = LogLogisticAFTFitter(penalizer=AFT_PENALIZER).fit(aft_frame, duration_col="T", event_col="E")
        alpha = aft.params_.loc["alpha_"]
        folds.append(
            {
                "formulas": rows,
                "cox": {"cols": list(keep), "mean": rounded(mean, 10), "std": rounded(std, 10), "coef": rounded(cox.params_.loc[list(keep)], 10), "norm": rounded(np.asarray(cox._norm_mean.loc[list(keep)]), 10)},
                "h0": rounded(h0, 7),
                "aft": {"cols": list(keep_aft), "mean": rounded(mean_aft, 10), "std": rounded(std_aft, 10), "coef": rounded(alpha.loc[list(keep_aft)], 10), "intercept": float(alpha.loc["Intercept"]), "rho": float(np.exp(aft.params_.loc["beta_"].loc["Intercept"]))},
            }
        )

    # Every distinct path in the cohort, with the mean neural percentile it earned, shared through small dictionaries.
    scored = ctx.scored_paths
    node_table: dict[tuple, int] = {}
    node_rows: list[list] = []
    relation_table: dict[str, int] = {}

    def node_code(node: dict) -> int:
        key = (node["id"], node["type"], node["name"])
        if key not in node_table:
            node_table[key] = len(node_rows)
            node_rows.append([node["id"], node["type"], node["name"]])
        return node_table[key]

    def relation_code(name: str) -> int:
        return relation_table.setdefault(name, len(relation_table))

    grouped = scored.groupby(["altered_gene", "genes", "terminal_id"], sort=True).agg(neural=("neural", "mean"), nodes=("nodes", "first"), relations=("relations", "first"), n=("neural", "size"))
    path_rows, path_lookup = [], {}
    for key, row in grouped.iterrows():
        path_lookup[key] = len(path_rows)
        nodes = [n for n in json.loads(row["nodes"]) if n["type"] not in ("patient", "sample")]
        relations = json.loads(row["relations"])[-(len(nodes) - 1):] if len(nodes) > 1 else []
        path_rows.append({"g": gene_position[key[0]], "route": key[1], "end": key[2], "n": [node_code(item) for item in nodes], "r": [relation_code(str(item)) for item in relations], "neural": round(float(row["neural"]), 4)})

    magnitude = np.abs(scored["rule_score"].to_numpy())
    scale = float(np.quantile(magnitude[magnitude > 0], 0.95))

    predictions = ctx.predictions.loc[patient_ids]
    patients = []
    by_patient = alterations.groupby("patient_id")
    kind_code = {"mut": 0, "amp": 1, "del": 2}
    for index, pid in enumerate(patient_ids):
        record = clinical.loc[pid]
        alts = []
        if pid in by_patient.groups:
            for row in by_patient.get_group(pid).itertuples(index=False):
                alts.append(gene_position[row.gene] * 3 + kind_code[row.kind])
        top = []
        table = ctx.paths_for(pid)
        for row in (table.to_dict("records") if len(table) else []):
            top.append([path_lookup[(row["altered_gene"], row["genes"], row["terminal_id"])], round(float(row["neural"]), 4), round(float(row["rule_score"]), 5), int(row["rule_count"]), round(float(row["ns_score"]), 4), int(bool(row["prominent"]))])
        block = c19.loc[pid].to_numpy()
        patients.append(
            {
                "id": pid,
                "f": int(fold_id[index]),
                "c": {column: (None if pd.isna(record[column]) else (float(record[column]) if column in NUMERIC_CLINICAL else str(record[column]))) for column in clinical.columns},
                "sex": float(block[1]),
                "sub19": SUBTYPES[int(np.argmax(block[2:2 + len(SUBTYPES)]))],
                "st19": STAGES[int(np.argmax(block[2 + len(SUBTYPES):2 + len(SUBTYPES) + len(STAGES)]))],
                "m19": M_STAGES[int(np.argmax(block[2 + len(SUBTYPES) + len(STAGES):]))],
                "age19": float(block[0]),
                "alt": alts,
                "gnn": round(float(gnn[index]), 8),
                "t": None if not np.isfinite(time[index]) else round(float(time[index]), 2),
                "e": int(event[index]),
                "grp": predictions.loc[pid, "recorded_group"],
                "top": top,
            }
        )

    risks = oof[ctx.report_model].to_numpy()
    cohort = KaplanMeierFitter().fit(time[timed], event[timed]).survival_function_at_times(GRID).to_numpy()
    metrics = pd.read_csv(OUT / "logic" / "metrics.csv")
    data = {
        "columns": {"numeric": list(NUMERIC_CLINICAL), "nominal": list(NOMINAL_CLINICAL), "median": {k: float(v) for k, v in numeric_median.items()}, "design": list(design.columns)},
        "levels": {"sub": list(SUBTYPES), "stage": list(STAGES), "m": list(M_STAGES)},
        "age19": {"mean": float(c19["clinical_0"].mean())},
        "genes": genes,
        "entrez": [entrez[g] for g in genes],
        "driver": [i for i, g in enumerate(genes) if driver[g]],
        "atoms": atom_rows,
        "folds": folds,
        "grid": GRID.tolist(),
        "cohortKM": rounded(cohort, 5),
        "riskQuantiles": rounded(np.quantile(risks[np.isfinite(risks)], np.linspace(0, 1, 401)), 6),
        "nodes": node_rows,
        "relations": list(relation_table),
        "paths": path_rows,
        "ruleScale": scale,
        "cutoff": float(ctx.cutoff),
        "patients": patients,
        "summary": {
            "reasoners": metrics[["label", "c_mean", "c_std", "kg_rule_share", "rules_per_fold"]].round(4).to_dict("records"),
            "reportModel": ctx.report_model,
            "bestGnn": best_gnn,
        },
    }
    TARGET.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {TARGET} ({TARGET.stat().st_size / 1e6:.2f} MB): {len(used)} atoms, {len(path_rows)} paths, {len(patients)} patients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
