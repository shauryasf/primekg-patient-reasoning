#!/usr/bin/env python
"""Neuro-symbolic progression model: survival GNN + RuleKit survival rules + fused Cox.

Stages, each cached under ``--output-dir`` so a later stage can be rerun alone:

- ``neural``    survival GNNs over the fused patient + PrimeKG graphs (Cox loss)
- ``symbolic``  RuleKit survival rules (baseline) and PrimeKG fact evidence
- ``logic``     survival Markov logic, probabilistic soft logic, RL rules, and
                RL-structured Markov logic; the best becomes the rule engine
- ``fusion``    clinical baselines, a random survival forest, and the fused Cox
                model that stacks the neural and symbolic risks; per-patient
                survival curves and the predicted progression window
- ``explain``   GNN-scored knowledge-graph paths for every patient

All stages share one cohort (every patient with a ``PFS_STATUS``) and one set of
stratified folds, so every prediction written here is out-of-fold.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.train_and_evaluate_gnns as gnn_cv  # noqa: E402
from src.dataset_fusion import CLINICAL_DIM, FEATURE_DIM, standardize_ages  # noqa: E402
from src.gnn_models import build_model  # noqa: E402
from src.neurosymbolic.cohort import StudyCohort, load_study  # noqa: E402
from src.neurosymbolic.evaluation import brier_from_curves, metrics_frame, survival_metrics  # noqa: E402
from src.neurosymbolic.features import (  # noqa: E402
    alteration_table,
    cox_design_matrix,
    extended_clinical_table,
    outcome_table,
    predicate_table,
    record_table,
)
from src.neurosymbolic.kg_evidence import fact_carriers, fold_fact_evidence  # noqa: E402
from src.neurosymbolic.logic.atoms import build_atoms  # noqa: E402
from src.neurosymbolic.logic.runner import LABELS as LOGIC_LABELS  # noqa: E402
from src.neurosymbolic.logic.runner import METHODS as LOGIC_METHODS  # noqa: E402
from src.neurosymbolic.logic.runner import choose_engine, fold_c_index, run_reasoners  # noqa: E402
from src.neurosymbolic.paths import explain_patient, paths_frame  # noqa: E402
from src.neurosymbolic.rules import RULE_PARAMS, coverage_matrix, fit_rules, predict_curves, restricted_mean, rulekit_frame  # noqa: E402
from src.neurosymbolic.timing import progression_timing  # noqa: E402

LOGGER = logging.getLogger("neurosymbolic")
DEFAULT_NEURAL = ("graphsage:cox", "gcn:cox", "rgcn:cox", "graphsage:cindex")
GRID = np.arange(0.0, 241.0, 1.0)


def _graph_logits(model, graphs, indices) -> np.ndarray:
    """Raw log-risk for ``indices``. The training script returns sigmoid(logit)."""
    from torch_geometric.loader import DataLoader

    loader = DataLoader([graphs[int(index)] for index in indices], batch_size=64, shuffle=False)
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            values.append(gnn_cv._graph_logits(model, batch).cpu().numpy())
    return np.concatenate(values) if values else np.array([])


def load_fold_model(study: StudyCohort, arch: str, fold: int, state_dir: Path):
    """Rebuild one fold's GNN and restandardise the clinical block on that fold's training rows."""
    train, _ = study.folds[fold]
    standardize_ages(study.cohort.graphs, study.cohort.mlp_x, train)
    model = build_model(arch, FEATURE_DIM, study.cohort.catalog_size, study.cohort.mlp_x.shape[1], mlp_clinical_dim=CLINICAL_DIM, fusion=study.fusion)
    model.load_state_dict(torch.load(state_dir / f"fold{fold}.pt", weights_only=True))
    model.eval()
    return model


def stage_neural(study: StudyCohort, args, out: Path) -> None:
    neural_dir = out / "neural"
    rows = {}
    oof = pd.DataFrame({"patient_id": study.patient_ids, "fold": study.fold_id, "time": study.time, "event": study.event})
    for spec in args.neural:
        arch, loss = spec.split(":")
        started = clock.time()
        options = gnn_cv.TrainOptions(loss=loss, early_stop="cindex", batch_size=128 if loss == "cox" else 32)
        result = gnn_cv.cross_validate(
            arch,
            study.cohort,
            len(study.folds),
            args.epochs,
            args.patience,
            args.seed,
            fusion=study.fusion,
            options=options,
            survival=(study.time, study.event),
        )
        state_dir = neural_dir / f"{arch}_{loss}"
        state_dir.mkdir(parents=True, exist_ok=True)
        logits = np.full(len(study.patient_ids), np.nan)
        for fold, (train, test) in enumerate(study.folds):
            if not np.array_equal(np.sort(result["train_indices"][fold]), np.sort(train)):
                raise RuntimeError("GNN folds do not match the study folds; check the seed")
            torch.save(result["states"][fold], state_dir / f"fold{fold}.pt")
            model = load_fold_model(study, arch, fold, state_dir)
            logits[test] = _graph_logits(model, study.cohort.graphs, test)
        name = f"gnn_{arch}_{loss}"
        oof[name] = logits
        rows[name] = survival_metrics(study.time, study.event, logits, study.fold_id)
        rows[name]["minutes"] = (clock.time() - started) / 60
        (state_dir / "history.json").write_text(json.dumps(result["histories"]), encoding="utf-8")
        LOGGER.info("%s Harrell C %.3f (fold mean %.3f ± %.3f) in %.1f min", name, rows[name]["harrell_c"], rows[name]["harrell_c_fold_mean"], rows[name]["harrell_c_fold_std"], rows[name]["minutes"])
        oof.to_csv(neural_dir / "oof_risk.csv", index=False)
        metrics_frame(rows).to_csv(neural_dir / "metrics.csv", index=False)


def _tables(study: StudyCohort, out: Path):
    """Readable patient tables shared by the symbolic stage and the reports, cached as CSV."""
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    clinical = extended_clinical_table(study.dataset_dir, study.patient_ids)
    alterations = alteration_table(study.dataset_dir, study.patient_ids, study.panel)
    predicates = predicate_table(clinical, alterations, study.prime)
    clinical.to_csv(tables_dir / "clinical_extended.csv")
    alterations.to_csv(tables_dir / "alterations.csv", index=False)
    predicates.frame.to_csv(tables_dir / "predicates.csv")
    (tables_dir / "predicate_meta.json").write_text(
        json.dumps(
            {
                "labels": predicates.labels,
                "kinds": predicates.kinds,
                "members": {key: sorted(value) for key, value in predicates.members.items()},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    record_table(study.dataset_dir, study.patient_ids).to_csv(tables_dir / "record.csv")
    outcome_table(study.dataset_dir, study.patient_ids).to_csv(tables_dir / "outcomes.csv")
    return clinical, alterations, predicates


def stage_symbolic(study: StudyCohort, args, out: Path) -> None:
    """RuleKit survival rules per fold, out-of-fold rule risk, and PrimeKG fact evidence."""
    symbolic_dir = out / "symbolic"
    symbolic_dir.mkdir(parents=True, exist_ok=True)
    clinical, alterations, predicates = _tables(study, out)
    frame = rulekit_frame(predicates)
    # The count of altered genes produced non-monotone interval rules with no reading.
    frame = frame.drop(columns=["n_altered_genes"])
    timed = np.isfinite(study.time)
    curves = np.ones((len(study.patient_ids), len(GRID)))
    rule_rows, coverage_rows, km = [], [], {}
    for fold, (train, test) in enumerate(study.folds):
        train = train[timed[train]]
        started = clock.time()
        model, rules = fit_rules(frame.iloc[train], study.time[train], study.event[train], predicates, fold, RULE_PARAMS)
        curves[test] = predict_curves(model, frame.iloc[test], GRID)
        covered = coverage_matrix(model, frame.iloc[test])
        for rule in rules:
            row = rule.as_row()
            row["train_share"] = rule.coverage / len(train)
            rule_rows.append(row)
            km[rule.rule_id] = {"times": rule.km_times, "survival": rule.km_survival}
        for row_index, patient_index in enumerate(test):
            for rule_index in np.nonzero(covered[row_index])[0]:
                coverage_rows.append({"patient_id": study.patient_ids[patient_index], "rule_id": rules[rule_index].rule_id})
        LOGGER.info("RuleKit fold %d: %d rules in %.0f s", fold, len(rules), clock.time() - started)
    risk = -restricted_mean(curves, GRID, tau=120.0)
    pd.DataFrame(rule_rows).to_csv(symbolic_dir / "rules.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(symbolic_dir / "patient_rules.csv", index=False)
    (symbolic_dir / "rule_km.json").write_text(json.dumps(km), encoding="utf-8")
    np.save(symbolic_dir / "rule_curves.npy", curves)
    pd.DataFrame({"patient_id": study.patient_ids, "fold": study.fold_id, "rulekit": risk}).to_csv(symbolic_dir / "oof_risk.csv", index=False)
    metrics = {"rulekit": survival_metrics(study.time, study.event, risk, study.fold_id)}
    metrics_frame(metrics).to_csv(symbolic_dir / "metrics.csv", index=False)
    LOGGER.info("RuleKit Harrell C %.3f (fold mean %.3f)", metrics["rulekit"]["harrell_c"], metrics["rulekit"]["harrell_c_fold_mean"])

    altered = alterations.groupby("patient_id")["entrez"].agg(lambda values: set(values)).to_dict()
    facts, carriers = fact_carriers(study.patient_ids, altered, study.prime)
    evidence = fold_fact_evidence(facts, carriers, study.time, study.event, study.folds)
    names = {f"pathway:{key}": value for key, value in study.prime.pathway_names.items()}
    names.update({f"drug:{key}": value for key, value in study.prime.drug_names.items()})
    names.update({f"disease:{key}": value for key, value in study.prime.disease_names.items()})
    evidence["name"] = evidence["fact"].map(names)
    evidence.to_csv(symbolic_dir / "kg_evidence.csv", index=False)
    LOGGER.info("PrimeKG fact evidence: %d fact-fold rows", len(evidence))


def build_atom_table(study: StudyCohort, out: Path):
    """Ground atoms for the logic reasoners, from the cached readable tables."""
    tables = out / "tables"
    clinical = pd.read_csv(tables / "clinical_extended.csv", index_col=0).loc[study.patient_ids]
    predicates = pd.read_csv(tables / "predicates.csv", index_col=0).loc[study.patient_ids]
    meta = json.loads((tables / "predicate_meta.json").read_text(encoding="utf-8"))
    alterations = pd.read_csv(tables / "alterations.csv", dtype={"entrez": str})
    return build_atoms(clinical, predicates, meta["labels"], alterations, study.prime)


def stage_logic(study: StudyCohort, args, out: Path) -> None:
    """Survival MLN, survival PSL, RL rules, and RL-structured MLN on the shared folds."""
    logic_dir = out / "logic"
    logic_dir.mkdir(parents=True, exist_ok=True)
    atoms = build_atom_table(study, out)
    atoms.frame().to_csv(logic_dir / "atoms.csv", index=False)
    np.save(logic_dir / "atom_values.npy", atoms.values)
    LOGGER.info("Ground atoms: %d (%s)", len(atoms.names), pd.Series(atoms.kinds).value_counts().to_dict())
    risks, weights, infos, seconds = run_reasoners(atoms, study.time, study.event, study.folds, seed=args.seed, log=LOGGER.info)
    risks["rulekit"] = pd.read_csv(out / "symbolic" / "oof_risk.csv")["rulekit"].to_numpy()
    # Patients with identical facts differ only in the last bits of a summed score;
    # rounding makes them exact ties here and after the CSV round trip alike.
    risks = {method: np.round(risk, 10) for method, risk in risks.items()}
    rules = pd.read_csv(out / "symbolic" / "rules.csv")
    rows = []
    for method, risk in risks.items():
        per_fold = fold_c_index(study.time, study.event, risk, study.folds)
        metrics = survival_metrics(study.time, study.event, risk, study.fold_id)
        if method == "rulekit":
            n_rules = len(rules) / len(study.folds)
            kg_share = float((rules["n_kg_conditions"] > 0).mean())
        else:
            table = weights[method]
            n_rules = len(table) / len(study.folds)
            kg_share = float(table["uses_kg"].mean()) if len(table) else 0.0
        rows.append(
            {
                "method": method,
                "label": LOGIC_LABELS[method],
                "c_mean": float(np.mean(per_fold)),
                "c_std": float(np.std(per_fold, ddof=1)),
                "c_folds": " ".join(f"{value:.3f}" for value in per_fold),
                "uno_c_120": metrics["uno_c_120"],
                "td_auc_mean": metrics["td_auc_mean"],
                "rules_per_fold": n_rules,
                "kg_rule_share": kg_share,
                "seconds": seconds.get(method, float("nan")),
            }
        )
    summary = pd.DataFrame(rows).sort_values("c_mean", ascending=False)
    summary.to_csv(logic_dir / "metrics.csv", index=False)
    engine, selection = choose_engine(summary, len(study.folds))
    (logic_dir / "engine.json").write_text(json.dumps({"engine": engine, "label": LOGIC_LABELS[engine], **selection, "infos": infos}, indent=1, default=float), encoding="utf-8")
    pd.DataFrame({"patient_id": study.patient_ids, "fold": study.fold_id, **risks}).to_csv(logic_dir / "oof_risk.csv", index=False)
    for method, table in weights.items():
        table.to_csv(logic_dir / f"{method}_weights.csv", index=False)
    LOGGER.info("Logic reasoners:\n%s", summary[["method", "c_mean", "c_std", "rules_per_fold", "kg_rule_share"]].to_string(index=False))
    LOGGER.info("Symbolic engine: %s (%s)", engine, selection)


def _cox_fit_predict(train_x: pd.DataFrame, test_x: pd.DataFrame, time: np.ndarray, event: np.ndarray, penalizer: float = 0.1):
    """Standardised, ridge-penalised Cox model; returns test log-risk and survival on ``GRID``."""
    keep = train_x.columns[train_x.std(ddof=0) > 0]
    mean, std = train_x[keep].mean(), train_x[keep].std(ddof=0)
    train_z = (train_x[keep] - mean) / std
    test_z = (test_x[keep] - mean) / std
    frame = train_z.assign(T=time, E=event)
    fitter = CoxPHFitter(penalizer=penalizer).fit(frame, duration_col="T", event_col="E")
    risk = fitter.predict_log_partial_hazard(test_z).to_numpy()
    survival = fitter.predict_survival_function(test_z, times=GRID).to_numpy().T
    return risk, survival, fitter


def choose_report_model(metrics: pd.DataFrame, curves: dict, forced: str | None) -> tuple[str, dict]:
    """One-standard-error rule over the models that produce a survival curve.

    The best fold-mean Harrell C sets a bar of ``best - sd / sqrt(folds)``. Every
    model above the bar is statistically indistinguishable from the best, and among
    those the neuro-symbolic model is preferred because it is the one whose neural
    and symbolic parts the report can explain.
    """
    ranked = metrics[metrics["model"].isin(curves)].sort_values("harrell_c_fold_mean", ascending=False)
    best = ranked.iloc[0]
    standard_error = float(best["harrell_c_fold_std"]) / np.sqrt(5)
    bar = float(best["harrell_c_fold_mean"]) - standard_error
    eligible = ranked[ranked["harrell_c_fold_mean"] >= bar]["model"].tolist()
    if forced:
        chosen, reason = forced, "set with --report-model"
    elif "neurosymbolic_cox" in eligible:
        chosen, reason = "neurosymbolic_cox", f"within one standard error of the best ({best['model']})"
    else:
        chosen, reason = str(best["model"]), "best fold-mean Harrell C"
    return chosen, {"best_model": str(best["model"]), "best_c": float(best["harrell_c_fold_mean"]), "standard_error": standard_error, "bar": bar, "eligible": eligible, "reason": reason}


def stage_fusion(study: StudyCohort, args, out: Path) -> None:
    """Baselines, the fused neuro-symbolic Cox model, survival curves, and timing."""
    fusion_dir = out / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    clinical = pd.read_csv(out / "tables" / "clinical_extended.csv", index_col=0).loc[study.patient_ids]
    predicates = pd.read_csv(out / "tables" / "predicates.csv", index_col=0).loc[study.patient_ids]
    neural = pd.read_csv(out / "neural" / "oof_risk.csv")
    neural_metrics = pd.read_csv(out / "neural" / "metrics.csv").sort_values("harrell_c_fold_mean", ascending=False)
    best_gnn = str(neural_metrics.iloc[0]["model"])
    symbolic = pd.read_csv(out / "symbolic" / "oof_risk.csv")
    logic = pd.read_csv(out / "logic" / "oof_risk.csv")
    engine = json.loads((out / "logic" / "engine.json").read_text(encoding="utf-8"))["engine"]
    design = cox_design_matrix(clinical)
    design.index = study.patient_ids
    binary = (predicates[[column for column in predicates.columns if column.startswith(("mut_", "amp_", "del_", "kg_"))]] == "yes").astype(float)
    # The 19-field clinical block the GNN's clinical head reads (age, sex, subtype, stage, M stage).
    clinical_19 = pd.DataFrame(study.cohort.mlp_x[:, :CLINICAL_DIM], index=study.patient_ids, columns=[f"clinical_{i}" for i in range(CLINICAL_DIM)])
    clinical_19 = clinical_19.fillna(clinical_19.mean())
    # The rule risk is the chosen logic reasoner's out-of-fold risk (RuleKit stays as a baseline row).
    learned = {"gnn_risk": neural[best_gnn].to_numpy(), "rule_risk": logic[engine].to_numpy()}
    inputs = {
        "clinical_cox_19": clinical_19,
        "clinical_cox_extended": design,
        "neurosymbolic_cox": clinical_19.assign(**learned),
        "neurosymbolic_cox_extended": design.assign(**learned),
        "neural_symbolic_only": pd.DataFrame(learned, index=study.patient_ids),
    }
    timed = np.isfinite(study.time) & (study.time > 0)
    gnn_columns = [column for column in neural.columns if column.startswith("gnn_")]
    risks = {**{column: neural[column].to_numpy() for column in gnn_columns}, "rulekit": symbolic["rulekit"].to_numpy()}
    risks.update({f"logic_{method}": logic[method].to_numpy() for method in LOGIC_METHODS})
    curves = {name: np.ones((len(study.patient_ids), len(GRID))) for name in [*inputs, "random_survival_forest"]}
    coefficients = []
    for name, x in inputs.items():
        risk = np.full(len(study.patient_ids), np.nan)
        for fold, (train, test) in enumerate(study.folds):
            train = train[timed[train]]
            fold_risk, fold_curves, fitter = _cox_fit_predict(x.iloc[train], x.iloc[test], study.time[train], study.event[train])
            risk[test] = fold_risk
            curves[name][test] = fold_curves
            if name == "neurosymbolic_cox":
                summary = fitter.summary[["coef", "exp(coef)", "p"]].reset_index().assign(fold=fold)
                coefficients.append(summary)
        risks[name] = risk
    forest_x = pd.concat([design, binary], axis=1)
    risk = np.full(len(study.patient_ids), np.nan)
    for fold, (train, test) in enumerate(study.folds):
        train = train[timed[train]]
        forest = RandomSurvivalForest(n_estimators=400, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=args.seed)
        forest.fit(forest_x.iloc[train].to_numpy(), Surv.from_arrays(study.event[train].astype(bool), study.time[train]))
        risk[test] = forest.predict(forest_x.iloc[test].to_numpy())
        functions = forest.predict_survival_function(forest_x.iloc[test].to_numpy())
        curves["random_survival_forest"][test] = np.vstack([fn(np.clip(GRID, fn.x[0], fn.x[-1])) for fn in functions])
    risks["random_survival_forest"] = risk
    rows = {name: survival_metrics(study.time, study.event, value, study.fold_id) for name, value in risks.items()}
    for name, model_curves in curves.items():
        rows[name]["integrated_brier_120"] = brier_from_curves(study.time, study.event, GRID, model_curves)
    metrics = metrics_frame(rows).sort_values("harrell_c_fold_mean", ascending=False)
    metrics.to_csv(fusion_dir / "metrics.csv", index=False)
    pd.concat(coefficients).to_csv(fusion_dir / "neurosymbolic_coefficients.csv", index=False)
    risk_table = pd.DataFrame({"patient_id": study.patient_ids, "fold": study.fold_id, "time": study.time, "event": study.event, **risks})
    risk_table.to_csv(fusion_dir / "oof_risk_all.csv", index=False)
    chosen, selection = choose_report_model(metrics, curves, args.report_model)
    LOGGER.info("Report model: %s (%s)", chosen, selection["reason"])
    np.save(fusion_dir / "report_curves.npy", curves[chosen])
    timing, evaluation = progression_timing(curves[chosen], study, GRID, features=inputs["neurosymbolic_cox_extended"])
    timing.insert(0, "patient_id", study.patient_ids)
    timing["risk"] = risks[chosen]
    timing["risk_percentile"] = pd.Series(risks[chosen]).rank(pct=True).to_numpy()
    timing.to_csv(fusion_dir / "patient_predictions.csv", index=False)
    evaluation["report_model"] = chosen
    evaluation["best_gnn"] = best_gnn
    evaluation["selection"] = selection
    (fusion_dir / "timing_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    LOGGER.info("Timing on held-out progressors: %s", json.dumps(evaluation["chosen"]))


def stage_explain(study: StudyCohort, args, out: Path) -> None:
    """GNNExplainer paths for every patient from the fold model that held them out."""
    explain_dir = out / "explain"
    explain_dir.mkdir(parents=True, exist_ok=True)
    neural_metrics = pd.read_csv(out / "neural" / "metrics.csv").sort_values("harrell_c_fold_mean", ascending=False)
    best = str(neural_metrics.iloc[0]["model"])
    _, arch, loss = best.split("_")
    state_dir = out / "neural" / f"{arch}_{loss}"
    frames, genes = [], []
    for fold, (_, test) in enumerate(study.folds):
        model = load_fold_model(study, arch, fold, state_dir)
        started = clock.time()
        for index in test:
            patient_id = study.patient_ids[index]
            paths, importance, source = explain_patient(model, study.cohort.graphs[index], study.cohort.metas[index])
            frames.append(paths_frame(patient_id, fold, paths).assign(score_source=source))
            genes.extend({"patient_id": patient_id, "gene": gene, "importance": value} for gene, value in importance.items())
        LOGGER.info("Explained fold %d (%d patients) in %.1f min", fold, len(test), (clock.time() - started) / 60)
    pd.concat(frames, ignore_index=True).to_csv(explain_dir / "candidate_paths.csv", index=False)
    pd.DataFrame(genes).to_csv(explain_dir / "gene_importance.csv", index=False)
    (explain_dir / "model.json").write_text(json.dumps({"model": best, "architecture": arch, "loss": loss}), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default="neural,symbolic,logic,fusion,explain")
    parser.add_argument("--neural", default=",".join(DEFAULT_NEURAL), help="comma-separated arch:loss specs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "neurosymbolic")
    parser.add_argument("--report-model", default=None, help="curve model for the reports; defaults to the best by fold-mean Harrell C")
    args = parser.parse_args()
    args.neural = [spec.strip() for spec in args.neural.split(",") if spec.strip()]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    study = load_study(n_folds=args.folds, seed=args.seed)
    LOGGER.info("Cohort %d patients, %d progressions", len(study.patient_ids), int(study.event.sum()))
    stages = [stage.strip() for stage in args.stages.split(",")]
    if "neural" in stages:
        stage_neural(study, args, out)
    if "symbolic" in stages:
        stage_symbolic(study, args, out)
    if "logic" in stages:
        stage_logic(study, args, out)
    if "fusion" in stages:
        stage_fusion(study, args, out)
    if "explain" in stages:
        stage_explain(study, args, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
