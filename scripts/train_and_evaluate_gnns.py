#!/usr/bin/env python
"""Stratified 5-fold benchmark of MLP, GCN, GATv2, and GraphSAGE on PFS recurrence."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_fusion import (
    FEATURE_DIM,
    build_cohort,
    load_gene_panel,
    load_primekg_slice,
    relations_from_codes,
    standardize_ages,
)
from src.downloads import ensure_brca_dataset, ensure_primekg_csv
from src.explain import enumerate_explanation_paths, score_edges
from src.gnn_models import MODEL_NAMES, build_model
from src.labels import load_recurrence_labels

LOGGER = logging.getLogger("train_gnns")
COLORS = {"mlp": "#4C78A8", "gcn": "#F58518", "gatv2": "#54A24B", "graphsage": "#E45756"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _loss_fn(y_train: np.ndarray) -> torch.nn.Module:
    positives = max(int(y_train.sum()), 1)
    negatives = max(int(len(y_train) - y_train.sum()), 1)
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float))


def _graph_logits(model, batch) -> torch.Tensor:
    return model(batch.x, batch.edge_index, batch.emb_id, batch.is_patient, batch.batch)


def _run_graphs(model, graphs, indices, y, loss_fn, optimizer=None) -> float:
    subset = [graphs[int(index)] for index in indices]
    loader = DataLoader(subset, batch_size=32, shuffle=optimizer is not None)
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad()
        logits = _graph_logits(model, batch)
        loss = loss_fn(logits, batch.y.view(-1))
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.item()) * int(batch.num_graphs)
        count += int(batch.num_graphs)
    return total / max(count, 1)


def _predict_graphs(model, graphs, indices) -> np.ndarray:
    subset = [graphs[int(index)] for index in indices]
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = _graph_logits(model, batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.array([])


def _run_mlp(model, features, indices, y, loss_fn, optimizer=None) -> float:
    order = np.array(indices)
    if optimizer is not None:
        order = order.copy()
        np.random.shuffle(order)
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for start in range(0, len(order), 64):
        batch_index = order[start : start + 64]
        inputs = torch.tensor(features[batch_index], dtype=torch.float)
        target = torch.tensor(y[batch_index], dtype=torch.float)
        if optimizer is not None:
            optimizer.zero_grad()
        loss = loss_fn(model(inputs), target)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.item()) * len(batch_index)
        count += len(batch_index)
    return total / max(count, 1)


def _predict_mlp(model, features, indices) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features[indices], dtype=torch.float)
        return torch.sigmoid(model(inputs)).cpu().numpy()


def _metric_row(model: str, endpoint: str, fold: int, y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = (probabilities >= threshold).astype(int)
    row = {
        "model": model,
        "endpoint": endpoint,
        "fold": fold,
        "threshold": threshold,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
    }
    if len(np.unique(y_true)) > 1:
        row["auroc"] = float(roc_auc_score(y_true, probabilities))
        row["auprc"] = float(average_precision_score(y_true, probabilities))
    return row


def cross_validate(model_name: str, cohort, folds: int, epochs: int, patience: int, seed: int) -> dict[str, object]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows, histories, states, train_folds = [], [], [], []
    oof_prob = np.full(len(cohort.y), np.nan)
    oof_pred = np.full(len(cohort.y), -1)
    oof_threshold = np.full(len(cohort.y), np.nan)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cohort.y)), cohort.y)):
        set_seed(seed + fold)
        inner_train, validation = train_test_split(
            train_idx, test_size=0.15, stratify=cohort.y[train_idx], random_state=seed + fold
        )
        features = standardize_ages(cohort.graphs, cohort.mlp_x, train_idx)
        model = build_model(model_name, FEATURE_DIM, cohort.catalog_size, features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = _loss_fn(cohort.y[inner_train])
        best_state, best_score, wait = None, -1.0, 0
        history = []
        for epoch in range(epochs):
            if model_name == "mlp":
                train_loss = _run_mlp(model, features, inner_train, cohort.y, loss_fn, optimizer)
                val_prob = _predict_mlp(model, features, validation)
            else:
                train_loss = _run_graphs(model, cohort.graphs, inner_train, cohort.y, loss_fn, optimizer)
                val_prob = _predict_graphs(model, cohort.graphs, validation)
            val_score = float(average_precision_score(cohort.y[validation], val_prob)) if len(np.unique(cohort.y[validation])) > 1 else 0.0
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_auprc": val_score})
            if val_score > best_score + 1e-4:
                best_score, best_state, wait = val_score, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        if model_name == "mlp":
            val_prob = _predict_mlp(model, features, validation)
            test_prob = _predict_mlp(model, features, test_idx)
        else:
            val_prob = _predict_graphs(model, cohort.graphs, validation)
            test_prob = _predict_graphs(model, cohort.graphs, test_idx)
        threshold = best_f1_threshold(cohort.y[validation], val_prob)
        rows.append(_metric_row(model_name, cohort.endpoint, fold, cohort.y[test_idx], test_prob, threshold))
        histories.append(history)
        states.append(copy.deepcopy(model.state_dict()))
        train_folds.append(np.array(train_idx))
        oof_prob[test_idx] = test_prob
        oof_pred[test_idx] = (test_prob >= threshold).astype(int)
        oof_threshold[test_idx] = threshold
        LOGGER.info(
            "%s fold %d %s AUROC %.3f AUPRC %.3f epochs %d",
            model_name,
            fold,
            cohort.endpoint,
            rows[-1]["auroc"],
            rows[-1]["auprc"],
            len(history),
        )
    return {
        "rows": rows,
        "histories": histories,
        "states": states,
        "train_indices": train_folds,
        "oof_prob": oof_prob,
        "oof_pred": oof_pred,
        "oof_threshold": oof_threshold,
    }


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auroc", "auprc", "f1", "balanced_accuracy"]
    rows = []
    for (model, endpoint), group in frame.groupby(["model", "endpoint"], sort=False):
        row = {"model": model, "endpoint": endpoint, "folds": int(len(group))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_curves(oof: pd.DataFrame, output_dir: Path) -> None:
    primary = oof[oof["endpoint"] == "pfs"]
    fig, axis = plt.subplots(figsize=(6.2, 5))
    for model, color in COLORS.items():
        group = primary[primary["model"] == model]
        if group.empty or group["y"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(group["y"], group["probability"])
        axis.plot(fpr, tpr, color=color, label=f"{model} {roc_auc_score(group['y'], group['probability']):.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("PFS recurrence ROC (out-of-fold)")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 5))
    for model, color in COLORS.items():
        group = primary[primary["model"] == model]
        if group.empty or group["y"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(group["y"], group["probability"])
        axis.plot(recall, precision, color=color, label=f"{model} {average_precision_score(group['y'], group['probability']):.3f}")
    prevalence = float(primary.drop_duplicates("patient_id")["y"].mean()) if len(primary) else 0
    axis.axhline(prevalence, linestyle="--", color="#888888", label=f"prevalence {prevalence:.3f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("PFS recurrence precision-recall (out-of-fold)")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curves.png", dpi=160)
    plt.close(fig)


def _plot_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    primary = summary[summary["endpoint"] == "pfs"]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(primary))
    width = 0.36
    axis.bar(positions - width / 2, primary["auroc_mean"], width, yerr=primary["auroc_std"], color="#4C78A8", label="AUROC", capsize=3)
    axis.bar(positions + width / 2, primary["auprc_mean"], width, yerr=primary["auprc_std"], color="#F58518", label="AUPRC", capsize=3)
    axis.set_xticks(positions, primary["model"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("5-fold mean ± std on PFS")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_bars.png", dpi=160)
    plt.close(fig)


def _plot_loss(histories: dict[str, list[list[dict]]], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 4.6))
    for model, color in COLORS.items():
        fold_histories = histories.get(model, [])
        if not fold_histories:
            continue
        longest = max(len(history) for history in fold_histories)
        grid = np.full((len(fold_histories), longest), np.nan)
        for fold, history in enumerate(fold_histories):
            grid[fold, : len(history)] = [item["train_loss"] for item in history]
        epochs = np.arange(1, longest + 1)
        axis.plot(epochs, np.nanmean(grid, axis=0), color=color, label=model)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.set_title("Training loss by architecture")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close(fig)


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, model: str, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(4.6, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    axis.set_yticks([0, 1], ["Censored", "Progression"])
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    axis.set_title(f"{model} out-of-fold confusion")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _explain(model_name: str, result: dict, cohort, output_dir: Path, limit: int) -> pd.DataFrame:
    order = np.argsort(-result["oof_prob"])
    chosen = [int(index) for index in order if cohort.y[int(index)] == 1 and not np.isnan(result["oof_prob"][int(index)])][:limit]
    rows = []
    for index in chosen:
        fold = next(fold for fold, train_idx in enumerate(result["train_indices"]) if index not in set(int(item) for item in train_idx.tolist()))
        standardize_ages(cohort.graphs, cohort.mlp_x, result["train_indices"][fold])
        model = build_model(model_name, FEATURE_DIM, cohort.catalog_size, cohort.mlp_x.shape[1])
        model.load_state_dict(result["states"][fold])
        model.eval()
        data = cohort.graphs[index]
        meta = cohort.metas[index]
        altered = [node for node, flag in enumerate(meta["altered"]) if flag]
        scores, source = score_edges(model, data, altered)
        paths = enumerate_explanation_paths(
            meta["node_ids"],
            meta["node_types"],
            meta["node_names"],
            meta["altered"],
            data.edge_index,
            data.is_reverse,
            relations_from_codes(data.rel_code),
            scores,
        )
        if not paths:
            rows.append(
                {
                    "patient_id": cohort.patient_ids[index],
                    "endpoint": cohort.endpoint,
                    "model": model_name,
                    "y_true": int(cohort.y[index]),
                    "y_prob": float(result["oof_prob"][index]),
                    "path_rank": 0,
                    "path_score": float("nan"),
                    "score_source": source,
                    "path": "",
                }
            )
            continue
        for rank, path in enumerate(paths, start=1):
            rows.append(
                {
                    "patient_id": cohort.patient_ids[index],
                    "endpoint": cohort.endpoint,
                    "model": model_name,
                    "y_true": int(cohort.y[index]),
                    "y_prob": float(result["oof_prob"][index]),
                    "path_rank": rank,
                    "path_score": float(path["path_score"]),
                    "score_source": source,
                    "path": path["path"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "explanations.csv", index=False)
    return frame


def _prepare_cohort(dataset_dir: Path, endpoint: str, panel, prime_frame):
    labels = load_recurrence_labels(dataset_dir, endpoint)
    cohort = build_cohort(dataset_dir, labels, prime_frame, panel)
    LOGGER.info("%s cohort: %d patients, %d positives", endpoint, len(cohort.y), int(cohort.y.sum()))
    return cohort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explain-n", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "gnn_benchmark")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ensure_brca_dataset() if args.dataset_dir is None else args.dataset_dir
    panel = load_gene_panel()
    panel_entrez = {str(info["entrez"]) for info in panel.values() if info.get("entrez")}
    prime_csv = ensure_primekg_csv()
    prime_frame = load_primekg_slice(prime_csv, panel_entrez)
    LOGGER.info("PrimeKG slice edges: %d", len(prime_frame))
    cohort = _prepare_cohort(dataset_dir, "pfs", panel, prime_frame)
    audit = {
        "endpoint": "pfs",
        "patients": int(len(cohort.y)),
        "positives": int(cohort.y.sum()),
        "negatives": int((cohort.y == 0).sum()),
        "primekg_edges": int(len(prime_frame)),
        "feature_dim": FEATURE_DIM,
        "catalog_size": int(cohort.catalog_size),
        "note": "Y=0 is censored at last follow-up, not lifelong disease-free confirmation. PFS months are not features.",
    }
    (output_dir / "label_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    results = {}
    histories = {}
    fold_rows = []
    oof_frames = []
    for model_name in MODEL_NAMES:
        result = cross_validate(model_name, cohort, args.folds, args.epochs, args.patience, args.seed)
        results[model_name] = result
        histories[model_name] = result["histories"]
        fold_rows.extend(result["rows"])
        oof_frames.append(
            pd.DataFrame(
                {
                    "patient_id": cohort.patient_ids,
                    "endpoint": "pfs",
                    "model": model_name,
                    "y": cohort.y,
                    "probability": result["oof_prob"],
                    "prediction": result["oof_pred"],
                    "threshold": result["oof_threshold"],
                }
            )
        )
    summary = _summary(pd.DataFrame(fold_rows))
    primary = summary[summary["endpoint"] == "pfs"].sort_values(["auprc_mean", "auroc_mean"], ascending=False)
    best_model = str(primary.iloc[0]["model"])
    LOGGER.info("Best PFS model by mean AUPRC: %s", best_model)

    sensitivity = _prepare_cohort(dataset_dir, "new_tumor_event", panel, prime_frame)
    sensitive = cross_validate(best_model, sensitivity, args.folds, args.epochs, args.patience, args.seed)
    fold_rows.extend(sensitive["rows"])
    oof_frames.append(
        pd.DataFrame(
            {
                "patient_id": sensitivity.patient_ids,
                "endpoint": "new_tumor_event",
                "model": best_model,
                "y": sensitivity.y,
                "probability": sensitive["oof_prob"],
                "prediction": sensitive["oof_pred"],
                "threshold": sensitive["oof_threshold"],
            }
        )
    )
    fold_frame = pd.DataFrame(fold_rows)
    summary = _summary(fold_frame)
    fold_frame.to_csv(output_dir / "metrics_folds.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    oof = pd.concat(oof_frames, ignore_index=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "loss_history.json").write_text(json.dumps(histories), encoding="utf-8")
    _plot_curves(oof, output_dir)
    _plot_bars(summary, output_dir)
    _plot_loss(histories, output_dir)
    best_oof = oof[(oof["model"] == best_model) & (oof["endpoint"] == "pfs")]
    _plot_confusion(best_oof["y"].to_numpy(), best_oof["prediction"].to_numpy(), best_model, output_dir)
    explanations = _explain(best_model, results[best_model], cohort, output_dir, args.explain_n)
    winning = summary[(summary["model"] == best_model) & (summary["endpoint"] == "pfs")].iloc[0]
    sensitive_row = summary[(summary["model"] == best_model) & (summary["endpoint"] == "new_tumor_event")]
    lines = [
        "# GNN recurrence benchmark",
        "",
        f"Primary endpoint PFS. Best model by mean AUPRC: **{best_model}**.",
        f"AUROC {winning['auroc_mean']:.3f} ± {winning['auroc_std']:.3f}; AUPRC {winning['auprc_mean']:.3f} ± {winning['auprc_std']:.3f}.",
        "",
        "Y=0 means censored at last follow-up. Age was z-scored inside each training fold. Outcome columns were not graph features.",
        "",
    ]
    if not sensitive_row.empty:
        row = sensitive_row.iloc[0]
        lines.append(
            f"Sensitivity on NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT with {best_model}: "
            f"AUROC {row['auroc_mean']:.3f} ± {row['auroc_std']:.3f}; AUPRC {row['auprc_mean']:.3f} ± {row['auprc_std']:.3f}."
        )
    lines.append(f"Explanation paths written: {len(explanations)}.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote benchmark artifacts to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
