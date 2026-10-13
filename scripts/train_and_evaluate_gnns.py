#!/usr/bin/env python
"""Stratified 5-fold benchmark of the patient-graph models on PFS recurrence.

Models: MLP, GCN, GATv2, GraphSAGE, R-GCN, RGAT, GIN, and a graph transformer.
Objectives: weighted BCE (default), a differentiable C-index surrogate, or the
Cox partial likelihood. The concordance objectives read ``PFS_MONTHS`` as the
follow-up axis of the target; it is never a model feature.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_fusion import (
    CLINICAL_DIM,
    FEATURE_DIM,
    build_cohort,
    load_fusion_config,
    load_gene_panel,
    load_primekg_slice,
    relations_from_codes,
    standardize_ages,
)
from src.gsl import edge_report
from src.downloads import ensure_brca_dataset, ensure_primekg_csv
from src.explain import enumerate_explanation_paths, score_edges
from src.gnn_models import GRAPH_ARCHITECTURES, MODEL_NAMES, build_model
from src.labels import LANDMARK_MONTHS, load_recurrence_labels
from src.survival import (
    DEFAULT_TEMPERATURE,
    LOSS_NAMES,
    SURVIVAL_ENDPOINTS,
    align_survival_targets,
    build_objective,
    concordance_audit,
    harrell_c_index,
    load_survival_targets,
)

LOGGER = logging.getLogger("train_gnns")
COLORS = {
    "clinical_lr": "#72B7B2",
    "mlp": "#4C78A8",
    "gcn": "#F58518",
    "gatv2": "#54A24B",
    "graphsage": "#E45756",
    "rgcn": "#B279A2",
    "rgat": "#9D755D",
    "gin": "#EECA3B",
    "transformer": "#79706E",
}
# Keep the default benchmark focused on patient-graph signal and avoid the
# slow, lower-value tabular/attention variants that do not materially improve the
# recurrence signal in this cohort. GraphSAGE is the lightweight comparator that
# performed best in the focused three-fold check.
COMPARISON_MODELS = ("rgcn", "graphsage")
SUPPORTED_MODELS = ("clinical_lr",) + MODEL_NAMES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray, max_candidates: int = 200) -> float:
    """Threshold with the best validation F1.

    Candidate cut points are the predicted scores themselves, so the search does
    not depend on the scale of the score. A fixed 0.05 .. 0.95 grid on
    probabilities would miss every cut point for a model trained on a ranking
    objective, whose sigmoid is not calibrated.
    """
    values = np.asarray(probabilities, dtype=float)
    candidates = np.unique(values[np.isfinite(values)])
    if candidates.size == 0:
        return 0.5
    if candidates.size > max_candidates:
        keep = np.linspace(0, candidates.size - 1, max_candidates).astype(int)
        candidates = candidates[np.unique(keep)]
    best_threshold, best_score = float(candidates[0]), -1.0
    for threshold in candidates:
        score = f1_score(y_true, values >= threshold, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


@dataclass
class TrainOptions:
    """Training knobs shared by every fold of one run."""

    loss: str = "bce"
    aux_bce: float = 0.0
    temperature: float = DEFAULT_TEMPERATURE
    batch_size: int = 32
    early_stop: str = "auprc"

    def validate(self) -> "TrainOptions":
        if self.loss not in LOSS_NAMES:
            raise ValueError(f"--loss must be one of {LOSS_NAMES}, received {self.loss!r}")
        if self.early_stop not in {"auprc", "cindex"}:
            raise ValueError(f"--early-stop must be auprc or cindex, received {self.early_stop!r}")
        if self.batch_size < 1:
            raise ValueError(f"--batch-size must be positive, received {self.batch_size}")
        return self

    def audit(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "aux_bce_weight": float(self.aux_bce),
            "temperature": float(self.temperature) if self.loss == "cindex" else None,
            "batch_size": int(self.batch_size),
            "early_stop": self.early_stop,
        }


def _early_stop_score(
    metric: str,
    y_validation: np.ndarray,
    probabilities: np.ndarray,
    time_validation: np.ndarray | None,
    event_validation: np.ndarray | None,
) -> float:
    """Validation score used for model selection. Higher is better."""
    if metric == "cindex":
        if time_validation is None or event_validation is None:
            return 0.0
        value = harrell_c_index(time_validation, event_validation, probabilities)
        return 0.0 if not np.isfinite(value) else float(value)
    if len(np.unique(y_validation)) < 2:
        return 0.0
    return float(average_precision_score(y_validation, probabilities))


def _graph_logits(model, batch) -> torch.Tensor:
    return model(
        batch.x,
        batch.edge_index,
        batch.emb_id,
        batch.is_patient,
        batch.batch,
        focus=getattr(batch, "focus", None),
        rel_code=getattr(batch, "rel_code", None),
        hop=getattr(batch, "hop", None),
    )


def _run_graphs(model, graphs, indices, objective, optimizer=None, batch_size: int = 32) -> float:
    subset = [graphs[int(index)] for index in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=optimizer is not None)
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad()
        logits = _graph_logits(model, batch)
        loss = objective(
            logits,
            batch.y.view(-1),
            getattr(batch, "time", None),
            getattr(batch, "event", None),
        )
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


def _run_mlp(
    model,
    features,
    indices,
    y,
    objective,
    optimizer=None,
    batch_size: int = 64,
    time: np.ndarray | None = None,
    event: np.ndarray | None = None,
) -> float:
    order = np.array(indices)
    if optimizer is not None:
        order = order.copy()
        np.random.shuffle(order)
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for start in range(0, len(order), batch_size):
        batch_index = order[start : start + batch_size]
        inputs = torch.tensor(features[batch_index], dtype=torch.float)
        target = torch.tensor(y[batch_index], dtype=torch.float)
        batch_time = None if time is None else torch.tensor(time[batch_index], dtype=torch.float)
        batch_event = None if event is None else torch.tensor(event[batch_index], dtype=torch.float)
        if optimizer is not None:
            optimizer.zero_grad()
        loss = objective(model(inputs), target, batch_time, batch_event)
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


def _metric_row(
    model: str,
    endpoint: str,
    fold: int,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    time: np.ndarray | None = None,
    event: np.ndarray | None = None,
) -> dict[str, object]:
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
        "c_index": float("nan"),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
    }
    if len(np.unique(y_true)) > 1:
        row["auroc"] = float(roc_auc_score(y_true, probabilities))
        row["auprc"] = float(average_precision_score(y_true, probabilities))
    if time is not None and event is not None:
        row["c_index"] = float(harrell_c_index(time, event, probabilities))
    return row


def cross_validate(
    model_name: str,
    cohort,
    folds: int,
    epochs: int,
    patience: int,
    seed: int,
    fusion=None,
    options: TrainOptions | None = None,
    survival: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, object]:
    options = (options or TrainOptions()).validate()
    survival_time, survival_event = survival if survival is not None else (None, None)
    if options.loss != "bce" and (survival_time is None or survival_event is None):
        raise ValueError(f"--loss {options.loss} needs the follow-up time and event indicator as targets")
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
        model = build_model(
            model_name,
            FEATURE_DIM,
            cohort.catalog_size,
            features.shape[1],
            mlp_clinical_dim=CLINICAL_DIM,
            fusion=fusion,
        )
        gsl = getattr(model, "gsl", None)
        if gsl is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        else:
            gsl_ids = {id(parameter) for parameter in gsl.parameters()}
            graph_parameters = [parameter for parameter in model.parameters() if id(parameter) not in gsl_ids]
            optimizer = torch.optim.Adam(
                [
                    {"params": graph_parameters, "lr": 1e-3},
                    {"params": list(gsl.parameters()), "lr": float(fusion.gsl_learning_rate) if fusion is not None else 1e-4, "weight_decay": 0.0},
                ],
                weight_decay=1e-4,
            )
        objective = build_objective(options.loss, cohort.y[inner_train], aux_bce=options.aux_bce, temperature=options.temperature)
        best_state, best_score, wait = None, -1.0, 0
        history = []
        for epoch in range(epochs):
            if model_name == "mlp":
                train_loss = _run_mlp(
                    model,
                    features,
                    inner_train,
                    cohort.y,
                    objective,
                    optimizer,
                    batch_size=64,
                    time=survival_time,
                    event=survival_event,
                )
                val_prob = _predict_mlp(model, features, validation)
            else:
                train_loss = _run_graphs(
                    model, cohort.graphs, inner_train, objective, optimizer, batch_size=options.batch_size
                )
                val_prob = _predict_graphs(model, cohort.graphs, validation)
            val_score = _early_stop_score(
                options.early_stop,
                cohort.y[validation],
                val_prob,
                None if survival_time is None else survival_time[validation],
                None if survival_event is None else survival_event[validation],
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_auprc": _early_stop_score("auprc", cohort.y[validation], val_prob, None, None),
                    "val_c_index": _early_stop_score(
                        "cindex",
                        cohort.y[validation],
                        val_prob,
                        None if survival_time is None else survival_time[validation],
                        None if survival_event is None else survival_event[validation],
                    ),
                }
            )
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
        rows.append(
            _metric_row(
                model_name,
                cohort.endpoint,
                fold,
                cohort.y[test_idx],
                test_prob,
                threshold,
                None if survival_time is None else survival_time[test_idx],
                None if survival_event is None else survival_event[test_idx],
            )
        )
        histories.append(history)
        states.append(copy.deepcopy(model.state_dict()))
        train_folds.append(np.array(train_idx))
        oof_prob[test_idx] = test_prob
        oof_pred[test_idx] = (test_prob >= threshold).astype(int)
        oof_threshold[test_idx] = threshold
        LOGGER.info(
            "%s fold %d %s AUROC %.3f AUPRC %.3f C-index %.3f epochs %d",
            model_name,
            fold,
            cohort.endpoint,
            rows[-1]["auroc"],
            rows[-1]["auprc"],
            rows[-1]["c_index"],
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


def cross_validate_clinical_lr(
    cohort,
    folds: int,
    seed: int,
    survival: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, object]:
    """Logistic regression on the 19 clinical fields. Alterations and PrimeKG are unused."""
    survival_time, survival_event = survival if survival is not None else (None, None)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    oof_prob = np.full(len(cohort.y), np.nan)
    oof_pred = np.full(len(cohort.y), -1)
    oof_threshold = np.full(len(cohort.y), np.nan)
    train_folds = []
    clinical = cohort.mlp_x[:, :CLINICAL_DIM].astype(np.float64)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cohort.y)), cohort.y)):
        inner_train, validation = train_test_split(
            train_idx, test_size=0.15, stratify=cohort.y[train_idx], random_state=seed + fold
        )
        scaler = StandardScaler()
        train_x = scaler.fit_transform(np.nan_to_num(clinical[inner_train], nan=0.0))
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(train_x, cohort.y[inner_train])
        val_prob = model.predict_proba(scaler.transform(np.nan_to_num(clinical[validation], nan=0.0)))[:, 1]
        test_prob = model.predict_proba(scaler.transform(np.nan_to_num(clinical[test_idx], nan=0.0)))[:, 1]
        threshold = best_f1_threshold(cohort.y[validation], val_prob)
        rows.append(
            _metric_row(
                "clinical_lr",
                cohort.endpoint,
                fold,
                cohort.y[test_idx],
                test_prob,
                threshold,
                None if survival_time is None else survival_time[test_idx],
                None if survival_event is None else survival_event[test_idx],
            )
        )
        train_folds.append(np.array(train_idx))
        oof_prob[test_idx] = test_prob
        oof_pred[test_idx] = (test_prob >= threshold).astype(int)
        oof_threshold[test_idx] = threshold
        LOGGER.info(
            "clinical_lr fold %d %s AUROC %.3f AUPRC %.3f C-index %.3f",
            fold,
            cohort.endpoint,
            rows[-1]["auroc"],
            rows[-1]["auprc"],
            rows[-1]["c_index"],
        )
    return {
        "rows": rows,
        "histories": [],
        "states": [],
        "train_indices": train_folds,
        "oof_prob": oof_prob,
        "oof_pred": oof_pred,
        "oof_threshold": oof_threshold,
    }


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auroc", "auprc", "c_index", "f1", "balanced_accuracy"]
    rows = []
    for (model, endpoint), group in frame.groupby(["model", "endpoint"], sort=False):
        row = {"model": model, "endpoint": endpoint, "folds": int(len(group))}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            # A fold can have no comparable pair. Report the mean of the folds that do.
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_curves(oof: pd.DataFrame, output_dir: Path, endpoint: str) -> None:
    primary = oof[oof["endpoint"] == endpoint]
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
    axis.set_title(f"{endpoint} ROC (out-of-fold)")
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
    axis.set_title(f"{endpoint} precision-recall (out-of-fold)")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curves.png", dpi=160)
    plt.close(fig)


def _plot_bars(summary: pd.DataFrame, output_dir: Path, endpoint: str) -> None:
    primary = summary[summary["endpoint"] == endpoint].copy()
    order = [model for model in COMPARISON_MODELS if model in set(primary["model"])]
    primary = primary.set_index("model").loc[order].reset_index()
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    positions = np.arange(len(primary))
    width = 0.27
    axis.bar(
        positions - width,
        primary["auroc_mean"],
        width,
        yerr=primary["auroc_std"],
        color="#4C78A8",
        label="AUROC",
        capsize=3,
    )
    axis.bar(
        positions,
        primary["auprc_mean"],
        width,
        yerr=primary["auprc_std"],
        color="#F58518",
        label="AUPRC",
        capsize=3,
    )
    axis.bar(
        positions + width,
        primary["c_index_mean"],
        width,
        yerr=primary["c_index_std"],
        color="#B279A2",
        label="C-index (Harrell)",
        capsize=3,
    )
    axis.set_xticks(positions, primary["model"], rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"5-fold mean ± std on {endpoint}")
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


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, model: str, output_dir: Path, endpoint: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(4.6, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    negative = "Event-free at 36m" if endpoint == "pfs_36m" else "Censored"
    positive = "Progression by 36m" if endpoint == "pfs_36m" else "Progression"
    axis.set_yticks([0, 1], [negative, positive])
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    axis.set_title(f"{model} out-of-fold confusion")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _explain(model_name: str, result: dict, cohort, output_dir: Path, limit: int, fusion=None) -> pd.DataFrame:
    order = np.argsort(-result["oof_prob"])
    chosen = [int(index) for index in order if cohort.y[int(index)] == 1 and not np.isnan(result["oof_prob"][int(index)])][:limit]
    rows = []
    for index in chosen:
        fold = next(fold for fold, train_idx in enumerate(result["train_indices"]) if index not in set(int(item) for item in train_idx.tolist()))
        standardize_ages(cohort.graphs, cohort.mlp_x, result["train_indices"][fold])
        model = build_model(
            model_name,
            FEATURE_DIM,
            cohort.catalog_size,
            cohort.mlp_x.shape[1],
            mlp_clinical_dim=CLINICAL_DIM,
            fusion=fusion,
        )
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
    if rows and fusion is not None and (fusion.gsl_enabled or fusion.mode == "gsl"):
        _write_gsl_report(model_name, result, cohort, output_dir, limit, fusion)
    return frame


def _write_gsl_report(model_name: str, result: dict, cohort, output_dir: Path, limit: int, fusion) -> None:
    order = np.argsort(-result["oof_prob"])
    chosen = [int(index) for index in order if cohort.y[int(index)] == 1 and not np.isnan(result["oof_prob"][int(index)])][:limit]
    kept_rows = []
    for index in chosen:
        fold = next(fold for fold, train_idx in enumerate(result["train_indices"]) if index not in set(int(item) for item in train_idx.tolist()))
        standardize_ages(cohort.graphs, cohort.mlp_x, result["train_indices"][fold])
        model = build_model(model_name, FEATURE_DIM, cohort.catalog_size, cohort.mlp_x.shape[1], mlp_clinical_dim=CLINICAL_DIM, fusion=fusion)
        model.load_state_dict(result["states"][fold])
        model.eval()
        data = cohort.graphs[index]
        weight = model.edge_weights(data.x, data.edge_index, data.rel_code, data.is_patient, data.hop, None)
        if weight is None:
            continue
        for row in edge_report(data, cohort.metas[index], weight, fusion.gsl_threshold, model.gsl.last_hop_gate):
            row["patient_id"] = cohort.patient_ids[index]
            kept_rows.append(row)
    if kept_rows:
        pd.DataFrame(kept_rows).to_csv(output_dir / "gsl_edges.csv", index=False)


def _prepare_cohort(dataset_dir: Path, endpoint: str, panel, prime_frame, fusion=None):
    labels = load_recurrence_labels(dataset_dir, endpoint)
    cohort = build_cohort(dataset_dir, labels, prime_frame, panel, fusion=fusion)
    LOGGER.info("%s cohort: %d patients, %d positives", endpoint, len(cohort.y), int(cohort.y.sum()))
    return cohort


def _attach_survival_targets(cohort, targets) -> tuple[np.ndarray, np.ndarray]:
    """Copy the follow-up target onto every graph and return the aligned arrays.

    ``time`` and ``event`` ride on the PyG ``Data`` object next to ``y`` so a
    mini-batch carries its own targets. They are targets only: the feature vector
    and the clinical block are untouched, which keeps the follow-up axis out of
    the model input.
    """
    time, event, usable = align_survival_targets(targets, cohort.patient_ids)
    for graph, row_time, row_event in zip(cohort.graphs, time, event):
        graph.time = torch.tensor([float(row_time)], dtype=torch.float)
        graph.event = torch.tensor([float(row_event)], dtype=torch.float)
    LOGGER.info(
        "%s concordance targets: %d patients, %d events, %d with follow-up time",
        targets.endpoint,
        len(time),
        int(event.sum()),
        int(usable.sum()),
    )
    return time, event


def _resolve_models(value: str | None, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    if value is None:
        return COMPARISON_MODELS
    chosen = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [name for name in chosen if name not in SUPPORTED_MODELS]
    if unknown or not chosen:
        parser.error(f"--models must be a comma-separated subset of {SUPPORTED_MODELS}; received {value!r}")
    return chosen


def _run_model(
    model_name: str,
    cohort,
    folds: int,
    epochs: int,
    patience: int,
    seed: int,
    fusion=None,
    options: TrainOptions | None = None,
    survival: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, object]:
    options = options or TrainOptions()
    if model_name == "clinical_lr":
        return cross_validate_clinical_lr(cohort, folds, seed, survival=survival)
    return cross_validate(
        model_name, cohort, folds, epochs, patience, seed, fusion=fusion, options=options, survival=survival
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--endpoint", choices=("pfs", "pfs_36m", "new_tumor_event"), default="pfs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explain-n", type=int, default=8)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--fusion-mode", choices=("fixed", "gsl"), default=None)
    parser.add_argument("--fixed-hops", type=int, choices=(0, 1, 2, 3), default=None)
    parser.add_argument("--max-candidate-hops", type=int, default=None)
    parser.add_argument(
        "--models",
        default=None,
        help=f"comma-separated subset of {SUPPORTED_MODELS}; the default runs {COMPARISON_MODELS}",
    )
    parser.add_argument(
        "--loss",
        choices=LOSS_NAMES,
        default="bce",
        help="bce: binary cross-entropy (default). cindex: differentiable C-index surrogate. cox: Breslow partial likelihood.",
    )
    parser.add_argument(
        "--aux-bce",
        type=float,
        default=0.0,
        help="weight of an extra BCE term added to a concordance objective; 0 keeps the ranking objective pure",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="soft C-index temperature")
    parser.add_argument("--batch-size", type=int, default=32, help="mini-batch size for the graph models")
    parser.add_argument(
        "--early-stop",
        choices=("auprc", "cindex"),
        default=None,
        help="validation metric for model selection; defaults to cindex for a concordance loss and auprc otherwise",
    )
    parser.add_argument(
        "--rank-by",
        choices=("auprc", "auroc", "cindex"),
        default="auprc",
        help="metric that picks the headline model and the explained model",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "gnn_benchmark")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    models = _resolve_models(args.models, parser)
    options = TrainOptions(
        loss=args.loss,
        aux_bce=args.aux_bce,
        temperature=args.temperature,
        batch_size=args.batch_size,
        early_stop=args.early_stop or ("auprc" if args.loss == "bce" else "cindex"),
    )
    try:
        options.validate()
    except ValueError as error:
        parser.error(str(error))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ensure_brca_dataset() if args.dataset_dir is None else args.dataset_dir
    panel = load_gene_panel()
    panel_entrez = {str(info["entrez"]) for info in panel.values() if info.get("entrez")}
    prime_csv = ensure_primekg_csv()
    prime_frame = load_primekg_slice(prime_csv, panel_entrez)
    LOGGER.info("PrimeKG slice edges: %d", len(prime_frame))
    endpoint = args.endpoint
    fusion = load_fusion_config()
    if args.fusion_mode is not None:
        fusion.mode = args.fusion_mode
        fusion.gsl_enabled = args.fusion_mode == "gsl"
    if args.fixed_hops is not None:
        fusion.fixed_hops = args.fixed_hops
    if args.max_candidate_hops is not None:
        fusion.max_candidate_hops = args.max_candidate_hops
    cohort = _prepare_cohort(dataset_dir, endpoint, panel, prime_frame, fusion)
    survival = None
    survival_targets = None
    if endpoint in SURVIVAL_ENDPOINTS:
        survival_targets = load_survival_targets(dataset_dir, endpoint)
        survival = _attach_survival_targets(cohort, survival_targets)
    elif options.loss != "bce":
        LOGGER.warning("%s has no event time; falling back to --loss bce", endpoint)
        options = replace(options, loss="bce", early_stop="auprc", aux_bce=0.0)
    LOGGER.info(
        "Objective %s (early stop %s, auxiliary BCE %.2f, batch %d)",
        options.loss,
        options.early_stop,
        options.aux_bce,
        options.batch_size,
    )
    if endpoint == "pfs_36m":
        note = (
            f"Y=1 is progression by {LANDMARK_MONTHS} months. "
            "Patients censored earlier are dropped. PFS months define the window and are not features."
        )
    else:
        note = "Y=0 is censored at last follow-up, not lifelong disease-free confirmation. PFS months are not features."
    audit = {
        "endpoint": endpoint,
        "patients": int(len(cohort.y)),
        "positives": int(cohort.y.sum()),
        "negatives": int((cohort.y == 0).sum()),
        "primekg_edges": int(len(prime_frame)),
        "feature_dim": FEATURE_DIM,
        "catalog_size": int(cohort.catalog_size),
        "fusion_mode": fusion.mode,
        "candidate_hops": int(fusion.candidate_hops),
        "gsl_enabled": bool(fusion.gsl_enabled or fusion.mode == "gsl"),
        "models": list(models),
        "objective": options.audit(),
        "rank_by": args.rank_by,
        "survival_target": None if survival_targets is None else survival_targets.audit(),
        "survival_pairs": None if survival is None else concordance_audit(survival[0], survival[1]),
        "c_index_note": (
            "C-index is Harrell's concordance over comparable pairs of (PFS_MONTHS, PFS_STATUS). "
            "A censored patient is only ever the later member of a pair. Time and event are targets, never features."
        ),
        "note": note,
    }
    (output_dir / "label_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    results = {}
    histories = {}
    fold_rows = []
    oof_frames = []
    for model_name in models:
        result = _run_model(
            model_name, cohort, args.folds, args.epochs, args.patience, args.seed, fusion, options, survival
        )
        results[model_name] = result
        histories[model_name] = result["histories"]
        fold_rows.extend(result["rows"])
        oof_frame = {
            "patient_id": cohort.patient_ids,
            "endpoint": endpoint,
            "model": model_name,
            "y": cohort.y,
            "probability": result["oof_prob"],
            "prediction": result["oof_pred"],
            "threshold": result["oof_threshold"],
        }
        if survival is not None:
            # The follow-up target is written alongside the score so the C-index in
            # metrics_folds.csv can be recomputed from this file alone.
            oof_frame["follow_up_months"] = survival[0]
            oof_frame["event"] = survival[1]
        oof_frames.append(pd.DataFrame(oof_frame))
    summary = _summary(pd.DataFrame(fold_rows))
    rank_column = {
        "auprc": ["auprc_mean", "auroc_mean"],
        "auroc": ["auroc_mean", "auprc_mean"],
        "cindex": ["c_index_mean", "auprc_mean"],
    }[args.rank_by]
    primary = summary[summary["endpoint"] == endpoint].sort_values(rank_column, ascending=False)
    row = primary.iloc[0]
    best_model = str(row["model"])
    graph_primary = primary[primary["model"].isin(GRAPH_ARCHITECTURES)]
    explain_model = str(graph_primary.iloc[0]["model"]) if not graph_primary.empty else best_model
    LOGGER.info("Best %s model by mean %s: %s", endpoint, args.rank_by, best_model)

    run_sensitivity = endpoint == "pfs" and not args.skip_sensitivity and explain_model in MODEL_NAMES
    if run_sensitivity:
        sensitivity = _prepare_cohort(dataset_dir, "new_tumor_event", panel, prime_frame)
        # The new-tumor-event endpoint has no event time, so this comparison stays binary.
        sensitive = cross_validate(
            explain_model,
            sensitivity,
            args.folds,
            args.epochs,
            args.patience,
            args.seed,
            fusion,
            options=replace(options, loss="bce", early_stop="auprc", aux_bce=0.0),
        )
        fold_rows.extend(sensitive["rows"])
        oof_frames.append(
            pd.DataFrame(
                {
                    "patient_id": sensitivity.patient_ids,
                    "endpoint": "new_tumor_event",
                    "model": explain_model,
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
    _plot_curves(oof, output_dir, endpoint)
    _plot_bars(summary, output_dir, endpoint)
    _plot_loss(histories, output_dir)
    best_oof = oof[(oof["model"] == best_model) & (oof["endpoint"] == endpoint)]
    _plot_confusion(best_oof["y"].to_numpy(), best_oof["prediction"].to_numpy(), best_model, output_dir, endpoint)
    explanations = _explain(explain_model, results[explain_model], cohort, output_dir, args.explain_n, fusion)
    winning = summary[(summary["model"] == best_model) & (summary["endpoint"] == endpoint)].iloc[0]
    sensitive_row = summary[(summary["model"] == explain_model) & (summary["endpoint"] == "new_tumor_event")]
    lines = [
        "# GNN recurrence benchmark",
        "",
        f"Primary endpoint {endpoint}. Best model by mean {args.rank_by}: **{best_model}**.",
        f"AUROC {winning['auroc_mean']:.3f} ± {winning['auroc_std']:.3f}; "
        f"AUPRC {winning['auprc_mean']:.3f} ± {winning['auprc_std']:.3f}; "
        f"C-index {winning['c_index_mean']:.3f} ± {winning['c_index_std']:.3f}.",
        "",
        f"Objective: `{options.loss}` (early stop on {options.early_stop}, auxiliary BCE weight {options.aux_bce:g}).",
        "",
        note,
        "",
        "C-index is Harrell's concordance over comparable pairs of (PFS_MONTHS, PFS_STATUS). "
        "A censored patient is only ever the later member of a pair, so the concordance objectives do not treat "
        "`Y = 0` as lifelong disease-free. Time and event are targets, never features.",
        "",
        "Graph and MLP scores add a clinical linear head to the alteration or graph score. Pooling uses the patient and altered genes.",
        "",
    ]
    if not sensitive_row.empty:
        row = sensitive_row.iloc[0]
        lines.append(
            f"Sensitivity on NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT with {explain_model}: "
            f"AUROC {row['auroc_mean']:.3f} ± {row['auroc_std']:.3f}; AUPRC {row['auprc_mean']:.3f} ± {row['auprc_std']:.3f}."
        )
    lines.append(f"Explanation paths written: {len(explanations)}.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote benchmark artifacts to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
