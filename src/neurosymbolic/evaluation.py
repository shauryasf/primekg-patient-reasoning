"""Survival metrics shared by every model in the neuro-symbolic comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

from ..survival import harrell_c_index

HORIZONS = (12.0, 24.0, 36.0, 60.0)
UNO_TAU = 120.0


def _structured(time: np.ndarray, event: np.ndarray):
    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time, dtype=float))


def survival_metrics(time: np.ndarray, event: np.ndarray, risk: np.ndarray, fold_id: np.ndarray) -> dict[str, float]:
    """Harrell's C (pooled and per fold), Uno's C to 120 months, and time-dependent AUC.

    ``risk`` is out-of-fold: every value comes from a model that did not train on
    that patient. Patients without a follow-up time are dropped.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    keep = np.isfinite(time) & np.isfinite(risk) & (time > 0)
    per_fold = [
        harrell_c_index(time[keep & (fold_id == fold)], event[keep & (fold_id == fold)], risk[keep & (fold_id == fold)])
        for fold in sorted(set(fold_id.tolist()))
    ]
    survival = _structured(time[keep], event[keep])
    row = {
        "harrell_c": float(harrell_c_index(time[keep], event[keep], risk[keep])),
        "harrell_c_fold_mean": float(np.nanmean(per_fold)),
        "harrell_c_fold_std": float(np.nanstd(per_fold, ddof=1)),
        "uno_c_120": float(concordance_index_ipcw(survival, survival, risk[keep], tau=UNO_TAU)[0]),
    }
    horizons = [h for h in HORIZONS if time[keep][event[keep] == 1].min() < h < time[keep].max()]
    aucs, mean_auc = cumulative_dynamic_auc(survival, survival, risk[keep], horizons)
    for horizon, auc in zip(horizons, aucs):
        row[f"td_auc_{int(horizon)}"] = float(auc)
    row["td_auc_mean"] = float(mean_auc)
    return row


def brier_from_curves(time: np.ndarray, event: np.ndarray, grid: np.ndarray, curves: np.ndarray, max_time: float = 120.0) -> float:
    """Integrated Brier score of out-of-fold survival curves on ``grid`` up to ``max_time``."""
    time = np.asarray(time, dtype=float)
    keep = np.isfinite(time) & (time > 0)
    survival = _structured(time[keep], np.asarray(event)[keep])
    upper = min(max_time, float(np.max(time[keep][np.asarray(event)[keep] == 1])) - 1e-6)
    lower = float(np.min(time[keep])) + 1e-6
    points = grid[(grid > lower) & (grid < upper)]
    columns = np.searchsorted(grid, points)
    return float(integrated_brier_score(survival, survival, curves[keep][:, columns], points))


def metrics_frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).T
    frame.index.name = "model"
    return frame.reset_index()
