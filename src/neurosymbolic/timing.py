"""Progression-timeline groups and the predicted month of progression.

Reports are grouped by the **recorded** progression month, so a group never mixes
true progressors with patients who were never seen to progress:

====  =====================  =================
code  group                  recorded PFS
====  =====================  =================
A     early progression      ≤ 12 months
B     progression year 2     12–24 months
C     progression year 3     24–36 months
D     late progression       > 36 months
N     no progression         censored
====  =====================  =================

The model's own estimate of *when* a patient progresses is read off the
patient's out-of-fold survival curve. Because most patients never progress, the
plain median of ``S(t)`` is usually undefined. The estimate is therefore the
**conditional** time to progression, given that progression happens before a
horizon ``tau``: the month at which ``F(t) / F(tau)`` reaches one half, with
``F = 1 - S``. Two more estimators fit log-normal and log-logistic accelerated
failure time models on the training-fold progressors only. All are scored on
held-out progressors against a constant baseline, and the one with the lowest
mean absolute error is the report's estimate.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter

# Fitted on training-fold progressors only, so each predicts "when, given progression".
AFT_FAMILIES = {"lognormal_aft_on_progressors": LogNormalAFTFitter, "loglogistic_aft_on_progressors": LogLogisticAFTFitter}
AFT_PENALIZER = 0.5

GROUPS = (
    ("A", "Early progression", 0.0, 12.0),
    ("B", "Progression in year 2", 12.0, 24.0),
    ("C", "Progression in year 3", 24.0, 36.0),
    ("D", "Late progression", 36.0, math.inf),
)
NO_PROGRESSION = ("N", "No progression recorded")
GROUP_NAMES = {code: name for code, name, _, _ in GROUPS} | {NO_PROGRESSION[0]: NO_PROGRESSION[1]}
GROUP_RANGES = {code: (low, high) for code, _, low, high in GROUPS}
HORIZONS = (12.0, 24.0, 36.0, 60.0)
TAUS = (60.0, 120.0, 180.0)


def window_code(months: float) -> str:
    for code, _, low, high in GROUPS:
        if low < months <= high or (low == 0.0 and months <= high):
            return code
    return GROUPS[-1][0]


def timeline_group(event: int, months: float) -> str:
    """Report group from the recorded outcome. Censored patients are group N."""
    if int(event) != 1 or not np.isfinite(months):
        return NO_PROGRESSION[0]
    return window_code(float(months))


def window_label(code: str) -> str:
    low, high = GROUP_RANGES[code]
    if math.isinf(high):
        return f"> {low:g} months"
    if low == 0:
        return f"≤ {high:g} months"
    return f"{low:g}–{high:g} months"


def conditional_quantile(curve: np.ndarray, grid: np.ndarray, tau: float, q: float) -> float:
    """Month where ``F(t) / F(tau)`` first reaches ``q`` (``nan`` if ``F(tau)`` is 0)."""
    failure = 1.0 - curve
    upto = grid <= tau
    total = failure[upto][-1]
    if total <= 1e-9:
        return float("nan")
    share = failure[upto] / total
    hit = np.nonzero(share >= q)[0]
    return float(grid[upto][hit[0]]) if len(hit) else float(tau)


def _score(predicted: np.ndarray, observed: np.ndarray, low: np.ndarray | None = None, high: np.ndarray | None = None) -> dict[str, float]:
    keep = np.isfinite(predicted) & np.isfinite(observed)
    error = np.abs(predicted[keep] - observed[keep])
    windows = np.array([window_code(value) for value in predicted[keep]])
    truth = np.array([window_code(value) for value in observed[keep]])
    row = {
        "patients": int(keep.sum()),
        "mae_months": float(error.mean()),
        "median_abs_error_months": float(np.median(error)),
        "within_12_months": float((error <= 12).mean()),
        "window_accuracy": float((windows == truth).mean()),
    }
    if low is not None and high is not None:
        row["interval_coverage"] = float(((observed[keep] >= low[keep]) & (observed[keep] <= high[keep])).mean())
    return row


def progression_timing(curves: np.ndarray, study, grid: np.ndarray, features: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-patient progression probabilities and predicted month, plus the held-out evaluation."""
    time, event = study.time, study.event
    table = pd.DataFrame(index=range(len(time)))
    for horizon in HORIZONS:
        column = np.searchsorted(grid, horizon)
        table[f"p_progress_{int(horizon)}m"] = 1.0 - curves[:, column]
    estimates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for tau in TAUS:
        quantiles = np.array([[conditional_quantile(curve, grid, tau, q) for q in (0.25, 0.5, 0.75)] for curve in curves])
        estimates[f"conditional_median_tau{int(tau)}"] = (quantiles[:, 1], quantiles[:, 0], quantiles[:, 2])
    constant = np.full(len(time), np.nan)
    constant_low = np.full(len(time), np.nan)
    constant_high = np.full(len(time), np.nan)
    aft = {name: np.full((len(time), 3), np.nan) for name in AFT_FAMILIES}
    for train, test in study.folds:
        progressed = train[(event[train] == 1) & np.isfinite(time[train])]
        constant[test] = np.median(time[progressed])
        constant_low[test], constant_high[test] = np.percentile(time[progressed], [25, 75])
        if features is None:
            continue
        x = features.iloc[progressed]
        keep = x.columns[x.std(ddof=0) > 0]
        mean, std = x[keep].mean(), x[keep].std(ddof=0)
        frame = ((x[keep] - mean) / std).assign(T=time[progressed], E=1)
        test_x = (features.iloc[test][keep] - mean) / std
        for name, family in AFT_FAMILIES.items():
            fitter = family(penalizer=AFT_PENALIZER).fit(frame, duration_col="T", event_col="E")
            # predict_percentile(p) is the month where survival equals p, so p = 0.75 is the 25th percentile.
            aft[name][test, 0] = fitter.predict_percentile(test_x, p=0.75).to_numpy()
            aft[name][test, 1] = fitter.predict_median(test_x).to_numpy()
            aft[name][test, 2] = fitter.predict_percentile(test_x, p=0.25).to_numpy()
    estimates["training_progressor_median"] = (constant, constant_low, constant_high)
    if features is not None:
        for name, values in aft.items():
            estimates[name] = (values[:, 1], values[:, 0], values[:, 2])
    progressed = (event == 1) & np.isfinite(time)
    evaluation = {
        name: _score(values[0][progressed], time[progressed], values[1][progressed], values[2][progressed])
        for name, values in estimates.items()
    }
    candidates = {name: row for name, row in evaluation.items() if name != "training_progressor_median"}
    best = min(candidates, key=lambda name: (candidates[name]["mae_months"], -candidates[name]["window_accuracy"]))
    month, low, high = estimates[best]
    table["predicted_month"] = month
    table["predicted_low"] = low
    table["predicted_high"] = high
    table["predicted_window"] = [window_code(value) if np.isfinite(value) else "" for value in month]
    table["recorded_group"] = [timeline_group(e, t) for e, t in zip(event, time)]
    table["time"] = time
    table["event"] = event
    summary = {"estimators": evaluation, "chosen_estimator": best, "chosen": evaluation[best], "baseline": evaluation["training_progressor_median"]}
    return table, summary
