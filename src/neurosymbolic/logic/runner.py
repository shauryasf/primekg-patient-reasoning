"""Run every symbolic reasoner on the shared folds and pick the report's engine.

All reasoners see the same ground atoms and the same five folds, so their
out-of-fold risks and C-indices are directly comparable with each other and
with the RuleKit baseline. The engine is chosen with the same one-standard-error
rule as the report model: among reasoners within one standard error of the best
fold-mean C-index, the earliest in :data:`PREFERENCE` wins. That order puts the
reasoner with the richest reasoning first: RL structure learning feeding Markov
logic weight learning.
"""
from __future__ import annotations

import time as clock

import numpy as np
import pandas as pd

from ...survival import harrell_c_index
from . import mln, psl, rl
from .atoms import AtomTable

METHODS = ("rl_mln", "mln", "psl", "rl")
PREFERENCE = ("rl_mln", "mln", "psl", "rl", "rulekit")
LABELS = {
    "rl_mln": "RL-structured survival MLN",
    "mln": "Survival MLN (Cox-MLN)",
    "psl": "Survival PSL (collective)",
    "rl": "RL rules (policy gradient)",
    "rulekit": "RuleKit LR-Rules (baseline)",
}


def fold_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray, folds) -> list[float]:
    values = []
    for _, test in folds:
        keep = test[np.isfinite(time[test]) & (time[test] > 0) & np.isfinite(risk[test])]
        values.append(float(harrell_c_index(time[keep], event[keep], risk[keep])))
    return values


def run_reasoners(atoms: AtomTable, time: np.ndarray, event: np.ndarray, folds, seed: int = 42, log=print):
    """Out-of-fold risk, learned weights, and timing for every reasoner."""
    n = len(atoms.patient_ids)
    risks = {method: np.full(n, np.nan) for method in METHODS}
    weights: dict[str, list[pd.DataFrame]] = {method: [] for method in METHODS}
    infos: dict[str, list[dict]] = {method: [] for method in METHODS}
    seconds = {method: 0.0 for method in METHODS}
    for fold, (train, test) in enumerate(folds):
        fold_seed = seed + fold
        timed = train[np.isfinite(time[train]) & (time[train] > 0)]

        started = clock.time()
        proposed, _, info = rl.propose_rules(atoms, time, event, timed, seed=fold_seed, limit=150)
        risk, table, mln_info = mln.fit_fold(atoms, time, event, train, test, seed=fold_seed, proposed=proposed)
        risks["rl_mln"][test] = risk
        weights["rl_mln"].append(table.assign(fold=fold))
        infos["rl_mln"].append({**info, **mln_info})
        seconds["rl_mln"] += clock.time() - started

        started = clock.time()
        risk, table, info = mln.fit_fold(atoms, time, event, train, test, seed=fold_seed)
        risks["mln"][test] = risk
        weights["mln"].append(table.assign(fold=fold))
        infos["mln"].append(info)
        seconds["mln"] += clock.time() - started

        started = clock.time()
        risk, table, info = psl.fit_fold(atoms, time, event, train, test, seed=fold_seed)
        risks["psl"][test] = risk
        weights["psl"].append(table.assign(fold=fold))
        infos["psl"].append(info)
        seconds["psl"] += clock.time() - started

        started = clock.time()
        risk, table, info = rl.fit_fold(atoms, time, event, train, test, seed=fold_seed)
        risks["rl"][test] = risk
        weights["rl"].append(table.assign(fold=fold))
        infos["rl"].append(info)
        seconds["rl"] += clock.time() - started
        log(f"logic fold {fold} done")
    return risks, {method: pd.concat(frames, ignore_index=True) for method, frames in weights.items()}, infos, seconds


def choose_engine(summary: pd.DataFrame, folds: int = 5) -> tuple[str, dict]:
    ranked = summary.sort_values("c_mean", ascending=False)
    best = ranked.iloc[0]
    bar = float(best["c_mean"]) - float(best["c_std"]) / np.sqrt(folds)
    eligible = ranked[ranked["c_mean"] >= bar]["method"].tolist()
    chosen = next(method for method in PREFERENCE if method in eligible)
    return chosen, {"best": str(best["method"]), "best_c": float(best["c_mean"]), "bar": bar, "eligible": eligible}
