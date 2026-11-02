"""Survival Markov logic network (Cox-MLN).

A Markov logic network attaches a real weight to each first-order formula. The
probability of a world grows with ``exp(sum_k w_k n_k)``, where ``n_k`` is the
number of true groundings of formula ``k``. Here every formula concludes
``Progress(p)``:

    w_k : Body_k(p) => Progress(p)

Bodies are the ground atoms of :mod:`.atoms` (including the relational PrimeKG
atoms ``Altered(p, g) AND InPathway(g, w)``) and conjunctions of two atoms that
are mined on each training fold. All evidence atoms are observed and there is one
query atom per patient, so MLN inference factorises patient by patient into the
log-linear score ``sum_k w_k n_k(p)``. The usual conditional log-likelihood
(a logistic loss on ``Progress``) treats censored patients as negatives. We
instead learn the weights by maximising the elastic-net-penalised **Cox partial
likelihood** of progression-free survival, which uses each patient's follow-up
time and treats censoring correctly. The L1 part is structure learning: formulas
whose weight is driven to zero are pruned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from ...survival import harrell_c_index
from ..kg_evidence import logrank_table
from .atoms import AtomTable

L1_RATIO = 0.5
TOP_CONTEXT = 25
TOP_KG = 60
MAX_CONJUNCTIONS = 150
INNER_FOLDS = 3


@dataclass
class Formula:
    name: str
    text: str
    atoms: tuple[int, ...]


def grounding_counts(atoms: AtomTable, formulas: list[Formula]) -> np.ndarray:
    """``n_k(p)``: true groundings of each formula, with a sublinear ``log1p`` on counts.

    A patient with many co-amplified genes in one pathway would otherwise dominate
    the linear score through a single formula.
    """
    columns = []
    for formula in formulas:
        count = np.ones(len(atoms.patient_ids), dtype=np.float64)
        for atom in formula.atoms:
            count = count * atoms.values[:, atom]
        columns.append(np.log1p(count))
    return np.vstack(columns).T


def candidate_formulas(atoms: AtomTable, time: np.ndarray, event: np.ndarray, train: np.ndarray, proposed: list[tuple[int, ...]] | None = None) -> list[Formula]:
    """Unary formulas for every atom, plus conjunctions mined on the training fold only.

    ``proposed`` adds formulas found by another structure learner, such as the
    reinforcement-learned rules of :mod:`.rl`; the Cox-MLN then learns their weights.
    """
    formulas = [Formula(name, f"{text} ⇒ Progress(p)", (index,)) for index, (name, text) in enumerate(zip(atoms.names, atoms.formulas))]
    seen = {formula.atoms for formula in formulas}
    for key in proposed or []:
        key = tuple(sorted(key))
        if key in seen or len(key) < 2:
            continue
        seen.add(key)
        formulas.append(Formula(" & ".join(atoms.names[a] for a in key), " ∧ ".join(atoms.formulas[a] for a in key) + " ⇒ Progress(p)", key))
    binary = atoms.binary[train]
    table = logrank_table(time[train], event[train], binary.astype(float))
    strength = np.nan_to_num(np.abs(table["z"].to_numpy()))
    kinds = np.array(atoms.kinds)
    context = [i for i in np.argsort(-strength) if kinds[i] in ("clinical", "genomic")][:TOP_CONTEXT]
    knowledge = [i for i in np.argsort(-strength) if kinds[i].startswith("kg_")][:TOP_KG]
    pairs = [(a, b) for a in context for b in knowledge]
    pairs += [(a, b) for i, a in enumerate(context) for b in context[i + 1 :] if atoms.variables[a] != atoms.variables[b]]
    if not pairs:
        return formulas
    joint = np.vstack([binary[:, a] & binary[:, b] for a, b in pairs]).T
    share = joint.mean(axis=0)
    usable = (share >= 0.03) & (share <= 0.9)
    pairs = [pair for pair, keep in zip(pairs, usable) if keep]
    joint = joint[:, usable]
    if not pairs:
        return formulas
    z = np.nan_to_num(np.abs(logrank_table(time[train], event[train], joint.astype(float))["z"].to_numpy()))
    for position in np.argsort(-z)[:MAX_CONJUNCTIONS]:
        a, b = pairs[position]
        if tuple(sorted((a, b))) in seen:
            continue
        formulas.append(Formula(f"{atoms.names[a]} & {atoms.names[b]}", f"{atoms.formulas[a]} ∧ {atoms.formulas[b]} ⇒ Progress(p)", (a, b)))
    return formulas


def _standardise(train_x: np.ndarray, *others: np.ndarray):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return [(x - mean) / std for x in (train_x, *others)], mean, std


def fit_fold(atoms: AtomTable, time: np.ndarray, event: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int = 0, proposed: list[tuple[int, ...]] | None = None):
    """Learn formula weights on ``train`` and score ``test``. Alpha is chosen by inner CV on C-index."""
    train = train[np.isfinite(time[train]) & (time[train] > 0)]
    formulas = candidate_formulas(atoms, time, event, train, proposed)
    proposed_keys = {tuple(sorted(key)) for key in (proposed or [])}
    counts = grounding_counts(atoms, formulas)
    (x_train, x_test), _, std = _standardise(counts[train], counts[test])
    y_train = Surv.from_arrays(event[train].astype(bool), time[train])
    path = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, alpha_min_ratio=0.02, n_alphas=30, max_iter=200000)
    path.fit(x_train, y_train)
    alphas = path.alphas_
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    scores = np.zeros(len(alphas))
    for inner in range(INNER_FOLDS):
        held = order[inner::INNER_FOLDS]
        keep = np.setdiff1d(order, held)
        model = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, alphas=alphas, max_iter=200000)
        try:
            model.fit(x_train[keep], Surv.from_arrays(event[train][keep].astype(bool), time[train][keep]))
        except ArithmeticError:
            continue
        for index, alpha in enumerate(model.alphas_):
            risk = model.predict(x_train[held], alpha=alpha)
            value = harrell_c_index(time[train][held], event[train][held], risk)
            scores[index] += 0.0 if not np.isfinite(value) else value
    best = int(np.argmax(scores))
    alpha = float(alphas[best])
    risk_test = path.predict(x_test, alpha=alpha)
    coef = path.coef_[:, best]
    rows = []
    for formula, weight, scale in zip(formulas, coef, std):
        if abs(weight) < 1e-8:
            continue
        rows.append(
            {
                "formula": formula.text,
                "name": formula.name,
                "atoms": "|".join(atoms.names[a] for a in formula.atoms),
                "weight_per_sd": float(weight),
                "weight_per_grounding": float(weight / scale),
                "arity": len(formula.atoms),
                "source": "unary" if len(formula.atoms) == 1 else "rl" if formula.atoms in proposed_keys else "mined",
                "uses_kg": any(atoms.kinds[a].startswith("kg_") for a in formula.atoms),
                "support": int(np.all(atoms.binary[train][:, list(formula.atoms)], axis=1).sum()),
            }
        )
    weights = pd.DataFrame(rows).sort_values("weight_per_sd", key=np.abs, ascending=False) if rows else pd.DataFrame()
    return risk_test, weights, {"alpha": alpha, "candidates": len(formulas), "selected": len(rows), "inner_c": float(scores[best] / INNER_FOLDS)}
