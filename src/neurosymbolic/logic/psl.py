"""Survival probabilistic soft logic (S-PSL) with collective inference.

PSL relaxes logic to truth values in ``[0, 1]`` with the Lukasiewicz t-norm, so a
weighted rule ``w : B => H`` becomes the hinge potential ``w * max(0, B - H)^2``
(its distance to satisfaction). MAP inference minimises the sum of potentials,
which is convex. The program here has four rule families over the query atom
``HighRisk(p)``:

    w+_k : Fact_k(p)                     =>  HighRisk(p)      (risk rules)
    w-_k : Fact_k(p)                     =>  !HighRisk(p)     (protective rules)
    s+   : Similar(p, q) & HighRisk(q)   =>  HighRisk(p)      (collective)
    s-   : Similar(p, q) & !HighRisk(q)  =>  !HighRisk(p)     (collective)
    b    : !HighRisk(p)                                        (prior)

``Fact_k`` are the strongest ground atoms of :mod:`.atoms` on the training fold,
with relational PrimeKG atoms softened to ``min(1, n / 2)`` groundings.
``Similar(p, q)`` links each patient to their ten most similar *training*
patients by cosine similarity of their atoms. For those neighbours ``HighRisk(q)``
is observed and set to the censoring-aware soft target of
:func:`.atoms.soft_targets`, so risk propagates through the patient graph
without any test outcome. The rule weights are learned end to end. Inference is
unrolled as projected gradient steps, and the weights are fitted by back-propagating
a soft-label cross-entropy plus a Cox partial-likelihood term on the training patients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from ...survival import cox_partial_likelihood
from ..kg_evidence import logrank_table
from .atoms import AtomTable, soft_targets

N_FACTS = 40
NEIGHBOURS = 10
INFERENCE_STEPS = 40
STEP = 0.2
EPOCHS = 150
L1_PENALTY = 1e-2
COX_WEIGHT = 0.5


def soft_truth(atoms: AtomTable) -> np.ndarray:
    return np.clip(atoms.values / 2.0, 0.0, 1.0).astype(np.float32)


class SoftLogicProgram(nn.Module):
    def __init__(self, n_facts: int) -> None:
        super().__init__()
        self.risk = nn.Parameter(torch.full((n_facts,), -3.0))
        self.protect = nn.Parameter(torch.full((n_facts,), -3.0))
        self.collective = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.prior = nn.Parameter(torch.tensor(0.0))

    def weights(self):
        softplus = nn.functional.softplus
        return softplus(self.risk), softplus(self.protect), softplus(self.collective), softplus(self.prior)

    def infer(self, facts: torch.Tensor, similarity: torch.Tensor, neighbour_truth: torch.Tensor) -> torch.Tensor:
        """MAP state of HighRisk by projected gradient on the convex hinge-loss energy."""
        w_risk, w_protect, w_collective, w_prior = self.weights()
        risk = torch.full((facts.size(0),), 0.2)
        for _ in range(INFERENCE_STEPS):
            gradient = (
                -2 * (w_risk * torch.relu(facts - risk[:, None])).sum(1)
                + 2 * (w_protect * torch.relu(facts + risk[:, None] - 1)).sum(1)
                - 2 * w_collective[0] * (similarity * torch.relu(neighbour_truth - risk[:, None])).sum(1)
                + 2 * w_collective[1] * (similarity * torch.relu(risk[:, None] - neighbour_truth)).sum(1)
                + 2 * w_prior * risk
            )
            scale = 1.0 + w_risk.sum() + w_protect.sum() + w_collective.sum() + w_prior
            risk = torch.clamp(risk - STEP * gradient / scale, 0.0, 1.0)
        return risk


def _neighbours(truth: np.ndarray, rows: np.ndarray, pool: np.ndarray, exclude_self: bool):
    unit = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), 1e-9)
    similarity = unit[rows] @ unit[pool].T
    if exclude_self:
        similarity[np.arange(len(rows)), np.arange(len(rows))] = -np.inf
    order = np.argsort(-similarity, axis=1)[:, :NEIGHBOURS]
    values = np.take_along_axis(similarity, order, axis=1)
    return order, np.clip(values, 0, 1).astype(np.float32)


def fit_fold(atoms: AtomTable, time: np.ndarray, event: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int = 0):
    torch.manual_seed(seed)
    train = train[np.isfinite(time[train]) & (time[train] > 0)]
    targets = soft_targets(time, event, train)
    z = np.nan_to_num(logrank_table(time[train], event[train], atoms.binary[train].astype(float))["z"].to_numpy())
    chosen = np.argsort(-np.abs(z))[:N_FACTS]
    truth = soft_truth(atoms)[:, chosen]
    similarity_basis = soft_truth(atoms)
    train_neighbours, train_similarity = _neighbours(similarity_basis, train, train, exclude_self=True)
    test_neighbours, test_similarity = _neighbours(similarity_basis, test, train, exclude_self=False)
    target_train = torch.tensor(targets[train])
    facts_train = torch.tensor(truth[train])
    facts_test = torch.tensor(truth[test])
    neighbour_train = target_train[torch.tensor(train_neighbours)]
    neighbour_test = target_train[torch.tensor(test_neighbours)]
    sim_train = torch.tensor(train_similarity)
    sim_test = torch.tensor(test_similarity)
    time_train = torch.tensor(time[train], dtype=torch.float32)
    event_train = torch.tensor(event[train], dtype=torch.float32)
    program = SoftLogicProgram(len(chosen))
    optimiser = torch.optim.Adam(program.parameters(), lr=0.05)
    for _ in range(EPOCHS):
        optimiser.zero_grad()
        risk = program.infer(facts_train, sim_train, neighbour_train)
        clipped = risk.clamp(1e-4, 1 - 1e-4)
        soft_label = nn.functional.binary_cross_entropy(clipped, target_train)
        logit = torch.log(clipped) - torch.log1p(-clipped)
        cox = cox_partial_likelihood(time_train, event_train, logit)
        w_risk, w_protect, _, _ = program.weights()
        loss = soft_label + COX_WEIGHT * cox + L1_PENALTY * (w_risk.sum() + w_protect.sum())
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        risk_test = program.infer(facts_test, sim_test, neighbour_test).numpy()
        w_risk, w_protect, w_collective, w_prior = (value.numpy() for value in program.weights())
    rows = []
    for position, atom in enumerate(chosen):
        net = float(w_risk[position] - w_protect[position])
        rows.append(
            {
                "rule": f"{atoms.formulas[atom]} ⇒ {'HighRisk' if net > 0 else '¬HighRisk'}(p)",
                "name": atoms.names[atom],
                "risk_weight": float(w_risk[position]),
                "protect_weight": float(w_protect[position]),
                "net_weight": net,
                "uses_kg": atoms.kinds[atom].startswith("kg_"),
                "support": int(atoms.binary[train][:, atom].sum()),
            }
        )
    weights = pd.DataFrame(rows).sort_values("net_weight", key=np.abs, ascending=False)
    info = {"collective_plus": float(w_collective[0]), "collective_minus": float(w_collective[1]), "prior": float(w_prior), "facts": int(len(chosen))}
    return risk_test, weights, info
