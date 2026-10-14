"""Population evidence for PrimeKG facts, used to score explanation paths symbolically.

A GNN explanation path ends at a PrimeKG node: a pathway, a drug, or a breast
disease. The fact behind it is "this patient has an altered gene directly
annotated to that node". This module measures, on a training fold only, how
patients who carry that fact progress compared with everyone else, using the
log-rank observed/expected ratio. The result is the symbolic half of a path's
neuro-symbolic score: the GNN says the path mattered for this patient, the
cohort says whether that fact tracks progression at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset_fusion import PrimeSlice


def fact_carriers(patient_ids: list[str], altered: dict[str, set[str]], prime: PrimeSlice) -> tuple[list[str], np.ndarray]:
    """Boolean ``[patients, facts]`` matrix for every pathway, drug, and disease node.

    A patient carries ``pathway:P`` when one of their altered genes is annotated to
    P in PrimeKG. Indirect links through a protein partner are not counted here.
    """
    targets: dict[str, set[str]] = {}
    for prefix, mapping in (("pathway", prime.gene_pathways), ("drug", prime.gene_drugs), ("disease", prime.gene_diseases)):
        for gene, nodes in mapping.items():
            for node in nodes:
                targets.setdefault(f"{prefix}:{node}", set()).add(gene)
    facts = sorted(targets)
    matrix = np.zeros((len(patient_ids), len(facts)), dtype=bool)
    for row, patient_id in enumerate(patient_ids):
        genes = altered.get(patient_id, set())
        if not genes:
            continue
        for column, fact in enumerate(facts):
            if genes & targets[fact]:
                matrix[row, column] = True
    return facts, matrix


def logrank_table(time: np.ndarray, event: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Vectorised two-group log-rank for many binary indicators at once.

    Returns, per indicator column, the carriers, their observed and expected
    events, the O/E hazard-ratio estimate ``(O1/E1)/(O2/E2)``, and the log-rank z.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    keep = np.isfinite(time)
    time, event, groups = time[keep], event[keep], groups[keep].astype(float)
    event_times = np.unique(time[event == 1])
    at_risk = time[None, :] >= event_times[:, None]
    died = (time[None, :] == event_times[:, None]) & (event[None, :] == 1)
    n_total = at_risk.sum(axis=1).astype(float)
    d_total = died.sum(axis=1).astype(float)
    n_group = at_risk.astype(float) @ groups
    d_group = died.astype(float) @ groups
    expected_share = n_group / n_total[:, None]
    observed = d_group.sum(axis=0)
    expected = (d_total[:, None] * expected_share).sum(axis=0)
    variance = (
        d_total[:, None]
        * expected_share
        * (1 - expected_share)
        * ((n_total - d_total) / np.maximum(n_total - 1, 1))[:, None]
    ).sum(axis=0)
    other_observed = event.sum() - observed
    other_expected = d_total.sum() - expected
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard_ratio = (observed / expected) / (other_observed / other_expected)
        z = (observed - expected) / np.sqrt(variance)
    return pd.DataFrame(
        {
            "carriers": groups.sum(axis=0).astype(int),
            "observed": observed,
            "expected": expected,
            "hazard_ratio": hazard_ratio,
            "z": z,
        }
    )


def fold_fact_evidence(
    facts: list[str],
    carriers: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    min_carriers: int = 15,
) -> pd.DataFrame:
    """Log-rank evidence for every fact on every training fold.

    A patient's paths are scored with the evidence from the fold that held them
    out, so a path's symbolic score never uses that patient's own outcome.
    """
    rows = []
    for fold, (train, _) in enumerate(folds):
        table = logrank_table(time[train], event[train], carriers[train])
        table.insert(0, "fact", facts)
        table.insert(0, "fold", fold)
        share = table["carriers"] / len(train)
        table = table[(table["carriers"] >= min_carriers) & (share <= 0.95)]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)
