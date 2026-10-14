"""One cohort, one set of folds, shared by every stage of the neuro-symbolic run.

The cohort is every patient with a usable ``PFS_STATUS`` (1,083 in the datahub
release), so the survival models see the full follow-up instead of the 36-month
landmark subset. The neural, symbolic, and fused models all use the same
stratified folds, which makes their out-of-fold predictions directly comparable
and lets the fusion stage stack them patient by patient.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from ..dataset_fusion import (
    FusedCohort,
    FusionConfig,
    PrimeSlice,
    build_cohort,
    load_fusion_config,
    load_gene_panel,
    load_primekg_slice,
)
from ..downloads import brca_dir, primekg_csv_path
from ..labels import load_recurrence_labels
from ..survival import align_survival_targets, load_survival_targets

ENDPOINT = "pfs"


@dataclass
class StudyCohort:
    cohort: FusedCohort
    time: np.ndarray
    event: np.ndarray
    usable: np.ndarray
    fold_id: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    prime_frame: pd.DataFrame
    prime: PrimeSlice
    panel: dict[str, dict[str, object]]
    fusion: FusionConfig
    dataset_dir: Path

    @property
    def patient_ids(self) -> list[str]:
        return self.cohort.patient_ids


def load_study(n_folds: int = 5, seed: int = 42, dataset_dir: Path | None = None) -> StudyCohort:
    """Build the fused cohort and attach (time, event) targets and fold ids.

    The folds are ``StratifiedKFold(n_folds, shuffle=True, random_state=seed)`` on
    the event indicator, which is the split ``train_and_evaluate_gnns.cross_validate``
    draws for the same seed, so a GNN fold state lines up with ``fold_id``.
    """
    dataset_dir = dataset_dir or brca_dir()
    panel = load_gene_panel()
    panel_entrez = {str(info["entrez"]) for info in panel.values() if info.get("entrez")}
    prime_frame = load_primekg_slice(primekg_csv_path(), panel_entrez)
    fusion = load_fusion_config()
    labels = load_recurrence_labels(dataset_dir, ENDPOINT)
    cohort = build_cohort(dataset_dir, labels, prime_frame, panel, fusion=fusion)
    targets = load_survival_targets(dataset_dir, ENDPOINT)
    time, event, usable = align_survival_targets(targets, cohort.patient_ids)
    # Targets ride on each graph so a mini-batch carries its own (time, event); they
    # are never node features.
    for graph, row_time, row_event in zip(cohort.graphs, time, event):
        graph.time = torch.tensor([float(row_time)], dtype=torch.float)
        graph.event = torch.tensor([float(row_event)], dtype=torch.float)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [(np.array(train), np.array(test)) for train, test in splitter.split(np.zeros(len(cohort.y)), cohort.y)]
    fold_id = np.full(len(cohort.y), -1, dtype=int)
    for fold, (_, test) in enumerate(folds):
        fold_id[test] = fold
    return StudyCohort(
        cohort=cohort,
        time=time,
        event=event,
        usable=usable,
        fold_id=fold_id,
        folds=folds,
        prime_frame=prime_frame,
        prime=PrimeSlice.from_frame(prime_frame),
        panel=panel,
        fusion=fusion,
        dataset_dir=dataset_dir,
    )
