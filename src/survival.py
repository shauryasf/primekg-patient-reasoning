"""Concordance targets and objectives for the recurrence benchmark.

The binary label in :mod:`src.labels` answers "did progression happen?". The
concordance index (C-index) asks a different question: of two comparable
patients, does the model score the one with the earlier event higher? Answering
it needs a follow-up time and an event indicator. Both are outcome columns, so
they stay out of the graph: :data:`src.labels.OUTCOME_FIELDS` lists them and
``assert_no_outcome_leakage`` rejects them in the feature vector. This module
reads them as *targets* only, in the same spirit as ``landmark_pfs_label``,
which already reads ``PFS_MONTHS`` to place the 36-month window.

Pair definition (Harrell): patient ``i`` is comparable to patient ``j`` when

- ``event_i == 1`` and ``time_i < time_j``, or
- both patients have an event at the same time.

A censored patient can therefore only be the later member of a pair: their
event, if any, was not observed. Tied risks score 0.5. Pairs where the earlier
patient is censored are dropped, which is what stops ``Y = 0`` from meaning
"never recurs".

Two objectives optimise this ranking:

- ``cindex``: a differentiable surrogate of the C-index itself, ``1 -`` mean
  ``sigmoid((risk_i - risk_j) / temperature)`` over comparable pairs.
- ``cox``: the Breslow partial likelihood. Maximising it is the classic
  likelihood formulation whose maximiser ranks patients by hazard, so it moves
  the C-index too.

``bce`` remains available and is the default, so earlier runs stay reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .io_utils import normalize_identifier, read_cbioportal_table
from .labels import LANDMARK_MONTHS, landmark_pfs_label, parse_months, parse_pfs_status

# Endpoints whose label carries an event time. NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT
# has no time axis, so its sensitivity run stays on the binary objective.
SURVIVAL_ENDPOINTS = ("pfs", "pfs_36m")
LOSS_NAMES = ("bce", "cindex", "cox")
DEFAULT_TEMPERATURE = 1.0


@dataclass
class SurvivalTargets:
    """One row per patient with an observed event and follow-up time.

    ``time`` is the follow-up in months and may be ``nan`` when the clinical
    table has no usable value; those rows are dropped from comparable pairs.
    ``event`` is 1 for ``1:PROGRESSION`` and 0 for ``0:CENSORED``.
    """

    patient_ids: list[str]
    time: np.ndarray
    event: np.ndarray
    endpoint: str
    horizon: float | None = None
    note: str = ""

    def __len__(self) -> int:
        return len(self.patient_ids)

    def audit(self) -> dict[str, object]:
        timed = int(np.isfinite(self.time).sum())
        return {
            "endpoint": self.endpoint,
            "patients": int(len(self.patient_ids)),
            "events": int(self.event.sum()),
            "censored": int((self.event == 0).sum()),
            "with_follow_up_time": timed,
            "without_follow_up_time": int(len(self.patient_ids) - timed),
            "horizon_months": None if self.horizon is None else float(self.horizon),
            "note": self.note,
        }


def survival_targets_from_patients(
    patients: pd.DataFrame,
    endpoint: str,
    horizon: float = LANDMARK_MONTHS,
) -> SurvivalTargets:
    """Build (time, event) targets from ``data_clinical_patient.txt``.

    ``pfs`` uses the observed follow-up directly. ``pfs_36m`` administratively
    censors every patient at the horizon, which is exactly the window
    ``landmark_pfs_label`` already applies: progression at or before 36 months
    is an event at its own month, anything later is event-free at month 36.
    Patients censored before the horizon stay dropped, as in the binary label.
    """
    if endpoint not in SURVIVAL_ENDPOINTS:
        raise ValueError(
            f"Endpoint {endpoint!r} has no event time; expected one of {SURVIVAL_ENDPOINTS}. "
            "The new-tumor-event sensitivity endpoint keeps the binary objective."
        )
    if "PFS_STATUS" not in patients.columns:
        raise KeyError("PFS_STATUS is required to define the event indicator")
    if endpoint == "pfs_36m" and "PFS_MONTHS" not in patients.columns:
        raise KeyError("PFS_MONTHS is required to define the 36-month window and is not a model feature")
    if endpoint == "pfs_36m" and horizon <= 0:
        raise ValueError(f"horizon must be positive, received {horizon}")

    patient_ids: list[str] = []
    times: list[float] = []
    events: list[int] = []
    for record in patients.itertuples(index=False):
        patient_id = normalize_identifier(getattr(record, "PATIENT_ID"))
        if patient_id is None:
            continue
        months = parse_months(getattr(record, "PFS_MONTHS", None))
        if endpoint == "pfs":
            event = parse_pfs_status(getattr(record, "PFS_STATUS"))
            observed = months
        else:
            event = landmark_pfs_label(getattr(record, "PFS_STATUS"), months, horizon)
            observed = None if months is None else min(float(months), float(horizon))
        if event is None:
            continue
        patient_ids.append(patient_id)
        times.append(float("nan") if observed is None else float(observed))
        events.append(int(event))
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Survival targets must contain one row per patient")
    if endpoint == "pfs":
        note = (
            "PFS_MONTHS is the follow-up axis for the concordance objective only. "
            "It is not a feature, and a censored patient is never an earlier event."
        )
    else:
        note = (
            f"Follow-up is administratively censored at {horizon:g} months, matching the pfs_36m label. "
            "PFS_MONTHS is a target, never a feature."
        )
    return SurvivalTargets(
        patient_ids=patient_ids,
        time=np.array(times, dtype=np.float64),
        event=np.array(events, dtype=np.int64),
        endpoint=endpoint,
        horizon=None if endpoint == "pfs" else float(horizon),
        note=note,
    )


def load_survival_targets(dataset_dir: Path, endpoint: str, horizon: float = LANDMARK_MONTHS) -> SurvivalTargets:
    """Load concordance targets for one endpoint from the clinical patient table."""
    patients = read_cbioportal_table(dataset_dir / "data_clinical_patient.txt")
    return survival_targets_from_patients(patients, endpoint, horizon)


def align_survival_targets(targets: SurvivalTargets, patient_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder targets onto a cohort, returning time, event, and a usable mask.

    A patient that is missing from the target table or has no follow-up time is
    unusable for pairs and is reported rather than silently treated as event-free.
    """
    lookup = {patient_id: index for index, patient_id in enumerate(targets.patient_ids)}
    time = np.full(len(patient_ids), np.nan, dtype=np.float64)
    event = np.zeros(len(patient_ids), dtype=np.int64)
    usable = np.zeros(len(patient_ids), dtype=bool)
    for position, patient_id in enumerate(patient_ids):
        index = lookup.get(str(patient_id))
        if index is None:
            continue
        event[position] = int(targets.event[index])
        time[position] = float(targets.time[index])
        usable[position] = bool(math.isfinite(time[position]))
    return time, event, usable


def pair_indices(time: np.ndarray | torch.Tensor, event: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Indices of comparable pairs, each ordered ``(earlier event, later patient)``."""
    time_values = np.asarray(time, dtype=np.float64).reshape(-1)
    event_values = np.asarray(event, dtype=np.float64).reshape(-1)
    if time_values.shape != event_values.shape:
        raise ValueError("time and event must have the same length")
    count = time_values.size
    if count == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    timed = np.isfinite(time_values)
    observed = event_values == 1
    strict = timed[:, None] & timed[None, :] & observed[:, None] & (time_values[:, None] < time_values[None, :])
    tied = (
        timed[:, None]
        & timed[None, :]
        & observed[:, None]
        & observed[None, :]
        & (time_values[:, None] == time_values[None, :])
        & (np.arange(count)[:, None] < np.arange(count)[None, :])
    )
    earlier, later = np.nonzero(strict | tied)
    return earlier.astype(np.int64), later.astype(np.int64)


def concordance_audit(time: np.ndarray, event: np.ndarray) -> dict[str, object]:
    """Counts that make a C-index readable: who is timed, and how many pairs exist."""
    time_values = np.asarray(time, dtype=np.float64).reshape(-1)
    event_values = np.asarray(event, dtype=np.float64).reshape(-1)
    earlier, later = pair_indices(time_values, event_values)
    timed = int(np.isfinite(time_values).sum())
    events = int((event_values == 1).sum())
    return {
        "patients": int(time_values.size),
        "with_follow_up_time": timed,
        "events": events,
        "censored": int((event_values == 0).sum()),
        "comparable_pairs": int(earlier.size),
        "comparable_pairs_per_event": float(earlier.size / max(events, 1)),
    }


def harrell_c_index(
    time: np.ndarray | torch.Tensor,
    event: np.ndarray | torch.Tensor,
    risk: np.ndarray | torch.Tensor,
) -> float:
    """Harrell's C-index of ``risk`` (higher risk means earlier event).

    Returns ``nan`` when no pair is comparable, which is the honest answer for a
    fold with one event and no later patient, instead of an invented 0.5.
    """
    risk_values = risk.detach().cpu().numpy() if isinstance(risk, torch.Tensor) else np.asarray(risk, dtype=np.float64)
    risk_values = np.asarray(risk_values, dtype=np.float64).reshape(-1)
    time_values = np.asarray(time, dtype=np.float64).reshape(-1)
    if risk_values.size != time_values.size:
        raise ValueError(f"risk has {risk_values.size} rows but time has {time_values.size}")
    earlier, later = pair_indices(time_values, event)
    if earlier.size == 0:
        return float("nan")
    earlier_risk = risk_values[earlier]
    later_risk = risk_values[later]
    concordant = float((earlier_risk > later_risk).sum()) + 0.5 * float((earlier_risk == later_risk).sum())
    return concordant / float(earlier.size)


def soft_concordance(
    time: torch.Tensor,
    event: torch.Tensor,
    risk: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor | None:
    """Differentiable per-pair concordance surrogates, or ``None`` when no pair exists.

    ``sigmoid((risk_i - risk_j) / temperature)`` is 1 when the earlier event is
    scored higher, 0.5 at a tie, and 0 when the order is reversed, so its mean is
    a soft C-index and ``1 - mean`` is a smooth ranking loss.
    """
    earlier, later = pair_indices(time.detach().cpu().numpy(), event.detach().cpu().numpy())
    if earlier.size == 0:
        return None
    scale = max(float(temperature), 1e-3)
    index_earlier = torch.as_tensor(earlier, dtype=torch.long, device=risk.device)
    index_later = torch.as_tensor(later, dtype=torch.long, device=risk.device)
    return torch.sigmoid((risk[index_earlier] - risk[index_later]) / scale)


def cox_partial_likelihood(time: torch.Tensor, event: torch.Tensor, risk: torch.Tensor) -> torch.Tensor | None:
    """Negative Breslow partial log-likelihood over comparable risk sets, or ``None``.

    Ties in time are handled by putting every patient with ``time_j >= time_i`` in
    the risk set of patient ``i``, which is the standard Breslow convention.
    """
    time_values = time.detach().reshape(-1).to(dtype=torch.float64)
    event_values = event.detach().reshape(-1).to(dtype=torch.float64)
    risk_values = risk.reshape(-1)
    timed = torch.isfinite(time_values)
    observed = timed & (event_values == 1)
    if int(observed.sum()) == 0:
        return None
    at_risk = timed.unsqueeze(0) & (time_values.unsqueeze(0) >= time_values.unsqueeze(1))
    masked = risk_values.unsqueeze(0).expand(time_values.numel(), -1)
    masked = masked.masked_fill(~at_risk, float("-inf"))
    log_denominator = torch.logsumexp(masked, dim=1)
    terms = risk_values[observed] - log_denominator[observed]
    return -terms.mean().to(dtype=risk.dtype)


class BinaryObjective(nn.Module):
    """Weighted BCE on the binary label. Ignores time and event."""

    name = "bce"
    needs_survival = False

    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float))

    def forward(
        self,
        risk: torch.Tensor,
        target: torch.Tensor,
        time: torch.Tensor | None = None,
        event: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss(risk, target)

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "pos_weight": self.pos_weight}


class ConcordanceObjective(nn.Module):
    """``1 -`` soft C-index over comparable pairs, with an optional BCE anchor.

    The surrogate is computed inside each mini-batch, so it is a stochastic but
    unbiased estimate of the fold-level soft C-index. A batch that happens to
    contain no comparable pair returns a zero-valued term that still carries the
    graph, so a step is never silently skipped without a gradient.
    """

    name = "cindex"
    needs_survival = True

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, aux_bce: float = 0.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        if aux_bce < 0:
            raise ValueError("aux_bce must not be negative")
        self.temperature = float(temperature)
        self.aux_bce = float(aux_bce)
        self.pos_weight = float(pos_weight)
        self.anchor = (
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float)) if aux_bce > 0 else None
        )
        self.last_soft_c_index: float | None = None

    def forward(
        self,
        risk: torch.Tensor,
        target: torch.Tensor,
        time: torch.Tensor | None = None,
        event: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if time is None or event is None:
            raise ValueError("the concordance objective needs the follow-up time and the event indicator as targets")
        loss = risk.reshape(-1).sum() * 0.0
        soft = soft_concordance(time, event, risk, self.temperature)
        if soft is not None:
            self.last_soft_c_index = float(soft.mean().detach())
            loss = loss + (1.0 - soft.mean())
        if self.anchor is not None:
            loss = loss + self.aux_bce * self.anchor(risk, target)
        return loss

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "temperature": self.temperature,
            "aux_bce_weight": self.aux_bce,
            "pos_weight": self.pos_weight,
            "pairs": "in-batch comparable pairs (Harrell)",
        }


class CoxObjective(nn.Module):
    """Negative Breslow partial likelihood, with the same optional BCE anchor."""

    name = "cox"
    needs_survival = True

    def __init__(self, aux_bce: float = 0.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        if aux_bce < 0:
            raise ValueError("aux_bce must not be negative")
        self.aux_bce = float(aux_bce)
        self.pos_weight = float(pos_weight)
        self.anchor = (
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float)) if aux_bce > 0 else None
        )

    def forward(
        self,
        risk: torch.Tensor,
        target: torch.Tensor,
        time: torch.Tensor | None = None,
        event: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if time is None or event is None:
            raise ValueError("the Cox objective needs the follow-up time and the event indicator as targets")
        loss = risk.reshape(-1).sum() * 0.0
        partial = cox_partial_likelihood(time, event, risk)
        if partial is not None:
            loss = loss + partial
        if self.anchor is not None:
            loss = loss + self.aux_bce * self.anchor(risk, target)
        return loss

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "aux_bce_weight": self.aux_bce,
            "pos_weight": self.pos_weight,
            "risk_set": "Breslow, time_j >= time_i",
        }


def positive_weight(y_train: np.ndarray) -> float:
    """``negatives / positives`` on the inner training split, the same rule as before."""
    labels = np.asarray(y_train)
    positives = max(int(labels.sum()), 1)
    negatives = max(int(len(labels) - int(labels.sum())), 1)
    return float(negatives / positives)


def build_objective(
    name: str,
    y_train: np.ndarray,
    aux_bce: float = 0.0,
    temperature: float = DEFAULT_TEMPERATURE,
) -> nn.Module:
    """Objective for one training fold. ``bce`` is the default and the safe fallback."""
    if name not in LOSS_NAMES:
        raise ValueError(f"Unknown loss {name!r}; expected one of {LOSS_NAMES}")
    weight = positive_weight(y_train)
    if name == "bce":
        return BinaryObjective(weight)
    if name == "cindex":
        return ConcordanceObjective(temperature=temperature, aux_bce=aux_bce, pos_weight=weight)
    return CoxObjective(aux_bce=aux_bce, pos_weight=weight)
