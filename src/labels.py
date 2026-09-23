"""Binary recurrence labels from clinical outcomes.

Labels are read only when a supervised target is required. They are not graph
features. ``Y = 0`` means the patient was censored at last follow-up, not that
recurrence was ruled out for the rest of their life.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .io_utils import normalize_identifier, read_cbioportal_table

# Columns that would leak the learning target if copied onto patient graphs.
OUTCOME_FIELDS = (
    "OS_STATUS",
    "OS_MONTHS",
    "DSS_STATUS",
    "DSS_MONTHS",
    "DFS_STATUS",
    "DFS_MONTHS",
    "PFS_STATUS",
    "PFS_MONTHS",
    "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
)

# Clinical attributes the fused graph is allowed to encode. Outcome fields are absent.
CLINICAL_FEATURE_FIELDS = (
    "AGE",
    "SEX",
    "SUBTYPE",
    "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "PATH_M_STAGE",
)

PFS_POSITIVE = "1:PROGRESSION"
PFS_NEGATIVE = "0:CENSORED"
LANDMARK_MONTHS = 36
ENDPOINTS = ("pfs", "new_tumor_event", "pfs_36m")


def parse_pfs_status(value: object) -> int | None:
    """Map PFS_STATUS onto {0, 1}. Any other token, including blanks, is missing."""
    text = normalize_identifier(value)
    if text is None:
        return None
    upper = text.upper()
    if upper in {PFS_POSITIVE, "PROGRESSION"}:
        return 1
    if upper in {PFS_NEGATIVE, "CENSORED"}:
        return 0
    return None


def parse_new_tumor_event(value: object) -> int | None:
    """Map NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT onto {0, 1} without imputing blanks."""
    text = normalize_identifier(value)
    if text is None:
        return None
    upper = text.upper()
    if upper in {"YES", "Y"}:
        return 1
    if upper in {"NO", "N"}:
        return 0
    return None


def parse_months(value: object) -> float | None:
    """Parse a follow-up time. Non-numeric tokens are missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        months = float(value)
    except (TypeError, ValueError):
        text = normalize_identifier(value)
        if text is None:
            return None
        try:
            months = float(text)
        except ValueError:
            return None
    if not math.isfinite(months):
        return None
    return months


def landmark_pfs_label(status: object, months: object, horizon: float = LANDMARK_MONTHS) -> int | None:
    """Progression by ``horizon`` months.

    ``PFS_MONTHS`` is used only to decide whether the event window was observed.
    It is not returned as a feature. Censoring before the horizon is missing,
    not a negative. Progression after the horizon is event-free at the horizon.
    """
    parsed = parse_pfs_status(status)
    observed = parse_months(months)
    if parsed is None or observed is None:
        return None
    if parsed == 1 and observed <= horizon:
        return 1
    if (parsed == 1 and observed > horizon) or (parsed == 0 and observed >= horizon):
        return 0
    return None


def _parser_for(endpoint: str):
    if endpoint == "pfs":
        return "PFS_STATUS", parse_pfs_status
    if endpoint == "new_tumor_event":
        return "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT", parse_new_tumor_event
    if endpoint == "pfs_36m":
        return "PFS_STATUS", parse_pfs_status
    raise ValueError(f"Unknown endpoint {endpoint!r}; expected one of {ENDPOINTS}")


def labels_from_patients(patients: pd.DataFrame, endpoint: str = "pfs") -> pd.DataFrame:
    """Return one row per patient with a usable binary label. Incomplete rows are dropped."""
    column, parser = _parser_for(endpoint)
    if column not in patients.columns:
        raise KeyError(f"{column} is not in the clinical patient table")
    if endpoint == "pfs_36m" and "PFS_MONTHS" not in patients.columns:
        raise KeyError("PFS_MONTHS is required to define the 36-month label and is not a model feature")
    rows: list[dict[str, object]] = []
    for record in patients.itertuples(index=False):
        patient_id = normalize_identifier(getattr(record, "PATIENT_ID"))
        if patient_id is None:
            continue
        raw = getattr(record, column)
        if endpoint == "pfs_36m":
            label = landmark_pfs_label(raw, getattr(record, "PFS_MONTHS", None), LANDMARK_MONTHS)
            source = f"{raw}|{getattr(record, 'PFS_MONTHS', None)}"
        else:
            label = parser(raw)
            source = str(raw)
        if label is None:
            continue
        rows.append({"PATIENT_ID": patient_id, "y": int(label), "endpoint": endpoint, "source_value": source})
    frame = pd.DataFrame(rows, columns=["PATIENT_ID", "y", "endpoint", "source_value"])
    if frame["PATIENT_ID"].duplicated().any():
        raise ValueError("Recurrence labels must contain one row per patient")
    return frame


def load_recurrence_labels(dataset_dir: Path, endpoint: str = "pfs") -> pd.DataFrame:
    """Load binary labels from ``data_clinical_patient.txt`` for one endpoint."""
    patients = read_cbioportal_table(dataset_dir / "data_clinical_patient.txt")
    return labels_from_patients(patients, endpoint)


def assert_no_outcome_leakage(feature_names: list[str] | tuple[str, ...]) -> None:
    """Raise if a feature name is an outcome column or embeds an endpoint token."""
    names = [str(name) for name in feature_names]
    overlap = sorted(set(names).intersection(OUTCOME_FIELDS))
    if overlap:
        raise ValueError(f"Outcome columns leaked into features: {overlap}")
    banned_tokens = ("PFS", "DFS", "DSS", "OS_STATUS", "OS_MONTHS", "NEW_TUMOR")
    leaked = [name for name in names if any(token in name.upper() for token in banned_tokens)]
    if leaked:
        raise ValueError(f"Feature names contain outcome tokens: {leaked}")
