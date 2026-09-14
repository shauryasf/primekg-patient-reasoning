"""Clinical data loading, normalization, and patient/sample linkage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io_utils import normalize_identifier, read_cbioportal_table


@dataclass
class ClinicalData:
    patients: pd.DataFrame
    samples: pd.DataFrame
    malformed_rows: list[dict[str, str]]


def _clean_ids(frame: pd.DataFrame, columns: list[str], source: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frame = frame.copy()
    malformed: list[dict[str, str]] = []
    for column in columns:
        frame[column] = frame[column].map(normalize_identifier)
    invalid = frame[columns].isna().any(axis=1)
    for index in frame.index[invalid]:
        malformed.append({"source": source, "row": str(index), "reason": "missing required identifier"})
    return frame.loc[~invalid].copy(), malformed


def load_clinical_data(dataset_dir: Path) -> ClinicalData:
    """Load the actual patient- and sample-level clinical tables and validate keys."""
    patients = read_cbioportal_table(dataset_dir / "data_clinical_patient.txt")
    samples = read_cbioportal_table(dataset_dir / "data_clinical_sample.txt")
    patients, malformed_patient = _clean_ids(patients, ["PATIENT_ID"], "data_clinical_patient.txt")
    samples, malformed_sample = _clean_ids(samples, ["PATIENT_ID", "SAMPLE_ID"], "data_clinical_sample.txt")
    duplicate_patients = patients.duplicated("PATIENT_ID", keep=False)
    duplicate_samples = samples.duplicated("SAMPLE_ID", keep=False)
    if duplicate_patients.any() or duplicate_samples.any():
        raise ValueError("Clinical identifiers must be unique; duplicate records are not safely collapsible.")
    return ClinicalData(patients=patients, samples=samples, malformed_rows=malformed_patient + malformed_sample)


def link_patients_to_samples(clinical: ClinicalData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return linked samples and sample records lacking a patient mapping."""
    patient_columns = ["PATIENT_ID", "SUBTYPE", "CANCER_TYPE_ACRONYM", "AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_M_STAGE"]
    linked = clinical.samples.merge(clinical.patients[patient_columns], on="PATIENT_ID", how="left", validate="many_to_one")
    unmapped = linked[linked["CANCER_TYPE_ACRONYM"].isna()].copy()
    return linked.loc[~linked.index.isin(unmapped.index)].copy(), unmapped
