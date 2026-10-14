"""Readable patient facts for the symbolic layer, the fused Cox model, and the reports.

Three families of facts are built here, one row per patient:

- extended clinical context: age, T/N/M stage, AJCC stage, PAM50 subtype,
  histology, radiation therapy, aneuploidy, and tumour mutational burden;
- gene alterations on the 500-gene panel (mutation, high-level amplification,
  homozygous deletion), with the protein change kept for the report;
- PrimeKG predicates: "an altered gene of this patient is a member of pathway P"
  or "is linked to breast disease D". These are the knowledge-graph facts the
  rule learner can put in a rule body.

Outcome columns (PFS/OS/DFS/DSS, new tumour event, and the person's cancer
status at last contact) are never read into a feature. They are loaded
separately by :func:`outcome_table` for the report's record section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..clinical import link_patients_to_samples, load_clinical_data
from ..cna import load_cna_for_samples
from ..config import load_feature_policy
from ..dataset_fusion import PrimeSlice, collapse_m_stage, collapse_stage, parse_age
from ..io_utils import normalize_identifier
from ..labels import OUTCOME_FIELDS, assert_no_outcome_leakage, parse_months, parse_pfs_status
from ..mutations import load_mutations

# Status at last contact is an outcome, so it is banned next to the survival columns.
EXTRA_OUTCOME_FIELDS = ("PERSON_NEOPLASM_CANCER_STATUS", "DAYS_LAST_FOLLOWUP")
NUMERIC_CLINICAL = ("age", "aneuploidy_score", "tmb_nonsynonymous")
NOMINAL_CLINICAL = ("t_stage", "n_stage", "m_stage", "ajcc_stage", "subtype", "histology", "radiation_therapy")
CLINICAL_LABELS = {
    "age": "Age",
    "aneuploidy_score": "Aneuploidy score",
    "tmb_nonsynonymous": "TMB (nonsynonymous)",
    "t_stage": "T stage",
    "n_stage": "N stage",
    "m_stage": "M stage",
    "ajcc_stage": "AJCC stage",
    "subtype": "PAM50 subtype",
    "histology": "Histology",
    "radiation_therapy": "Radiation therapy",
}
ALTERATION_LABELS = {"mut": "mutated", "amp": "amplified", "del": "deleted"}


def collapse_t_stage(value: object) -> str:
    text = normalize_identifier(value)
    match = re.match(r"^T([1-4])", (text or "").upper())
    return f"T{match.group(1)}" if match else "missing"


def collapse_n_stage(value: object) -> str:
    text = normalize_identifier(value)
    match = re.match(r"^N([0-3])", (text or "").upper())
    return f"N{match.group(1)}" if match else "missing"


def collapse_histology(value: object) -> str:
    text = (normalize_identifier(value) or "").lower()
    if "ductal" in text:
        return "ductal"
    if "lobular" in text:
        return "lobular"
    return "other" if text else "missing"


def _yes_no(value: object) -> str:
    text = (normalize_identifier(value) or "").lower()
    if text in {"yes", "y"}:
        return "yes"
    if text in {"no", "n"}:
        return "no"
    return "missing"


def _float(value: object) -> float:
    text = normalize_identifier(value)
    if text is None:
        return np.nan
    try:
        number = float(text)
    except ValueError:
        return np.nan
    return number if np.isfinite(number) else np.nan


def extended_clinical_table(dataset_dir: Path, patient_ids: list[str]) -> pd.DataFrame:
    """Baseline clinical context for each patient. Outcome columns are not read."""
    clinical = load_clinical_data(dataset_dir)
    patients = clinical.patients.set_index("PATIENT_ID")
    samples = clinical.samples.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
    rows = []
    for patient_id in patient_ids:
        row = patients.loc[patient_id]
        sample = samples.loc[patient_id] if patient_id in samples.index else pd.Series(dtype=object)
        m_stage = collapse_m_stage(row.get("PATH_M_STAGE"))
        rows.append(
            {
                "patient_id": patient_id,
                "age": parse_age(row.get("AGE")),
                "aneuploidy_score": _float(sample.get("ANEUPLOIDY_SCORE")),
                "tmb_nonsynonymous": _float(sample.get("TMB_NONSYNONYMOUS")),
                "t_stage": collapse_t_stage(row.get("PATH_T_STAGE")),
                "n_stage": collapse_n_stage(row.get("PATH_N_STAGE")),
                "m_stage": "M0" if m_stage == "CM0" else m_stage,
                "ajcc_stage": collapse_stage(row.get("AJCC_PATHOLOGIC_TUMOR_STAGE")),
                "subtype": (normalize_identifier(row.get("SUBTYPE")) or "missing").replace("BRCA_", ""),
                "histology": collapse_histology(sample.get("CANCER_TYPE_DETAILED")),
                "radiation_therapy": _yes_no(row.get("RADIATION_THERAPY")),
            }
        )
    frame = pd.DataFrame(rows).set_index("patient_id")
    assert_no_outcome_leakage(list(frame.columns))
    return frame


def record_table(dataset_dir: Path, patient_ids: list[str]) -> pd.DataFrame:
    """Everything the report prints in its clinical record, outcomes excluded."""
    clinical = load_clinical_data(dataset_dir)
    patients = clinical.patients.set_index("PATIENT_ID")
    samples = clinical.samples.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
    fields = {
        "AGE": "Age at diagnosis",
        "SEX": "Sex",
        "RACE": "Race",
        "ETHNICITY": "Ethnicity",
        "GENETIC_ANCESTRY_LABEL": "Genetic ancestry",
        "SUBTYPE": "PAM50 subtype",
        "AJCC_PATHOLOGIC_TUMOR_STAGE": "AJCC pathologic stage",
        "PATH_T_STAGE": "Pathologic T",
        "PATH_N_STAGE": "Pathologic N",
        "PATH_M_STAGE": "Pathologic M",
        "AJCC_STAGING_EDITION": "AJCC edition",
        "ICD_O_3_HISTOLOGY": "ICD-O-3 histology",
        "RADIATION_THERAPY": "Radiation therapy",
        "HISTORY_NEOADJUVANT_TRTYN": "Neoadjuvant therapy",
        "PRIOR_DX": "Prior diagnosis",
    }
    sample_fields = {
        "SAMPLE_ID": "Sample",
        "CANCER_TYPE_DETAILED": "Tumour type",
        "ONCOTREE_CODE": "OncoTree code",
        "ANEUPLOIDY_SCORE": "Aneuploidy score",
        "TMB_NONSYNONYMOUS": "TMB (nonsyn., /Mb)",
        "MSI_SENSOR_SCORE": "MSIsensor score",
    }
    rows = []
    for patient_id in patient_ids:
        row = patients.loc[patient_id]
        sample = samples.loc[patient_id] if patient_id in samples.index else pd.Series(dtype=object)
        record = {"patient_id": patient_id}
        for column, label in fields.items():
            record[label] = normalize_identifier(row.get(column)) or "—"
        for column, label in sample_fields.items():
            record[label] = normalize_identifier(sample.get(column)) or "—"
        rows.append(record)
    return pd.DataFrame(rows).set_index("patient_id")


def outcome_table(dataset_dir: Path, patient_ids: list[str]) -> pd.DataFrame:
    """Recorded outcomes. These are targets and report facts, never model inputs."""
    patients = load_clinical_data(dataset_dir).patients.set_index("PATIENT_ID")
    rows = []
    for patient_id in patient_ids:
        row = patients.loc[patient_id]
        rows.append(
            {
                "patient_id": patient_id,
                "pfs_event": parse_pfs_status(row.get("PFS_STATUS")),
                "pfs_months": parse_months(row.get("PFS_MONTHS")),
                "os_status": normalize_identifier(row.get("OS_STATUS")) or "—",
                "os_months": parse_months(row.get("OS_MONTHS")),
                "dfs_status": normalize_identifier(row.get("DFS_STATUS")) or "—",
            }
        )
    return pd.DataFrame(rows).set_index("patient_id")


def alteration_table(dataset_dir: Path, patient_ids: list[str], panel: dict[str, dict[str, object]]) -> pd.DataFrame:
    """One row per (patient, gene, alteration) on the 500-gene panel.

    Mutations keep the protein change and variant class so the report can print
    them. CNA keeps the same -2 / +2 policy the fused graph uses.
    """
    clinical = load_clinical_data(dataset_dir)
    linked, _ = link_patients_to_samples(clinical)
    linked = linked[linked["PATIENT_ID"].isin(set(patient_ids))]
    sample_to_patient = dict(zip(linked["SAMPLE_ID"].astype(str), linked["PATIENT_ID"].astype(str)))
    sample_ids = list(sample_to_patient)
    mutations, _ = load_mutations(dataset_dir, gene_panel=set(panel))
    mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(sample_ids)]
    policy = load_feature_policy().get("cna_policy", {})
    cna, _ = load_cna_for_samples(dataset_dir, sample_ids, gene_panel=set(panel), allowed_values=policy.get("include_values", [-2, 2]))
    rows = []
    for record in mutations.itertuples(index=False):
        symbol = str(record.Hugo_Symbol).upper()
        rows.append(
            {
                "patient_id": sample_to_patient[str(record.Tumor_Sample_Barcode)],
                "sample_id": str(record.Tumor_Sample_Barcode),
                "gene": symbol,
                "entrez": str(panel[symbol]["entrez"]),
                "kind": "mut",
                "detail": normalize_identifier(record.HGVSp_Short) or normalize_identifier(record.Variant_Classification) or "",
                "variant_class": normalize_identifier(record.Variant_Classification) or "",
                "driver": bool(panel[symbol].get("is_known_driver")),
            }
        )
    for record in cna.itertuples(index=False):
        symbol = str(record.Hugo_Symbol).upper()
        if symbol not in panel:
            continue
        rows.append(
            {
                "patient_id": sample_to_patient[str(record.SAMPLE_ID)],
                "sample_id": str(record.SAMPLE_ID),
                "gene": symbol,
                "entrez": str(panel[symbol]["entrez"]),
                "kind": "amp" if int(record.cna_value) == 2 else "del",
                "detail": "high-level amplification" if int(record.cna_value) == 2 else "homozygous deletion",
                "variant_class": "CNA",
                "driver": bool(panel[symbol].get("is_known_driver")),
            }
        )
    return pd.DataFrame(rows, columns=["patient_id", "sample_id", "gene", "entrez", "kind", "detail", "variant_class", "driver"])


@dataclass
class PredicateTable:
    """Readable predicates per patient, plus the label each column prints as."""

    frame: pd.DataFrame
    labels: dict[str, str]
    kinds: dict[str, str]
    members: dict[str, set[str]]


def _group_redundant(binary: pd.DataFrame, priority: list[str], threshold: float = 0.8) -> dict[str, list[str]]:
    """Greedy Jaccard grouping: each kept column absorbs columns that carry nearly the same patients.

    Co-amplified neighbours (for example the 8q24 or 1q amplicons) and pathways
    that differ by one gene would otherwise enter the rule learner as dozens of
    copies of one fact. ``priority`` decides which member names the group.
    """
    matrix = binary[priority].to_numpy(dtype=bool)
    sizes = matrix.sum(axis=0)
    representatives: list[int] = []
    groups: dict[str, list[str]] = {}
    for index, column in enumerate(priority):
        placed = False
        for rep in representatives:
            overlap = np.logical_and(matrix[:, index], matrix[:, rep]).sum()
            union = sizes[index] + sizes[rep] - overlap
            if union and overlap / union >= threshold:
                groups[priority[rep]].append(column)
                placed = True
                break
        if not placed:
            representatives.append(index)
            groups[column] = [column]
    return groups


def predicate_table(
    clinical: pd.DataFrame,
    alterations: pd.DataFrame,
    prime: PrimeSlice,
    min_gene_prevalence: float = 0.03,
    min_pathway_prevalence: float = 0.05,
    max_pathway_prevalence: float = 0.6,
    max_pathways: int = 120,
) -> PredicateTable:
    """Clinical, gene, and PrimeKG predicates for the rule learner.

    Column selection uses prevalence only, never the outcome, so it can be
    computed on the full cohort without leaking the target into a fold.
    """
    patient_ids = list(clinical.index)
    n = len(patient_ids)
    frame = clinical.copy()
    labels = {column: CLINICAL_LABELS[column] for column in clinical.columns}
    kinds = {column: "clinical" for column in clinical.columns}
    members: dict[str, set[str]] = {}

    counts = alterations.groupby("patient_id").agg(
        n_altered=("gene", "nunique"),
    )
    frame["n_altered_genes"] = counts["n_altered"].reindex(patient_ids).fillna(0).astype(float)
    labels["n_altered_genes"] = "Altered panel genes"
    kinds["n_altered_genes"] = "genomic"

    gene_columns: dict[str, pd.Series] = {}
    drivers = {gene for gene, flags in alterations.groupby("gene")["driver"] if bool(flags.iloc[0])}
    for (gene, kind), group in alterations.groupby(["gene", "kind"]):
        carriers = set(group["patient_id"])
        if len(carriers) < max(5, min_gene_prevalence * n):
            continue
        # Mutations outside known drivers are mostly long passenger genes (TTN, RYR2,
        # FLG, CSMD3); they only add noise rules such as "RYR2 not mutated".
        if kind == "mut" and gene not in drivers:
            continue
        column = f"{kind}_{gene}"
        gene_columns[column] = pd.Series([pid in carriers for pid in patient_ids], index=patient_ids)
        labels[column] = f"{gene} {ALTERATION_LABELS[kind]}"
        kinds[column] = "genomic"
        members[column] = {gene}
    altered_by_patient = alterations.groupby("patient_id")["entrez"].agg(lambda values: set(values)).to_dict()

    pathway_columns: dict[str, pd.Series] = {}
    for pathway_id, genes in prime.pathway_genes.items():
        hit = [len(altered_by_patient.get(pid, set()) & genes) for pid in patient_ids]
        prevalence = float(np.mean(np.array(hit) > 0))
        if not (min_pathway_prevalence <= prevalence <= max_pathway_prevalence):
            continue
        column = f"kg_pathway_{pathway_id}"
        pathway_columns[column] = pd.Series(np.array(hit) > 0, index=patient_ids)
        labels[column] = f"altered gene in pathway '{prime.pathway_names.get(pathway_id, pathway_id)}'"
        kinds[column] = "kg_pathway"
        members[column] = {f"pathway:{pathway_id}"}
    disease_genes: dict[str, set[str]] = {}
    for gene, diseases in prime.gene_diseases.items():
        for disease_id in diseases:
            disease_genes.setdefault(disease_id, set()).add(gene)
    for disease_id, genes in disease_genes.items():
        hit = np.array([len(altered_by_patient.get(pid, set()) & genes) > 0 for pid in patient_ids])
        if hit.mean() < min_pathway_prevalence:
            continue
        column = f"kg_disease_{disease_id}"
        pathway_columns[column] = pd.Series(hit, index=patient_ids)
        labels[column] = f"altered gene linked to '{prime.disease_names.get(disease_id, disease_id)}'"
        kinds[column] = "kg_disease"
        members[column] = {f"disease:{disease_id}"}

    binary = pd.DataFrame({**gene_columns, **pathway_columns}, index=patient_ids)
    prevalence = binary.mean()

    def gene_priority(column: str) -> tuple:
        # Name an amplicon after a known driver, then a protein-coding-looking symbol.
        gene = next(iter(members[column]))
        noncoding = bool(re.match(r"^(LINC|MIR|RNU|RP\d|SNOR|LOC)|-(AS|IT)\d", gene))
        return (gene not in drivers, noncoding, -prevalence[column], column)

    gene_groups = _group_redundant(binary, sorted(gene_columns, key=gene_priority))
    kg_groups = _group_redundant(binary, sorted(pathway_columns, key=lambda column: (-prevalence[column], column)))
    kg_kept = list(kg_groups)
    if len(kg_kept) > max_pathways:
        # Prefer predicates near 50% prevalence: they split the cohort the most.
        spread = prevalence[kg_kept].sub(0.5).abs().sort_values()
        kg_kept = list(spread.index[:max_pathways])
    for column in list(gene_groups) + kg_kept:
        frame[column] = np.where(binary[column].to_numpy(), "yes", "no")
        group = gene_groups.get(column) or kg_groups.get(column) or [column]
        if len(group) > 1 and kinds[column] == "genomic":
            labels[column] = f"{labels[column]} (+{len(group) - 1} co-altered genes)"
            members[column] = {gene for item in group for gene in members[item]}
        elif len(group) > 1:
            members[column] = {node for item in group for node in members[item]}
    labels = {column: label for column, label in labels.items() if column in frame.columns}
    kinds = {column: kind for column, kind in kinds.items() if column in frame.columns}
    members = {column: value for column, value in members.items() if column in frame.columns}
    # Gene symbols such as ADSS contain an outcome token as a substring, so the
    # token check runs on the clinical columns and the exact-name check on all.
    assert_no_outcome_leakage([column for column, kind in kinds.items() if kind == "clinical"])
    banned = set(OUTCOME_FIELDS) | set(EXTRA_OUTCOME_FIELDS)
    if banned.intersection(frame.columns):
        raise ValueError(f"Outcome columns reached the predicate table: {sorted(banned.intersection(frame.columns))}")
    return PredicateTable(frame=frame, labels=labels, kinds=kinds, members=members)


def cox_design_matrix(clinical: pd.DataFrame) -> pd.DataFrame:
    """One-hot clinical design for the Cox model. Missing levels become their own column."""
    numeric = clinical[list(NUMERIC_CLINICAL)].astype(float)
    numeric = numeric.fillna(numeric.median())
    numeric["tmb_nonsynonymous"] = np.log1p(numeric["tmb_nonsynonymous"].clip(lower=0))
    dummies = []
    reference = {"t_stage": "T1", "n_stage": "N0", "m_stage": "M0", "ajcc_stage": "I", "subtype": "LumA", "histology": "ductal", "radiation_therapy": "yes"}
    for column in NOMINAL_CLINICAL:
        encoded = pd.get_dummies(clinical[column], prefix=column, dtype=float)
        ref = f"{column}_{reference[column]}"
        if ref in encoded.columns:
            encoded = encoded.drop(columns=[ref])
        dummies.append(encoded)
    return pd.concat([numeric, *dummies], axis=1)
