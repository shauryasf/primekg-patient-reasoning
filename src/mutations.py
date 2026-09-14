"""MAF mutation ingestion and source-preserving mutation records."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import normalize_gene_symbol, normalize_identifier, read_cbioportal_table

MUTATION_FIELDS = [
    "Hugo_Symbol", "Entrez_Gene_Id", "Tumor_Sample_Barcode", "Variant_Classification",
    "Variant_Type", "Consequence", "HGVSp_Short", "HGVSc", "Chromosome", "Start_Position",
    "End_Position", "Reference_Allele", "Tumor_Seq_Allele2", "IMPACT", "FILTER", "NCALLERS",
]


def load_mutations(dataset_dir: Path, sample_ids: set[str] | None = None) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load MAF mutation records without collapsing biologically distinct calls."""
    frame = read_cbioportal_table(dataset_dir / "data_mutations.txt", usecols=lambda c: c in MUTATION_FIELDS)
    frame["Tumor_Sample_Barcode"] = frame["Tumor_Sample_Barcode"].map(normalize_identifier)
    frame["Hugo_Symbol"] = frame["Hugo_Symbol"].map(normalize_gene_symbol)
    malformed: list[dict[str, str]] = []
    invalid = frame[["Tumor_Sample_Barcode", "Hugo_Symbol"]].isna().any(axis=1)
    for index in frame.index[invalid]:
        malformed.append({"source": "data_mutations.txt", "row": str(index), "reason": "missing sample or gene"})
    frame = frame.loc[~invalid].copy()
    if sample_ids is not None:
        frame = frame[frame["Tumor_Sample_Barcode"].isin(sample_ids)].copy()
    frame["mutation_record_id"] = [f"maf:{index}" for index in frame.index]
    return frame, malformed


def mutation_record_counts(mutations: pd.DataFrame, sample_ids: list[str] | None = None) -> pd.Series:
    """Count raw MAF rows per tumour sample; this is not a clinical MUTATION_COUNT field."""
    counts = mutations.groupby("Tumor_Sample_Barcode").size().rename("mutation_record_count")
    if sample_ids is not None:
        counts = counts.reindex(sample_ids, fill_value=0)
    return counts
