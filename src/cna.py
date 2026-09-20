"""Discrete gene-level GISTIC CNA loading and non-neutral call extraction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import normalize_gene_symbol, read_cbioportal_table

CNA_STATES = {-2: "homozygous_deletion", -1: "hemizygous_deletion", 0: "neutral", 1: "gain", 2: "high_level_amplification"}


def load_cna_for_samples(dataset_dir: Path, sample_ids: list[str], gene_panel: set[str] | None = None, allowed_values: list[int] | None = None) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Read only selected CNA columns and return one row per non-neutral gene call."""
    path = dataset_dir / "data_cna.txt"
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    available = [sample for sample in sample_ids if sample in header]
    missing = [{"source": "data_cna.txt", "row": sample, "reason": "sample column absent"} for sample in sample_ids if sample not in header]
    if not available:
        return pd.DataFrame(columns=["Hugo_Symbol", "Entrez_Gene_Id", "SAMPLE_ID", "cna_value", "cna_category"]), missing
    matrix = read_cbioportal_table(path, usecols=["Hugo_Symbol", "Entrez_Gene_Id", *available])
    matrix["Hugo_Symbol"] = matrix["Hugo_Symbol"].map(normalize_gene_symbol)
    if gene_panel is not None:
        matrix = matrix[matrix["Hugo_Symbol"].isin(gene_panel)].copy()
    long = matrix.melt(id_vars=["Hugo_Symbol", "Entrez_Gene_Id"], var_name="SAMPLE_ID", value_name="cna_value")
    numeric = pd.to_numeric(long["cna_value"], errors="coerce")
    malformed_values = long[numeric.isna() | ~numeric.isin(CNA_STATES)]
    for index, row in malformed_values.iterrows():
        missing.append({"source": "data_cna.txt", "row": str(index), "reason": f"invalid CNA value {row['cna_value']}"})
    long = long.loc[~(numeric.isna() | ~numeric.isin(CNA_STATES))].copy()
    long["cna_source_row"] = long.index.astype(str)
    long["cna_value"] = pd.to_numeric(long["cna_value"], downcast="integer")
    
    if allowed_values:
        long = long[long["cna_value"].isin(allowed_values)].copy()
    else:
        long = long[long["cna_value"] != 0].copy()
        
    long = long.dropna(subset=["Hugo_Symbol"])
    identity = ["Hugo_Symbol", "Entrez_Gene_Id", "SAMPLE_ID"]
    # The source matrix contains repeated gene rows. Exact same-state calls are
    # structurally redundant; opposing values are ambiguous and cannot honestly
    # become one gene-level fact, so both are logged and withheld.
    exact_duplicates = long.duplicated(identity + ["cna_value"], keep="first")
    for _, row in long.loc[exact_duplicates].iterrows():
        missing.append({"source": "data_cna.txt", "row": f"{row['SAMPLE_ID']}:{row['cna_source_row']}", "reason": "duplicate exact discrete CNA call; deduplicated"})
    long = long.loc[~exact_duplicates].copy()
    conflicting = long.groupby(identity)["cna_value"].transform("nunique") > 1
    for _, row in long.loc[conflicting].iterrows():
        missing.append({"source": "data_cna.txt", "row": f"{row['SAMPLE_ID']}:{row['cna_source_row']}", "reason": "conflicting discrete CNA values for same gene/sample; excluded"})
    long = long.loc[~conflicting].copy()
    long["cna_category"] = long["cna_value"].map(CNA_STATES)
    return long, missing


def count_non_neutral_cna(dataset_dir: Path) -> tuple[int, int, int]:
    """Stream CNA matrix to count calls and avoid retaining a 25k x 1070 matrix."""
    path = dataset_dir / "data_cna.txt"
    total = 0
    rows = 0
    samples = 0
    for chunk in pd.read_csv(path, sep="\t", chunksize=1000):
        if samples == 0:
            samples = len(chunk.columns) - 2
        values = chunk.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
        total += int((values != 0).sum().sum())
        rows += len(chunk)
    return total, rows, samples


def cna_call_counts_by_sample(dataset_dir: Path) -> tuple[pd.Series, int, int]:
    """Stream per-sample non-neutral CNA counts for reporting/representative selection."""
    path = dataset_dir / "data_cna.txt"
    counts: pd.Series | None = None
    rows = 0
    samples = 0
    for chunk in pd.read_csv(path, sep="\t", chunksize=1000):
        if counts is None:
            counts = pd.Series(0, index=chunk.columns[2:], dtype="int64")
            samples = len(counts)
        values = chunk.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
        counts = counts.add((values != 0).sum(axis=0), fill_value=0).astype("int64")
        rows += len(chunk)
    return counts if counts is not None else pd.Series(dtype="int64"), rows, samples
