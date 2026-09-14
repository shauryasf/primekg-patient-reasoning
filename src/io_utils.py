"""Safe readers and lightweight audit helpers for cBioPortal study files."""
from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)
MISSING_TOKENS = {"", "NA", "N/A", "NULL", "NONE", "Unknown", "UNKNOWN", "[Not Available]"}


def normalize_identifier(value: object) -> str | None:
    """Return a trimmed identifier, retaining meaningful categorical values."""
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return None if cleaned in MISSING_TOKENS else cleaned


def normalize_gene_symbol(value: object) -> str | None:
    value = normalize_identifier(value)
    return value.upper() if value else None


def read_cbioportal_table(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a tab-delimited cBioPortal table, skipping metadata comment lines.

    Clinical files carry four comment metadata lines before their actual header;
    molecular tables normally start with the header. ``comment='#'`` handles both.
    """
    frame = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="warn",
        low_memory=False,
        **kwargs,
    )
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def audit_tabular_file(path: Path) -> dict[str, object]:
    """Stream a table to report schema, rows, and explicit missing-token counts."""
    header: list[str] | None = None
    missing: Counter[str] = Counter()
    rows = 0
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or (row[0].startswith("#") if row else False):
                continue
            if header is None:
                header = row
                continue
            rows += 1
            if len(row) != len(header):
                malformed += 1
                continue
            for column, value in zip(header, row):
                if value.strip() in MISSING_TOKENS:
                    missing[column] += 1
    if header is None:
        raise ValueError(f"No tabular header found in {path}")
    return {
        "file": path.name,
        "rows": rows,
        "columns": len(header),
        "column_names": header,
        "missing_counts": dict(missing),
        "malformed_rows": malformed,
    }


def graphml_value(value: object) -> str | int | float | bool:
    """Coerce values to GraphML-supported scalar attributes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_markdown_table(rows: Iterable[Iterable[object]], headers: Iterable[str]) -> str:
    headers = list(headers)
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(output)
