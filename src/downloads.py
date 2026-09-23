"""Fetch the public BRCA study tables and the PrimeKG edge list when they are absent."""
from __future__ import annotations

import csv
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import repository_root

LOGGER = logging.getLogger(__name__)

BRCA_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_tcga_pan_can_atlas_2018"
BRCA_FILES = ("data_clinical_patient.txt", "data_clinical_sample.txt", "data_cna.txt")
MUTATION_URL = (
    "https://www.cbioportal.org/api/molecular-profiles/"
    "brca_tcga_pan_can_atlas_2018_mutations/mutations/fetch"
    "?projection=DETAILED&pageSize=10000000"
)
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
MUTATION_COLUMNS = [
    "Hugo_Symbol",
    "Entrez_Gene_Id",
    "Tumor_Sample_Barcode",
    "Variant_Classification",
    "Variant_Type",
    "Consequence",
    "HGVSp_Short",
    "HGVSc",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "IMPACT",
    "FILTER",
    "NCALLERS",
]


def brca_dir(root: Path | None = None) -> Path:
    return (root or repository_root()) / "brca_tcga_pan_can_atlas_2018"


def primekg_csv_path(root: Path | None = None) -> Path:
    return (root or repository_root()) / "data" / "primekg" / "kg.csv"


def _download(url: str, destination: Path, min_bytes: int = 1000) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    if temporary.stat().st_size < min_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Download for {destination.name} was smaller than {min_bytes} bytes")
    temporary.replace(destination)


def _entrez_ids_from_cna(path: Path) -> list[int]:
    identifiers: set[int] = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if "Entrez_Gene_Id" not in header:
            raise ValueError(f"CNA file {path} has no Entrez_Gene_Id column")
        index = header.index("Entrez_Gene_Id")
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            if len(cells) <= index:
                continue
            text = cells[index].strip()
            if not text or text in {"NA", "N/A"}:
                continue
            try:
                identifiers.add(int(float(text)))
            except ValueError:
                continue
    return sorted(identifiers)


def _mutation_row(record: dict) -> dict[str, str] | None:
    gene = record.get("gene") or {}
    symbol = str(gene.get("hugoGeneSymbol") or "").strip()
    entrez = record.get("entrezGeneId")
    sample = str(record.get("sampleId") or "").strip()
    if not symbol or entrez in (None, "") or not sample:
        return None
    return {
        "Hugo_Symbol": symbol,
        "Entrez_Gene_Id": str(int(entrez)),
        "Tumor_Sample_Barcode": sample,
        "Variant_Classification": str(record.get("mutationType") or ""),
        "Variant_Type": str(record.get("variantType") or ""),
        "Consequence": "",
        "HGVSp_Short": str(record.get("proteinChange") or ""),
        "HGVSc": "",
        "Chromosome": str(record.get("chr") or ""),
        "Start_Position": "" if record.get("startPosition") is None else str(record.get("startPosition")),
        "End_Position": "" if record.get("endPosition") is None else str(record.get("endPosition")),
        "Reference_Allele": str(record.get("referenceAllele") or ""),
        "Tumor_Seq_Allele2": str(record.get("variantAllele") or ""),
        "IMPACT": "",
        "FILTER": "",
        "NCALLERS": "",
    }


def _fetch_mutation_batch(entrez_ids: list[int], attempts: int = 4) -> list[dict]:
    payload = json.dumps(
        {"entrezGeneIds": entrez_ids, "sampleListId": "brca_tcga_pan_can_atlas_2018_all"}
    ).encode()
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            MUTATION_URL,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read()
            parsed = json.loads(body) if body else []
            if isinstance(parsed, dict) and "message" in parsed:
                raise RuntimeError(parsed["message"])
            return list(parsed)
        except Exception as error:  # noqa: BLE001 - retry transient API failures
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Mutation batch failed for {entrez_ids[:3]}...: {last_error}")


def _write_mutations_from_api(dataset_dir: Path, entrez_ids: list[int]) -> None:
    destination = dataset_dir / "data_mutations.txt"
    batches = [entrez_ids[start : start + 200] for start in range(0, len(entrez_ids), 200)]
    LOGGER.info("Fetching BRCA mutations for %d genes in %d batches", len(entrez_ids), len(batches))
    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_fetch_mutation_batch, batch) for batch in batches]
        for done, future in enumerate(as_completed(futures), start=1):
            for record in future.result():
                row = _mutation_row(record)
                if row is not None:
                    rows.append(row)
            if done % 10 == 0 or done == len(futures):
                LOGGER.info("Mutation batches %d/%d, rows kept %d", done, len(futures), len(rows))
    rows.sort(key=lambda row: (row["Tumor_Sample_Barcode"], row["Hugo_Symbol"], row["Start_Position"], row["HGVSp_Short"]))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MUTATION_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %s (%d rows)", destination, len(rows))


def ensure_brca_dataset(root: Path | None = None) -> Path:
    """Return the study directory, downloading clinical/CNA tables and API mutations if needed."""
    dataset_dir = brca_dir(root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for filename in BRCA_FILES:
        path = dataset_dir / filename
        if path.exists() and path.stat().st_size > 1000 and not path.read_text(encoding="utf-8", errors="replace")[:40].startswith("version https://git-lfs"):
            continue
        LOGGER.info("Downloading %s", filename)
        _download(f"{BRCA_BASE}/{filename}", path)
    mutations = dataset_dir / "data_mutations.txt"
    if not mutations.exists() or mutations.stat().st_size < 10000:
        entrez_ids = _entrez_ids_from_cna(dataset_dir / "data_cna.txt")
        panel_path = (root or repository_root()) / "config" / "gene_panel_500.json"
        if panel_path.exists():
            panel = json.loads(panel_path.read_text(encoding="utf-8"))
            for info in panel.values():
                try:
                    entrez_ids.append(int(float(info["entrez_id"])))
                except (KeyError, TypeError, ValueError):
                    continue
            entrez_ids = sorted(set(entrez_ids))
        _write_mutations_from_api(dataset_dir, entrez_ids)
    return dataset_dir


def ensure_primekg_csv(root: Path | None = None) -> Path:
    """Return PrimeKG ``kg.csv``, downloading the Harvard Dataverse release when missing."""
    path = primekg_csv_path(root)
    if path.exists() and path.stat().st_size > 100_000_000:
        return path
    LOGGER.info("Downloading PrimeKG kg.csv")
    _download(PRIMEKG_URL, path, min_bytes=100_000_000)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_brca_dataset()
    ensure_primekg_csv()
