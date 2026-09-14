#!/usr/bin/env python
"""Build validated BRCA patient-specific NetworkX graphs from repository root."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clinical import link_patients_to_samples, load_clinical_data
from src.cna import cna_call_counts_by_sample, load_cna_for_samples
from src.config import load_feature_policy
from src.io_utils import audit_tabular_file, write_markdown_table
from src.mutations import load_mutations, mutation_record_counts
from src.patient_graph import build_patient_graph, graph_summary
from src.validation import validate_patient_graph
from src.visualization import draw_patient_graph

LOGGER = logging.getLogger("build_patient_graphs")
DATA_FILES = [
    "data_clinical_patient.txt", "data_clinical_sample.txt", "data_mutations.txt", "data_cna.txt", "data_log2_cna.txt", "data_armlevel_cna.txt",
    "data_mrna_seq_v2_rsem.txt", "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt", "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt",
    "data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt", "data_methylation_hm27_hm450_merged.txt", "data_methylation_hm450.txt", "data_sv.txt",
    "data_protein_quantification.txt", "data_protein_quantification_zscores.txt", "data_rppa.txt", "data_rppa_zscores.txt", "data_phosphoprotein_quantification.txt",
    "data_genetic_ancestry.txt", "data_resource_definition.txt", "data_resource_patient.txt", "data_timeline_sample_acquisition.txt", "data_clinical_supp_hypoxia.txt",
]


def safe_filename(identifier: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier)


def select_representative_patients(linked: pd.DataFrame, counts: pd.Series, cna_counts: pd.Series, size: int) -> list[str]:
    """Select deterministic cases spanning absent/high mutation burden, subtype, and CNA burden."""
    frame = linked[["PATIENT_ID", "SAMPLE_ID", "SUBTYPE", "AJCC_PATHOLOGIC_TUMOR_STAGE"]].copy()
    frame["mutation_count"] = frame["SAMPLE_ID"].map(counts).fillna(0).astype(int)
    frame["cna_count"] = frame["SAMPLE_ID"].map(cna_counts).fillna(0).astype(int)
    frame = frame.sort_values(["PATIENT_ID", "SAMPLE_ID"])
    selected: list[str] = []

    def add(row: pd.Series) -> None:
        if row["PATIENT_ID"] not in selected and len(selected) < size:
            selected.append(str(row["PATIENT_ID"]))

    zero = frame[frame["mutation_count"] == 0]
    if not zero.empty:
        add(zero.iloc[0])
    for subtype in ["BRCA_Basal", "BRCA_Her2", "BRCA_LumA", "BRCA_LumB"]:
        subgroup = frame[frame["SUBTYPE"] == subtype]
        if not subgroup.empty:
            target = subgroup["mutation_count"].median()
            add(subgroup.iloc[(subgroup["mutation_count"] - target).abs().argsort().iloc[0]])
    add(frame.loc[frame["mutation_count"].idxmax()])
    add(frame.loc[frame["cna_count"].idxmax()])
    for _, row in frame.sort_values(["mutation_count", "cna_count", "PATIENT_ID"]).iterrows():
        add(row)
        if len(selected) == size:
            break
    return selected


def create_dataset_audit(dataset_dir: Path, docs_dir: Path, outputs_dir: Path) -> dict[str, object]:
    """Inspect every requested raw data table with a streamed full-file audit."""
    records: list[dict[str, object]] = []
    for filename in DATA_FILES:
        path = dataset_dir / filename
        if path.exists():
            LOGGER.info("Auditing %s", filename)
            records.append(audit_tabular_file(path))
    payload = {"dataset": dataset_dir.name, "files": records}
    (outputs_dir / "dataset_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for record in records:
        columns = record["column_names"]
        missing = record["missing_counts"]
        top_missing = ", ".join(f"{key}:{value}" for key, value in sorted(missing.items(), key=lambda item: (-item[1], item[0]))[:4]) or "none"
        schema = ", ".join(columns[:5]) + (" …" if len(columns) > 5 else "")
        rows.append((record["file"], record["rows"], record["columns"], schema, top_missing, record["malformed_rows"]))
    content = "# Raw data audit\n\nThis report streams each requested raw tab-delimited file (including the large molecular matrices). Comment-prefixed cBioPortal metadata lines are excluded; the first non-comment line is the header. Missingness counts treat blank, `NA`, `N/A`, `NULL`, `NONE`, and `Unknown` as explicit missing tokens. Full column lists and all missing-count fields are in `outputs/dataset_audit.json`.\n\n"
    content += write_markdown_table(rows, ["File", "Rows", "Columns", "Schema / first columns", "Largest missing counts", "Malformed rows"]) + "\n\n"
    content += "## Milestone relevance\n\n"
    content += "- **Used:** `data_clinical_patient.txt` (patient context), `data_clinical_sample.txt` (sample context/link), `data_mutations.txt` (MAF mutation evidence), and `data_cna.txt` (discrete GISTIC gene-level CNA).\n"
    content += "- **Inspected but excluded:** continuous/log2 CNA and arm-level CNA are alternate resolutions; mRNA and z-score files are high-dimensional expression matrices; methylation, protein/RPPA/phosphoprotein are separate high-dimensional modalities; SV is a separate alteration modality; ancestry, resources, timeline, and hypoxia are metadata/supplemental rather than necessary for the initial graph.\n"
    content += "- `data_cna.txt` metadata defines values as -2 homozygous deletion, -1 hemizygous deletion, 0 neutral, 1 gain, 2 high-level amplification. `data_mutations.txt` is a 111-column MAF.\n"
    (docs_dir / "data_audit.md").write_text(content, encoding="utf-8")
    return payload


def write_quality_reports(outputs_dir: Path, clinical: object, linked: pd.DataFrame, mutations: pd.DataFrame, cna_counts: pd.Series, summaries: list[dict[str, object]], failures: list[dict[str, str]]) -> None:
    patient_features = ["SUBTYPE", "CANCER_TYPE_ACRONYM", "AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_M_STAGE"]
    sample_features = ["ONCOTREE_CODE", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "TUMOR_TYPE", "ANEUPLOIDY_SCORE", "SAMPLE_TYPE"]
    missingness = {column: int((linked[column].eq("") | linked[column].isin(["NA", "Unknown"])).sum()) for column in patient_features + sample_features}
    quality = {
        "patients": int(clinical.patients["PATIENT_ID"].nunique()), "samples": int(clinical.samples["SAMPLE_ID"].nunique()),
        "patients_with_multiple_samples": int((clinical.samples.groupby("PATIENT_ID")["SAMPLE_ID"].nunique() > 1).sum()),
        "mutation_records": int(len(mutations)), "unique_mutated_genes": int(mutations["Hugo_Symbol"].nunique()),
        "non_neutral_cna_records": int(cna_counts.sum()), "cna_samples": int(len(cna_counts)),
        "selected_clinical_missingness": missingness, "generated_patient_graphs": len(summaries),
        "failed_or_skipped_records": failures + clinical.malformed_rows,
    }
    (outputs_dir / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    markdown = "# Data quality report\n\n" + write_markdown_table(quality.items(), ["Metric", "Value"]) + "\n\n"
    markdown += "## Missing selected clinical attributes\n\n" + write_markdown_table(missingness.items(), ["Attribute", "Explicit missing values"]) + "\n"
    (outputs_dir / "data_quality_report.md").write_text(markdown, encoding="utf-8")


def write_mutation_count_validation(docs_dir: Path, linked: pd.DataFrame, counts: pd.Series) -> None:
    selected = linked[["PATIENT_ID", "SAMPLE_ID"]].sort_values("SAMPLE_ID").iloc[[0, 200, 500, 800, 1083]].copy()
    selected["raw_maf_record_count"] = selected["SAMPLE_ID"].map(counts).fillna(0).astype(int)
    rows = [(row.PATIENT_ID, row.SAMPLE_ID, row.raw_maf_record_count, "No MUTATION_COUNT field in this bundle") for row in selected.itertuples(index=False)]
    content = "# Mutation-count validation\n\nNeither `data_clinical_patient.txt` nor `data_clinical_sample.txt` contains a `MUTATION_COUNT` field, so there is no reported cBioPortal summary against which to compare. The pipeline therefore never labels its count as a reported clinical feature. `mutation_record_count` is explicitly the number of raw rows in `data_mutations.txt` grouped by `Tumor_Sample_Barcode`, including all MAF classifications represented in the source.\n\n"
    content += write_markdown_table(rows, ["Patient ID", "Sample ID", "Independent raw MAF row count", "Reported-count comparison"]) + "\n\n"
    content += "Each mutation edge is keyed by its original MAF row index. Automated tests assert that graph mutation-edge counts equal the independently grouped raw MAF-row count for deterministic source samples. This establishes the chosen representation without inventing a clinical mutation-burden formula.\n"
    (docs_dir / "mutation_count_validation.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "brca_tcga_pan_can_atlas_2018")
    parser.add_argument("--patient-id", action="append", help="Build this patient; repeatable.")
    parser.add_argument("--sample-size", type=int, default=8, help="Representative patient graph count when no patient is supplied.")
    parser.add_argument("--all", action="store_true", help="Build all patients (large: ~12M non-neutral CNA edges).")
    parser.add_argument("--skip-audit", action="store_true", help="Skip full raw-table audit regeneration.")
    parser.add_argument("--combined", action="store_true", help="Also write a disjoint combined GraphML for selected patients.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset_dir = args.dataset_dir.resolve()
    outputs = ROOT / "outputs"; docs = ROOT / "docs"; graphs_dir = outputs / "patient_graphs"; visualizations = outputs / "visualizations"
    for directory in [outputs, docs, graphs_dir, visualizations]: directory.mkdir(parents=True, exist_ok=True)
    load_feature_policy()  # validates that selection policy is parseable and present
    if not args.skip_audit:
        create_dataset_audit(dataset_dir, docs, outputs)
    clinical = load_clinical_data(dataset_dir)
    linked, unmapped = link_patients_to_samples(clinical)
    all_mutations, mutation_malformed = load_mutations(dataset_dir)
    mutation_counts = all_mutations.groupby("Tumor_Sample_Barcode").size()
    cna_counts, _, _ = cna_call_counts_by_sample(dataset_dir)
    if args.all:
        patient_ids = linked["PATIENT_ID"].sort_values().tolist()
    elif args.patient_id:
        patient_ids = list(dict.fromkeys(args.patient_id))
    else:
        patient_ids = select_representative_patients(linked, mutation_counts, cna_counts, args.sample_size)
    unknown = sorted(set(patient_ids) - set(linked["PATIENT_ID"]))
    if unknown:
        raise ValueError(f"Requested patient IDs absent from linked clinical data: {unknown}")
    selected_samples = linked[linked["PATIENT_ID"].isin(patient_ids)].copy()
    selected_ids = selected_samples["SAMPLE_ID"].tolist()
    selected_mutations = all_mutations[all_mutations["Tumor_Sample_Barcode"].isin(selected_ids)].copy()
    selected_cna, cna_issues = load_cna_for_samples(dataset_dir, selected_ids)
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = mutation_malformed + cna_issues + [{"source": "clinical_link", "row": str(row.SAMPLE_ID), "reason": "patient mapping absent"} for row in unmapped.itertuples()]
    graphs: list[nx.MultiDiGraph] = []
    for patient_id in patient_ids:
        patient_row = clinical.patients.loc[clinical.patients["PATIENT_ID"] == patient_id].iloc[0]
        samples = selected_samples[selected_samples["PATIENT_ID"] == patient_id]
        graph = build_patient_graph(patient_row, samples, selected_mutations, selected_cna, mutation_counts)
        result = validate_patient_graph(graph, patient_row, samples)
        if not result.valid:
            failures.append({"source": "graph_validation", "row": patient_id, "reason": "; ".join(result.errors)})
            raise ValueError(f"Invalid graph for {patient_id}: {result.errors}")
        graph_path = graphs_dir / f"{safe_filename(patient_id)}.graphml"
        nx.write_graphml(graph, graph_path)
        draw_patient_graph(graph, visualizations / f"{safe_filename(patient_id)}.png")
        summaries.append(graph_summary(graph))
        graphs.append(graph)
        LOGGER.info("Wrote %s (%s nodes, %s edges)", graph_path.name, graph.number_of_nodes(), graph.number_of_edges())
    pd.DataFrame(summaries).to_csv(outputs / "patient_summaries.csv", index=False)
    if args.combined:
        # Node IDs are intentionally globally meaningful (patient/sample IDs and
        # gene symbol+Entrez IDs), so compose preserves provenance and permits a
        # shared Gene node in this optional cohort view.
        combined = nx.compose_all(graphs)
        combined.graph.update({"source_dataset": "cBioPortal BRCA TCGA PanCancer Atlas 2018", "graph_scope": "selected_patients", "patient_count": len(graphs)})
        nx.write_graphml(combined, outputs / "all_patient_graphs.graphml")
    write_quality_reports(outputs, clinical, linked, all_mutations, cna_counts, summaries, failures)
    write_mutation_count_validation(docs, linked, mutation_counts)
    LOGGER.info("Generated %d validated graphs for %d selected patient IDs", len(summaries), len(patient_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
