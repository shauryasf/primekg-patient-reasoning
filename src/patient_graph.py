"""Patient graph construction from linked clinical, MAF, and discrete CNA evidence."""
from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import pandas as pd

from .graph_schema import add_typed_node, gene_node_id, new_patient_graph, patient_node_id, sample_node_id

PATIENT_ATTRIBUTES = ["SUBTYPE", "CANCER_TYPE_ACRONYM", "AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_M_STAGE"]
SAMPLE_ATTRIBUTES = ["ONCOTREE_CODE", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "TUMOR_TYPE", "ANEUPLOIDY_SCORE", "SAMPLE_TYPE"]


def _gene_node(graph: nx.MultiDiGraph, symbol: str, entrez_id: object) -> str:
    node = gene_node_id(symbol, entrez_id)
    if node not in graph:
        add_typed_node(graph, node, "Gene", gene_symbol=symbol, entrez_gene_id=entrez_id, identifier_namespace="HGNC_symbol")
    return node


def build_patient_graph(
    patient_row: pd.Series,
    sample_rows: pd.DataFrame,
    mutations: pd.DataFrame,
    cna_calls: pd.DataFrame,
    mutation_counts: pd.Series,
) -> nx.MultiDiGraph:
    """Build a compact, source-provenanced MultiDiGraph for a single patient."""
    patient_id = str(patient_row["PATIENT_ID"])
    graph = new_patient_graph(patient_id)
    patient_node = patient_node_id(patient_id)
    add_typed_node(graph, patient_node, "Patient", patient_id=patient_id, **{attribute.lower(): patient_row.get(attribute, "") for attribute in PATIENT_ATTRIBUTES})
    for _, sample in sample_rows.iterrows():
        sample_id = str(sample["SAMPLE_ID"])
        sample_node = sample_node_id(sample_id)
        add_typed_node(
            graph,
            sample_node,
            "Sample",
            sample_id=sample_id,
            patient_id=patient_id,
            mutation_record_count=int(mutation_counts.get(sample_id, 0)),
            **{attribute.lower(): sample.get(attribute, "") for attribute in SAMPLE_ATTRIBUTES},
        )
        graph.add_edge(patient_node, sample_node, key=f"has_sample:{sample_id}", edge_type="has_sample", source_file="data_clinical_sample.txt")
        for _, mutation in mutations[mutations["Tumor_Sample_Barcode"] == sample_id].iterrows():
            gene = _gene_node(graph, mutation["Hugo_Symbol"], mutation.get("Entrez_Gene_Id", ""))
            attributes: dict[str, Any] = {
                "edge_type": "has_mutation", "source_file": "data_mutations.txt", "source_study": "brca_tcga_pan_can_atlas_2018",
                "mutation_record_id": mutation["mutation_record_id"], "gene_symbol": mutation["Hugo_Symbol"],
            }
            for field in ["Variant_Classification", "Variant_Type", "Consequence", "HGVSp_Short", "HGVSc", "Chromosome", "Start_Position", "End_Position", "Reference_Allele", "Tumor_Seq_Allele2", "IMPACT", "FILTER", "NCALLERS"]:
                attributes[field.lower()] = mutation.get(field, "")
            graph.add_edge(sample_node, gene, key=mutation["mutation_record_id"], **attributes)
        for _, call in cna_calls[cna_calls["SAMPLE_ID"] == sample_id].iterrows():
            gene = _gene_node(graph, call["Hugo_Symbol"], call.get("Entrez_Gene_Id", ""))
            graph.add_edge(
                sample_node, gene, key=f"cna:{sample_id}:{call['cna_source_row']}", edge_type="has_CNA", source_file="data_cna.txt",
                source_study="brca_tcga_pan_can_atlas_2018", cna_value=int(call["cna_value"]), cna_category=call["cna_category"], gene_symbol=call["Hugo_Symbol"],
            )
    return graph


def graph_summary(graph: nx.MultiDiGraph) -> dict[str, object]:
    edge_counts = Counter(data["edge_type"] for _, _, data in graph.edges(data=True))
    samples = [data for _, data in graph.nodes(data=True) if data.get("node_type") == "Sample"]
    patient = next((data for _, data in graph.nodes(data=True) if data.get("node_type") == "Patient"), {})
    return {
        "patient_id": patient.get("patient_id", ""), "sample_ids": ";".join(sample["sample_id"] for sample in samples),
        "subtype": patient.get("subtype", ""), "tumor_stage": patient.get("ajcc_pathologic_tumor_stage", ""),
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "samples": len(samples),
        "genes": sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "Gene"),
        "mutation_edges": edge_counts["has_mutation"], "cna_edges": edge_counts["has_CNA"],
    }
