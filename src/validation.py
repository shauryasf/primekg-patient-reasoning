"""Graph consistency checks against source subsets."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

from .graph_schema import patient_node_id, sample_node_id
from .patient_graph import PATIENT_ATTRIBUTES, SAMPLE_ATTRIBUTES


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def validate_patient_graph(graph: nx.MultiDiGraph, patient_row: pd.Series, sample_rows: pd.DataFrame) -> ValidationResult:
    """Verify schema, source-consistent ownership, and absence of orphan gene nodes."""
    errors: list[str] = []
    warnings: list[str] = []
    patient_id = str(patient_row["PATIENT_ID"])
    expected_patient = patient_node_id(patient_id)
    if expected_patient not in graph or graph.nodes[expected_patient].get("patient_id") != patient_id:
        errors.append("patient node does not match source PATIENT_ID")
    elif any(str(graph.nodes[expected_patient].get(attribute.lower(), "")) != str(patient_row.get(attribute, "")) for attribute in PATIENT_ATTRIBUTES):
        errors.append("patient clinical attributes differ from source")
    expected_samples = set(sample_rows["SAMPLE_ID"].astype(str))
    actual_samples = {data.get("sample_id") for _, data in graph.nodes(data=True) if data.get("node_type") == "Sample"}
    if expected_samples != actual_samples:
        errors.append(f"sample nodes differ from source: expected={sorted(expected_samples)}, actual={sorted(actual_samples)}")
    for sample_id in expected_samples:
        sample_node = sample_node_id(sample_id)
        links = [data for _, target, data in graph.out_edges(expected_patient, data=True) if target == sample_node and data.get("edge_type") == "has_sample"]
        if len(links) != 1:
            errors.append(f"sample {sample_id} has {len(links)} patient links")
        source_sample = sample_rows.loc[sample_rows["SAMPLE_ID"] == sample_id].iloc[0]
        if any(str(graph.nodes[sample_node].get(attribute.lower(), "")) != str(source_sample.get(attribute, "")) for attribute in SAMPLE_ATTRIBUTES):
            errors.append(f"sample {sample_id} clinical attributes differ from source")
        for _, _, _, data in graph.out_edges(sample_node, keys=True, data=True):
            if data.get("edge_type") not in {"has_mutation", "has_CNA"}:
                errors.append(f"sample {sample_id} has unexpected edge type")
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "Gene" and graph.in_degree(node) == 0:
            errors.append(f"orphan gene node {node}")
    cna_seen: set[tuple[str, str]] = set()
    mutation_ids: set[str] = set()
    for source, target, _, data in graph.edges(keys=True, data=True):
        if data.get("edge_type") == "has_CNA":
            pair = (source, target)
            if pair in cna_seen:
                errors.append(f"duplicate CNA edge {pair}")
            cna_seen.add(pair)
            if data.get("cna_value") not in {-2, -1, 1, 2}:
                errors.append("CNA edge has invalid or neutral value")
        if data.get("edge_type") == "has_mutation":
            record_id = data.get("mutation_record_id")
            if not record_id or record_id in mutation_ids:
                errors.append("duplicate or missing mutation record identifier")
            mutation_ids.add(record_id)
    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)
