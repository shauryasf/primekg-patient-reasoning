"""NetworkX MultiDiGraph schema and stable node identifiers."""
from __future__ import annotations

from typing import Any

import networkx as nx

from .io_utils import graphml_value

SOURCE_STUDY = "brca_tcga_pan_can_atlas_2018"
SOURCE_DATASET = "cBioPortal BRCA TCGA PanCancer Atlas 2018"


def patient_node_id(patient_id: str) -> str:
    return f"patient:{patient_id}"


def sample_node_id(sample_id: str) -> str:
    return f"sample:{sample_id}"


def gene_node_id(symbol: str, entrez_id: object) -> str:
    return f"gene:{symbol}:{str(entrez_id).strip() or 'unknown'}"


def add_typed_node(graph: nx.MultiDiGraph, node_id: str, node_type: str, **attributes: Any) -> None:
    graph.add_node(node_id, node_type=node_type, **{key: graphml_value(value) for key, value in attributes.items()})


def new_patient_graph(patient_id: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        source_dataset=SOURCE_DATASET,
        source_study=SOURCE_STUDY,
        graph_generation_version="0.1.0",
        graph_scope="single_patient",
        patient_id=patient_id,
    )
    return graph
