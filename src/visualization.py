"""Compact, evidence-focused patient graph visualization."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def draw_patient_graph(graph: nx.MultiDiGraph, output_path: Path, max_gene_nodes: int = 30) -> None:
    """Render patient/sample plus selected mutation/CNA genes, not the full CNA matrix."""
    patient_nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "Patient"]
    sample_nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "Sample"]
    mutation_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get("edge_type") == "has_mutation"]
    cna_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get("edge_type") == "has_CNA"]
    chosen = {v for _, v, _ in mutation_edges[:max_gene_nodes]}
    for _, target, data in sorted(cna_edges, key=lambda item: abs(int(item[2]["cna_value"])), reverse=True):
        if len(chosen) >= max_gene_nodes:
            break
        chosen.add(target)
    visible = set(patient_nodes + sample_nodes) | chosen
    subgraph = graph.subgraph(visible).copy()
    position = nx.spring_layout(subgraph, seed=17, k=1.2)
    fig, axis = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(subgraph, position, nodelist=patient_nodes, node_color="#4C78A8", node_shape="s", node_size=1250, ax=axis, label="Patient")
    nx.draw_networkx_nodes(subgraph, position, nodelist=sample_nodes, node_color="#59A14F", node_shape="o", node_size=1050, ax=axis, label="Sample")
    nx.draw_networkx_nodes(subgraph, position, nodelist=list(chosen), node_color="#F2CF5B", node_size=500, ax=axis, label="Gene")
    nx.draw_networkx_edges(subgraph, position, edgelist=[(u, v) for u, v, d in subgraph.edges(data=True) if d.get("edge_type") == "has_sample"], edge_color="#555555", arrows=True, ax=axis)
    nx.draw_networkx_edges(subgraph, position, edgelist=[(u, v) for u, v, d in subgraph.edges(data=True) if d.get("edge_type") == "has_mutation"], edge_color="#E15759", arrows=True, connectionstyle="arc3,rad=0.08", ax=axis)
    nx.draw_networkx_edges(subgraph, position, edgelist=[(u, v) for u, v, d in subgraph.edges(data=True) if d.get("edge_type") == "has_CNA"], edge_color="#B07AA1", arrows=True, connectionstyle="arc3,rad=-0.08", ax=axis)
    labels = {node: data.get("gene_symbol", data.get("sample_id", data.get("patient_id", node))) for node, data in subgraph.nodes(data=True)}
    nx.draw_networkx_labels(subgraph, position, labels=labels, font_size=7, ax=axis)
    axis.set_title(f"Patient graph: {graph.graph['patient_id']} (display capped at {max_gene_nodes} genes)")
    axis.legend(handles=[
        Line2D([0], [0], marker="s", color="w", label="Patient", markerfacecolor="#4C78A8", markersize=14),
        Line2D([0], [0], marker="o", color="w", label="Sample", markerfacecolor="#59A14F", markersize=14),
        Line2D([0], [0], marker="o", color="w", label="Gene", markerfacecolor="#F2CF5B", markersize=12),
        Line2D([0], [0], color="#E15759", label="has_mutation"),
        Line2D([0], [0], color="#B07AA1", label="has_CNA"),
    ], loc="upper left")
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
