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
    fig, axis = plt.subplots(figsize=(15, 10))
    nx.draw_networkx_nodes(subgraph, position, nodelist=patient_nodes, node_color="#4C78A8", node_shape="s", node_size=1300, ax=axis, label="Patient")
    nx.draw_networkx_nodes(subgraph, position, nodelist=sample_nodes, node_color="#59A14F", node_shape="o", node_size=1100, ax=axis, label="Sample")
    nx.draw_networkx_nodes(subgraph, position, nodelist=list(chosen), node_color="#F2CF5B", node_size=500, ax=axis, label="Gene")
    nx.draw_networkx_edges(subgraph, position, edgelist=[(u, v) for u, v, d in subgraph.edges(data=True) if d.get("edge_type") == "has_sample"], edge_color="#555555", arrows=True, ax=axis)
    nx.draw_networkx_edges(subgraph, position, edgelist=[(u, v) for u, v, d in subgraph.edges(data=True) if d.get("edge_type") == "has_mutation"], edge_color="#E15759", arrows=True, connectionstyle="arc3,rad=0.08", ax=axis)
    nx.draw_networkx_edges(subgraph, position, edgelist=[(u, v) for u, v, d in subgraph.edges(data=True) if d.get("edge_type") == "has_CNA"], edge_color="#B07AA1", arrows=True, connectionstyle="arc3,rad=-0.08", ax=axis)
    labels = {node: data.get("gene_symbol", data.get("sample_id", data.get("patient_id", node))) for node, data in subgraph.nodes(data=True)}
    nx.draw_networkx_labels(subgraph, position, labels=labels, font_size=7, ax=axis)

    # Extract clinical context to display on canvas
    patient_info = [data for _, data in graph.nodes(data=True) if data.get("node_type") == "Patient"]
    sample_info = [data for _, data in graph.nodes(data=True) if data.get("node_type") == "Sample"]

    p_lines = []
    if patient_info:
        p = patient_info[0]
        p_lines = [
            f"Patient ID: {p.get('patient_id', 'N/A')}",
            f"Age: {p.get('age', 'N/A')}  |  Sex: {p.get('sex', 'N/A')}",
            f"Subtype: {p.get('subtype', 'N/A')}",
            f"Tumor Stage: {p.get('ajcc_pathologic_tumor_stage', 'N/A')}",
            f"Metastasis (M): {p.get('path_m_stage', 'N/A')}",
        ]
    s_lines = []
    if sample_info:
        s = sample_info[0]
        s_lines = [
            f"Sample ID: {s.get('sample_id', 'N/A')}",
            f"Oncotree: {s.get('oncotree_code', 'N/A')}",
            f"Sample Type: {s.get('sample_type', 'N/A')}",
            f"Aneuploidy Score: {s.get('aneuploidy_score', 'N/A')}",
            f"Raw MAF Rows: {s.get('mutation_record_count', 0)}",
        ]
    box_text = "PATIENT ATTRIBUTES\n" + "\n".join(p_lines) + "\n\nSAMPLE ATTRIBUTES\n" + "\n".join(s_lines)
    props = dict(boxstyle="round,pad=0.6", facecolor="#EBF2FA", alpha=0.9, edgecolor="#4C78A8", linewidth=1.5)
    axis.text(0.02, 0.98, box_text, transform=axis.transAxes, fontsize=8.5, verticalalignment="top", fontfamily="monospace", bbox=props)

    axis.set_title(f"Patient Graph: {graph.graph['patient_id']} (Genes capped at {max_gene_nodes})", fontsize=13, fontweight="bold", pad=12)
    axis.legend(handles=[
        Line2D([0], [0], marker="s", color="w", label="Patient Node", markerfacecolor="#4C78A8", markersize=13),
        Line2D([0], [0], marker="o", color="w", label="Sample Node", markerfacecolor="#59A14F", markersize=13),
        Line2D([0], [0], marker="o", color="w", label="Gene Node", markerfacecolor="#F2CF5B", markersize=11),
        Line2D([0], [0], color="#555555", label="has_sample"),
        Line2D([0], [0], color="#E15759", label="has_mutation"),
        Line2D([0], [0], color="#B07AA1", label="has_CNA"),
    ], loc="upper right", framealpha=0.9, fontsize=9)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

