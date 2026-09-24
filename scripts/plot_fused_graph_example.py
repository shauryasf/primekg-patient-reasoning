"""Draw one fused-graph neighborhood from the stored explanation walks.

The full fused graph also contains the other PrimeKG edges on this patient's
genes. This figure keeps the walks written for TCGA-AO-A0J5 so the shape is readable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "gnn_benchmark_36m" / "fused_graph_example.png"


def draw(output_path: Path = OUTPUT) -> Path:
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            ("patient", {"label": "Patient\nTCGA-AO-A0J5", "kind": "patient"}),
            ("sample", {"label": "Sample\nTCGA-AO-A0J5-01", "kind": "sample"}),
            ("akt3", {"label": "AKT3\namplified", "kind": "altered"}),
            ("actn2", {"label": "ACTN2\namplified", "kind": "altered"}),
            ("pik3r1", {"label": "PIK3R1\npartner protein", "kind": "partner"}),
            ("akt1", {"label": "AKT1\npartner protein", "kind": "partner"}),
            ("pathway_pi3k", {"label": "Pathway\nDownstream signal transduction", "kind": "pathway"}),
            ("disease", {"label": "Disease\nbreast adenocarcinoma", "kind": "disease"}),
            ("pathway_nephrin", {"label": "Pathway\nNephrin family interactions", "kind": "pathway"}),
        ]
    )
    graph.add_edges_from(
        [
            ("patient", "sample", {"relation": "has_sample"}),
            ("sample", "akt3", {"relation": "has_amplification"}),
            ("sample", "actn2", {"relation": "has_amplification"}),
            ("akt3", "pik3r1", {"relation": "protein_protein"}),
            ("akt3", "akt1", {"relation": "protein_protein"}),
            ("pik3r1", "pathway_pi3k", {"relation": "pathway_protein"}),
            ("akt1", "disease", {"relation": "disease_protein"}),
            ("actn2", "pathway_nephrin", {"relation": "pathway_protein"}),
        ]
    )
    position = {
        "patient": (0.0, 4.35),
        "sample": (0.0, 3.15),
        "akt3": (-1.7, 1.9),
        "actn2": (1.7, 1.9),
        "pik3r1": (-2.45, 0.55),
        "akt1": (-0.85, 0.55),
        "pathway_pi3k": (-2.45, -0.85),
        "disease": (-0.85, -0.85),
        "pathway_nephrin": (1.7, 0.4),
    }
    colors = {
        "patient": "#4C78A8",
        "sample": "#59A14F",
        "altered": "#F2CF5B",
        "partner": "#F4A261",
        "pathway": "#B07AA1",
        "disease": "#E15759",
    }
    edge_colors = {
        "has_sample": "#4C4C4C",
        "has_amplification": "#B07AA1",
        "protein_protein": "#E07A3D",
        "pathway_protein": "#6B4C7A",
        "disease_protein": "#C44E52",
    }
    fig, axis = plt.subplots(figsize=(12.5, 8.2))
    for kind, shape, size in (
        ("patient", "s", 900),
        ("sample", "o", 900),
        ("altered", "o", 780),
        ("partner", "o", 780),
        ("pathway", "h", 900),
        ("disease", "d", 780),
    ):
        nodelist = [node for node, data in graph.nodes(data=True) if data["kind"] == kind]
        nx.draw_networkx_nodes(
            graph,
            position,
            nodelist=nodelist,
            node_color=colors[kind],
            node_shape=shape,
            node_size=size,
            linewidths=1.2,
            edgecolors="#333333",
            ax=axis,
        )
    for relation, color in edge_colors.items():
        edgelist = [(u, v) for u, v, data in graph.edges(data=True) if data["relation"] == relation]
        nx.draw_networkx_edges(
            graph,
            position,
            edgelist=edgelist,
            edge_color=color,
            width=1.8,
            arrows=True,
            arrowsize=16,
            connectionstyle="arc3,rad=0.04",
            ax=axis,
            node_size=900,
        )
    label_shift = {"sample": (0.72, -0.02)}
    for node, (x_pos, y_pos) in position.items():
        dx, dy = label_shift.get(node, (0.0, -0.28))
        axis.text(
            x_pos + dx,
            y_pos + dy,
            graph.nodes[node]["label"],
            ha="left" if node == "sample" else "center",
            va="center" if node == "sample" else "top",
            fontsize=8,
            linespacing=1.25,
        )
    axis.set_title(
        "Fused graph neighborhood for TCGA-AO-A0J5",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    axis.legend(
        handles=[
            Line2D([0], [0], marker="s", color="w", label="Patient", markerfacecolor=colors["patient"], markeredgecolor="#333333", markersize=12),
            Line2D([0], [0], marker="o", color="w", label="Sample", markerfacecolor=colors["sample"], markeredgecolor="#333333", markersize=12),
            Line2D([0], [0], marker="o", color="w", label="Altered gene", markerfacecolor=colors["altered"], markeredgecolor="#333333", markersize=12),
            Line2D([0], [0], marker="o", color="w", label="PrimeKG partner protein", markerfacecolor=colors["partner"], markeredgecolor="#333333", markersize=12),
            Line2D([0], [0], marker="h", color="w", label="Pathway", markerfacecolor=colors["pathway"], markeredgecolor="#333333", markersize=12),
            Line2D([0], [0], marker="d", color="w", label="Disease", markerfacecolor=colors["disease"], markeredgecolor="#333333", markersize=12),
            Line2D([0], [0], color=edge_colors["has_sample"], lw=2, label="has_sample"),
            Line2D([0], [0], color=edge_colors["has_amplification"], lw=2, label="has_high_level_amplification"),
            Line2D([0], [0], color=edge_colors["protein_protein"], lw=2, label="protein_protein"),
            Line2D([0], [0], color=edge_colors["pathway_protein"], lw=2, label="pathway_protein"),
            Line2D([0], [0], color=edge_colors["disease_protein"], lw=2, label="disease_protein"),
        ],
        loc="upper right",
        framealpha=0.95,
        fontsize=8,
    )
    note = (
        "AKT3–PIK3R1 and AKT3–AKT1 are PrimeKG protein_protein edges.\n"
        "They are interaction records between two different proteins,\n"
        "not a link created because both genes are altered in this sample.\n"
        "Other PrimeKG edges on these genes are left off so this neighborhood stays readable."
    )
    axis.text(
        0.01,
        0.98,
        note,
        transform=axis.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        color="#333333",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC"},
    )
    axis.set_xlim(-3.9, 3.6)
    axis.set_ylim(-1.7, 4.85)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(draw())
