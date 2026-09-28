"""Draw the highest-scoring stored path from the run's own path report.

``outputs/gnn_benchmark_gsl/explanations.csv`` lists, for each explained patient,
the walks over edges that are already stored in that patient's graph, ranked by
``path_score`` (the mean edge importance along the walk). This script reads that
file, takes the single row with the largest ``path_score``, and draws it as a
chain that can be read left to right. Every node and every arrow comes from that
one row; the raw path string is printed on the figure so the drawing can be
checked against the CSV.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "gnn_benchmark_gsl" / "explanations.csv"
OUTPUT = ROOT / "outputs" / "gnn_benchmark_gsl" / "top_path_symbolic.png"

SPLIT = re.compile(r"\s+-([a-z_]+)->\s+")
NODE = re.compile(r"^(?P<ident>[^\s(]+)\s*(?:\((?P<name>.*)\))?$")
KIND_STYLE = {
    "patient": ("#4C78A8", "s", "Patient"),
    "sample": ("#59A14F", "o", "Sample"),
    "gene": ("#F2CF5B", "o", "Gene"),
    "protein": ("#F4A261", "o", "Protein"),
    "drug": ("#8CD17D", "^", "Drug"),
    "pathway": ("#B07AA1", "h", "Pathway"),
    "disease": ("#E15759", "d", "Disease"),
}


def parse_path(text: str) -> tuple[list[dict], list[str]]:
    """Split one stored walk into its nodes and the relation of each step."""
    parts = SPLIT.split(text)
    nodes: list[dict] = []
    for raw in parts[0::2]:
        match = NODE.match(raw.strip())
        ident = match.group("ident") if match else raw.strip()
        name = match.group("name") if match else None
        nodes.append({"raw": raw.strip(), "ident": ident, "name": name, "kind": ident.split(":", 1)[0]})
    relations = parts[1::2]
    assert len(nodes) == len(relations) + 1, text
    for node in nodes:
        if node["kind"] not in KIND_STYLE:
            raise ValueError(f"unknown node kind in {node['raw']!r}")
    return nodes, relations


def node_label(node: dict) -> str:
    """Two lines: the readable name, then the identifier the report printed."""
    ident, name = node["ident"], node["name"]
    if node["kind"] == "patient":
        return f"Patient\n{ident}"
    if node["kind"] == "sample":
        return f"Sample\n{name or ident}"
    if node["kind"] == "gene":
        return f"{name or ident}\nEntrez {ident.split(':')[-1]}"
    return f"{name or ident}\n{ident}"


def step_label(relation: str) -> str:
    """Readable relation text, wrapped so it fits between two nodes."""
    return textwrap.fill(relation.replace("_", " "), width=14)


def draw(output_path: Path = OUTPUT, report_path: Path = REPORT) -> Path:
    """Draw the walk with the largest path_score as a left-to-right chain."""
    report = pd.read_csv(report_path)
    row = report.loc[report["path_score"].idxmax()]
    nodes, relations = parse_path(str(row["path"]))

    fig, axis = plt.subplots(figsize=(14.0, 5.9))
    xs = [float(index) for index in range(len(nodes))]
    for index, node in enumerate(nodes):
        color, shape, _ = KIND_STYLE[node["kind"]]
        axis.scatter(
            [xs[index]],
            [0.0],
            s=2300,
            marker=shape,
            c=color,
            edgecolors="#333333",
            linewidths=1.3,
            zorder=3,
        )
        axis.text(
            xs[index],
            -0.30,
            node_label(node),
            ha="center",
            va="top",
            fontsize=10,
            linespacing=1.35,
        )
    for index, relation in enumerate(relations):
        axis.annotate(
            "",
            xy=(xs[index + 1] - 0.10, 0.0),
            xytext=(xs[index] + 0.10, 0.0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#4C4C4C",
                "linewidth": 2.0,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=2,
        )
        axis.text(
            (xs[index] + xs[index + 1]) / 2.0,
            0.16,
            step_label(relation),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
            linespacing=1.3,
        )

    axis.set_title(
        f"Highest-scoring stored path for {row['patient_id']}: "
        f"{nodes[-1]['name']} at path_score {float(row['path_score']):.3f}",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    axis.set_xlim(-0.62, len(nodes) - 0.38)
    axis.set_ylim(-1.30, 1.30)
    axis.axis("off")

    note = (
        f"{row['model']} risk for this patient: {float(row['y_prob']):.3f}   "
        f"path rank {int(row['path_rank'])} of the report   "
        f"edge importances from {row['score_source']}\n"
        "path_score is the mean edge importance along the walk. The walk uses only edges "
        "already stored in this patient's graph: no step is inferred."
    )
    axis.text(
        0.011,
        0.985,
        note,
        transform=axis.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        color="#333333",
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC"},
    )
    axis.text(
        0.011,
        0.085,
        "largest path_score row of "
        f"{report_path.relative_to(ROOT).as_posix()}, drawn as stored:",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        color="#555555",
    )
    axis.text(
        0.011,
        0.025,
        str(row["path"]),
        transform=axis.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        color="#555555",
        family="monospace",
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=KIND_STYLE[node["kind"]][1],
                color="w",
                label=KIND_STYLE[node["kind"]][2],
                markerfacecolor=KIND_STYLE[node["kind"]][0],
                markeredgecolor="#333333",
                markersize=13,
            )
            for node in nodes
        ],
        loc="upper right",
        ncol=len(nodes),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.995, 0.88),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    report = pd.read_csv(REPORT)
    row = report.loc[report["path_score"].idxmax()]
    nodes, relations = parse_path(str(row["path"]))
    print(f"{len(report)} stored paths in {REPORT.relative_to(ROOT).as_posix()}")
    print(
        f"highest path_score: {row['patient_id']} rank {int(row['path_rank'])} "
        f"score {float(row['path_score']):.3f} via {row['score_source']}"
    )
    print("nodes:", [node["raw"] for node in nodes])
    print("steps:", relations)
    print("wrote", draw())


if __name__ == "__main__":
    main()
