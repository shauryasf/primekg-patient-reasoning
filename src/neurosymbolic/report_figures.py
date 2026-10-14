"""Figures for the patient and group reports.

Every figure is drawn for print: a white surface, hairline recessive axes, 2 px
lines, thin bars with a capped thickness, and text in ink colors, never in a
series color. Node types in the path diagrams are a fixed categorical order
(patient, gene, pathway, drug, disease) and always carry a text label too.
"""
from __future__ import annotations

import io
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .timing import GROUP_RANGES

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID_GRAY = "#e4e3df"
SURFACE = "#ffffff"
NAVY = "#0d366b"
# Reference categorical order: blue, orange, aqua, yellow, magenta.
NODE_COLORS = {
    "patient": "#2a78d6",
    "gene": "#eb6834",
    "pathway": "#1baf7a",
    "drug": "#eda100",
    "disease": "#e87ba4",
}
NODE_LABELS = {"patient": "Patient", "gene": "Gene", "pathway": "Pathway", "drug": "Drug", "disease": "Disease"}
RISK_UP = "#e34948"
RISK_DOWN = "#2a78d6"
NEUTRAL = "#b9b8b2"
PATIENT_LINE = "#2a78d6"
COHORT_LINE = "#8a8984"
LEADER = {"arrowstyle": "-", "color": INK_3, "linewidth": 0.7, "shrinkA": 0, "shrinkB": 3}
LABEL_BOX = {"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9}
GROUP_COLORS = {"A": "#e34948", "B": "#eb6834", "C": "#c98500", "D": "#1c5cab", "N": "#008300"}
RELATION_TEXT = {
    "has_mutation": "mutated",
    "has_high_level_amplification": "amplified",
    "has_homozygous_deletion": "deleted",
    "protein_protein": "interacts with",
    "pathway_protein": "in pathway",
    "drug_protein": "targeted by",
    "disease_protein": "linked to",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": GRID_GRAY,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    }
)


def to_png(fig, dpi: int = 220) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return buffer.getvalue()


def _recessive(axis) -> None:
    axis.grid(axis="y", color=GRID_GRAY, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)


def survival_figure(
    grid: np.ndarray,
    curve: np.ndarray,
    cohort_times: np.ndarray,
    cohort_survival: np.ndarray,
    recorded_months: float,
    progressed: bool,
    predicted: tuple[float, float, float] | None,
    width: float = 7.2,
    height: float = 2.5,
) -> bytes:
    """The patient's out-of-fold PFS curve against the cohort Kaplan-Meier curve."""
    fig, axis = plt.subplots(figsize=(width, height))
    for code, color in (("A", "#fbe3e3"), ("B", "#fdeee6"), ("C", "#fdf3dc")):
        low, high = GROUP_RANGES[code]
        axis.axvspan(low, high, color=color, linewidth=0, zorder=0)
    axis.step(cohort_times, cohort_survival, where="post", color=COHORT_LINE, linewidth=1.6, label="Cohort (Kaplan–Meier)")
    axis.plot(grid, curve, color=PATIENT_LINE, linewidth=2.2, label="This patient (model)", solid_capstyle="round")
    if predicted is not None and np.isfinite(predicted[0]):
        month, low, high = predicted
        axis.axvspan(low, high, ymin=0.0, ymax=0.025, color=PATIENT_LINE, alpha=0.45, linewidth=0)
        axis.plot([month], [np.interp(month, grid, curve)], marker="o", markersize=6, color=PATIENT_LINE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        axis.annotate(f"● model estimate if progression occurs: {month:.0f} mo", (month, np.interp(month, grid, curve)), xytext=(0.52, 0.86), textcoords="axes fraction", fontsize=7, color=INK_2, bbox=LABEL_BOX, arrowprops=LEADER)
    if np.isfinite(recorded_months):
        y = np.interp(recorded_months, grid, curve)
        marker = "X" if progressed else "|"
        axis.plot([recorded_months], [y], marker=marker, markersize=9 if progressed else 12, color=INK, markeredgecolor=SURFACE if progressed else INK, markeredgewidth=1.2 if progressed else 2, zorder=6)
        label = "recorded progression" if progressed else "last follow-up (censored)"
        axis.annotate(f"{'✕' if progressed else '|'} {label}: {recorded_months:.1f} mo", (recorded_months, y), xytext=(0.52, 0.95), textcoords="axes fraction", fontsize=7, color=INK, bbox=LABEL_BOX, arrowprops=LEADER)
    axis.set_xlim(0, 180)
    axis.set_ylim(0, 1.02)
    axis.set_xticks([0, 12, 24, 36, 60, 120, 180])
    axis.set_xlabel("Months from diagnosis")
    axis.set_ylabel("Progression-free probability")
    _recessive(axis)
    for code, (low, high) in GROUP_RANGES.items():
        if np.isfinite(high):
            axis.text((low + high) / 2, 0.05, code, ha="center", va="bottom", fontsize=7, color=INK_3, fontweight="bold")
    axis.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.08), fontsize=7)
    return to_png(fig)


def _node_text(node: dict) -> tuple[str, str]:
    kind = node["type"]
    name = str(node["name"])
    if kind == "patient":
        return NODE_LABELS[kind], name.replace("TCGA-", "")
    # textwrap.shorten drops a long unbroken chemical name entirely, so cut by characters.
    return NODE_LABELS.get(kind, kind.title()), name if len(name) <= 30 else name[:29] + "…"


def path_figure(paths: list[dict], width: float = 7.2) -> bytes:
    """Concise chain diagrams, one row per selected path, with its score bar on the right.

    The sample node is folded into the first arrow ("mutated", "amplified"),
    because it carries no information beyond the patient in a one-sample cohort.
    """
    rows = max(len(paths), 1)
    fig = plt.figure(figsize=(width, 0.86 * rows + 0.45))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, rows + 0.55)
    axis.axis("off")
    chain_right = 78
    for row, path in enumerate(paths):
        y = rows - row - 0.35
        nodes = [node for node in json.loads(path["nodes"]) if node["type"] != "sample"]
        relations = json.loads(path["relations"])[1:]
        n = len(nodes)
        gap = chain_right / n
        box_w = min(gap * 0.64, 16.0)
        centers = [gap * (index + 0.5) + 1.5 for index in range(n)]
        axis.text(0.2, y + 0.36, f"{path['rank']}", fontsize=9, fontweight="bold", color=INK, va="center")
        for index, (node, x) in enumerate(zip(nodes, centers)):
            color = NODE_COLORS.get(node["type"], NEUTRAL)
            box = FancyBboxPatch((x - box_w / 2, y - 0.27), box_w, 0.54, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=0, facecolor=color, alpha=0.16)
            axis.add_patch(box)
            axis.add_patch(Rectangle((x - box_w / 2, y - 0.27), 0.7, 0.54, color=color, linewidth=0))
            kind, name = _node_text(node)
            axis.text(x - box_w / 2 + 1.4, y + 0.11, kind.upper(), fontsize=5.6, color=INK_2, va="center", fontweight="bold")
            axis.text(x - box_w / 2 + 1.4, y - 0.1, textwrap.fill(name, 17, break_long_words=True), fontsize=6.2 if len(name) <= 17 else 5.4, color=INK, va="center", linespacing=0.95)
            if index < n - 1:
                start, end = x + box_w / 2 + 0.3, centers[index + 1] - box_w / 2 - 0.3
                axis.add_patch(FancyArrowPatch((start, y), (end, y), arrowstyle="-|>", mutation_scale=7, color=INK_3, linewidth=0.9))
                relation = RELATION_TEXT.get(relations[index], relations[index].replace("_", " "))
                axis.text((start + end) / 2, y + 0.06, textwrap.fill(relation, 11), fontsize=5.2, color=INK_2, ha="center", va="bottom", linespacing=0.9)
        score = float(path["ns_score"])
        bar_x, bar_w = 81, 15
        axis.add_patch(Rectangle((bar_x, y - 0.06), bar_w, 0.12, color=GRID_GRAY, linewidth=0))
        neural_part = 0.6 * float(path["neural"]) * bar_w
        symbolic_part = 0.4 * float(path["symbolic"]) * bar_w
        axis.add_patch(Rectangle((bar_x, y - 0.06), neural_part, 0.12, color=NAVY, linewidth=0))
        direction = path["evidence_direction"]
        sym_color = RISK_UP if direction == "faster progression" else RISK_DOWN if direction == "slower progression" else NEUTRAL
        axis.add_patch(Rectangle((bar_x + neural_part, y - 0.06), symbolic_part, 0.12, color=sym_color, linewidth=0))
        axis.text(bar_x, y + 0.22, f"NS score {score:.2f}" + ("  ★ prominent" if path["prominent"] else ""), fontsize=6.4, color=INK, fontweight="bold")
        if np.isfinite(path.get("fact_hazard_ratio", np.nan)):
            evidence = f"HR {path['fact_hazard_ratio']:.2f} · n={int(path['fact_carriers'])}"
        else:
            evidence = "no cohort evidence"
        axis.text(bar_x, y - 0.26, evidence, fontsize=5.8, color=INK_2)
    legend_y = rows + 0.3
    x = 1.5
    for kind in ("patient", "gene", "pathway", "drug", "disease"):
        axis.add_patch(Rectangle((x, legend_y - 0.07), 1.6, 0.14, color=NODE_COLORS[kind], linewidth=0))
        axis.text(x + 2.1, legend_y, NODE_LABELS[kind], fontsize=6.2, color=INK_2, va="center")
        x += 10
    for label, color in (("neural", NAVY), ("KG evidence ↑ risk", RISK_UP), ("KG evidence ↓ risk", RISK_DOWN)):
        axis.add_patch(Rectangle((x + 4, legend_y - 0.07), 1.6, 0.14, color=color, linewidth=0))
        axis.text(x + 6.1, legend_y, label, fontsize=6.2, color=INK_2, va="center")
        x += 15
    return to_png(fig)


def gene_bar_figure(genes: list[tuple[str, float, str]], width: float = 3.4) -> bytes:
    """GNN importance of the patient's most important altered genes (single series, no legend)."""
    genes = genes[:8][::-1]
    fig, axis = plt.subplots(figsize=(width, 0.26 * len(genes) + 0.55))
    positions = np.arange(len(genes))
    values = [value for _, value, _ in genes]
    axis.barh(positions, values, height=0.55, color=NODE_COLORS["gene"])
    axis.set_yticks(positions, [f"{gene} ({kind})" for gene, _, kind in genes], fontsize=7, color=INK)
    for position, value in zip(positions, values):
        axis.text(value, position, f" {value:.2f}", va="center", fontsize=6.5, color=INK_2)
    axis.set_xlim(0, max(values) * 1.25 if values else 1)
    axis.set_xlabel("Mean GNNExplainer edge importance")
    axis.grid(axis="x", color=GRID_GRAY, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)
    axis.spines["left"].set_visible(False)
    return to_png(fig)


def rule_km_figure(rule_times: list[float], rule_survival: list[float], cohort_times: np.ndarray, cohort_survival: np.ndarray, label: str, risk_up: bool, width: float = 3.4) -> bytes:
    """Kaplan-Meier curve of the patients a rule covers (training fold) against the cohort."""
    fig, axis = plt.subplots(figsize=(width, 1.9))
    axis.step(cohort_times, cohort_survival, where="post", color=COHORT_LINE, linewidth=1.6, label="Cohort")
    axis.step(rule_times, rule_survival, where="post", color=RISK_UP if risk_up else RISK_DOWN, linewidth=2.0, label=label)
    axis.set_xlim(0, 180)
    axis.set_ylim(0, 1.02)
    axis.set_xticks([0, 36, 60, 120, 180])
    axis.set_xlabel("Months")
    axis.set_ylabel("PFS probability")
    _recessive(axis)
    axis.legend(frameon=False, fontsize=6.5, loc="lower left")
    return to_png(fig)


def risk_gauge_figure(percentile: float, width: float = 2.2) -> bytes:
    """A meter: the fill carries severity, the track is a light step of the same ramp."""
    fig, axis = plt.subplots(figsize=(width, 0.42))
    axis.add_patch(Rectangle((0, 0.35), 1, 0.3, color="#cde2fb", linewidth=0))
    color = RISK_UP if percentile >= 2 / 3 else "#eda100" if percentile >= 1 / 3 else RISK_DOWN
    axis.add_patch(Rectangle((0, 0.35), percentile, 0.3, color=color, linewidth=0))
    for cut in (1 / 3, 2 / 3):
        axis.plot([cut, cut], [0.3, 0.7], color=SURFACE, linewidth=1.5)
    axis.text(0, 0.0, "low", fontsize=6, color=INK_2, va="bottom")
    axis.text(0.5, 0.0, "intermediate", fontsize=6, color=INK_2, va="bottom", ha="center")
    axis.text(1, 0.0, "high", fontsize=6, color=INK_2, va="bottom", ha="right")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 0.75)
    axis.axis("off")
    return to_png(fig)


def group_km_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], width: float = 7.0) -> bytes:
    """Recorded PFS of each timeline group: one line per group, legend plus end labels."""
    fig, axis = plt.subplots(figsize=(width, 2.6))
    for code, (times, survival) in curves.items():
        axis.step(times, survival, where="post", color=GROUP_COLORS[code], linewidth=2.0, label=code)
    axis.set_xlim(0, 180)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Months from diagnosis")
    axis.set_ylabel("Progression-free probability")
    _recessive(axis)
    axis.legend(frameon=False, fontsize=7, ncol=len(curves), loc="lower left")
    return to_png(fig)


def predicted_vs_recorded_figure(recorded: np.ndarray, predicted: np.ndarray, low: np.ndarray, high: np.ndarray, width: float = 3.5) -> bytes:
    """Model-estimated month against the recorded month for progressors (single series)."""
    fig, axis = plt.subplots(figsize=(width, 2.6))
    order = np.argsort(recorded)
    axis.vlines(recorded[order], low[order], high[order], color="#b7d3f6", linewidth=1.2)
    axis.scatter(recorded, predicted, s=16, color="#2a78d6", edgecolor=SURFACE, linewidth=0.8, zorder=3)
    limit = max(np.nanmax(recorded), np.nanmax(high)) * 1.05
    axis.plot([0, limit], [0, limit], color=INK_3, linewidth=1)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Recorded progression (months)")
    axis.set_ylabel("Model estimate (months)")
    _recessive(axis)
    return to_png(fig)


def model_comparison_figure(names: list[str], means: list[float], stds: list[float], highlight: str, width: float = 7.0) -> bytes:
    """Fold-mean Harrell C per model, with the report model in the accent and the rest muted."""
    fig, axis = plt.subplots(figsize=(width, 0.32 * len(names) + 0.7))
    positions = np.arange(len(names))[::-1]
    colors = ["#2a78d6" if name == highlight else "#b9b8b2" for name in names]
    axis.barh(positions, means, xerr=stds, height=0.55, color=colors, error_kw={"ecolor": INK_3, "elinewidth": 1, "capsize": 2})
    axis.set_yticks(positions, names, fontsize=7.5, color=INK)
    for position, mean, name in zip(positions, means, names):
        axis.text(0.505, position, f"{mean:.3f}", va="center", fontsize=7, color=SURFACE if name == highlight else INK, fontweight="bold")
    axis.axvline(0.5, color=INK_3, linewidth=1)
    axis.set_xlim(0.5, max(np.array(means) + np.array(stds)) + 0.02)
    axis.set_xlabel("Harrell's C-index, 5-fold mean ± sd (out-of-fold)")
    axis.grid(axis="x", color=GRID_GRAY, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)
    axis.spines["left"].set_visible(False)
    return to_png(fig)
