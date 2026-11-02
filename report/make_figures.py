#!/usr/bin/env python
"""Build every figure the LaTeX report includes, into report/figures/.

New figures are drawn from the cached outputs of the neuro-symbolic pipeline;
figures from earlier milestones are copied from outputs/. Run from the
repository root after scripts/run_neurosymbolic_pipeline.py and
scripts/build_patient_reports.py.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.neurosymbolic import report_figures as F  # noqa: E402
from src.neurosymbolic.report_pdf import load_context  # noqa: E402

OUT = ROOT / "outputs" / "neurosymbolic"
REPORTS = ROOT / "outputs" / "patient_reports"
FIGURES = Path(__file__).resolve().parent / "figures"
EXAMPLE = ("A_progression_within_12m", "TCGA-BH-A1EV")


def save(data: bytes, name: str) -> None:
    (FIGURES / name).write_bytes(data)


def pipeline_figure() -> None:
    """Block diagram of the whole system, left to right."""
    fig, axis = plt.subplots(figsize=(11.5, 4.6))
    axis.set_xlim(0, 124)
    axis.set_ylim(0, 46)
    axis.axis("off")
    boxes = {
        "data": (1, 30, 18, 12, "#cde2fb", "TCGA BRCA\nPanCancer Atlas\n1,083 patients"),
        "kg": (1, 6, 18, 12, "#d4f0e4", "PrimeKG\n8.1M edges\n(4,570-edge slice)"),
        "graph": (29, 30, 20, 12, "#fde4d8", "Patient graph\npatient-sample-gene\n500-gene panel"),
        "fused": (29, 6, 20, 12, "#fde4d8", "Fused graph\n+ pathway, drug,\nbreast-disease nodes"),
        "neural": (61, 30, 18, 12, "#cde2fb", "Neural\nsurvival GNN\n(Cox / C-index)"),
        "symbolic": (61, 6, 18, 12, "#d4f0e4", "Symbolic logic\nMLN · PSL · RL\n(RL-structured MLN)"),
        "fusion": (90, 6, 14, 36, "#fdf0d0", "Fused Cox\n\nrisk,\nsurvival curve,\nprogression\nmonth"),
        "report": (109, 18, 14, 12, "#ece8f8", "Patient\nreport\n(groups A-N)"),
    }
    for x, y, w, h, color, text in boxes.values():
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", facecolor=color, edgecolor="#52514e", linewidth=0.8))
        axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#0b0b0b", linespacing=1.25)

    def arrow(a, b, label="", y=None):
        xa, ya, wa, ha = boxes[a][:4]
        xb = boxes[b][0]
        level = y if y is not None else ya + ha / 2
        start, end = (xa + wa + 0.8, level), (xb - 0.8, level)
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#52514e", linewidth=1.1))
        if label:
            axis.text((start[0] + end[0]) / 2, level + 1.0, label, ha="center", va="bottom", fontsize=7.2, color="#52514e")

    arrow("data", "graph", "MAF + CNA")
    arrow("kg", "fused", "Entrez join")
    axis.add_patch(FancyArrowPatch((39, 29.2), (39, 18.8), arrowstyle="-|>", mutation_scale=12, color="#52514e", linewidth=1.1))
    axis.text(40, 24, "graft PrimeKG\nneighbourhood", fontsize=7.2, color="#52514e", va="center")
    axis.add_patch(FancyArrowPatch((49.8, 14), (60.2, 34), arrowstyle="-|>", mutation_scale=12, color="#52514e", linewidth=1.1))
    arrow("fused", "symbolic", "ground atoms", y=10)
    arrow("neural", "fusion", "GNN risk")
    arrow("symbolic", "fusion", "rule risk")
    arrow("fusion", "report")
    axis.add_patch(FancyArrowPatch((70, 29.2), (70, 18.8), arrowstyle="<|-|>", mutation_scale=10, color="#8a8984", linewidth=0.9, linestyle="--"))
    axis.text(71, 24, "path score =\n0.6 neural +\n0.4 rule", fontsize=7.2, color="#52514e", va="center")
    fig.savefig(FIGURES / "fig_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pfs_histogram(pred: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 2.6))
    months = pred.loc[pred.event == 1, "time"]
    axis.hist(months, bins=np.arange(0, 200, 6), color="#2a78d6", rwidth=0.85)
    for edge, code in ((12, "A | B"), (24, "B | C"), (36, "C | D")):
        axis.axvline(edge, color="#52514e", linewidth=1)
        axis.text(edge + 1, axis.get_ylim()[1] * 0.88, code, fontsize=7.5, color="#52514e")
    axis.set_xlabel("Recorded progression (months)")
    axis.set_ylabel("Patients")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e4e3df")
    axis.set_axisbelow(True)
    fig.savefig(FIGURES / "fig_pfs_hist.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ns_histogram(ctx) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 2.4))
    axis.hist(ctx.scored_paths.ns_score, bins=60, color="#1c5cab", rwidth=0.9)
    axis.axvline(ctx.cutoff, color="#e34948", linewidth=1.5)
    axis.text(ctx.cutoff + 0.005, axis.get_ylim()[1] * 0.88, f"prominence cutoff {ctx.cutoff:.2f}", fontsize=8)
    axis.set_xlabel("Neuro-symbolic path score")
    axis.set_ylabel("Candidate paths")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e4e3df")
    axis.set_axisbelow(True)
    fig.savefig(FIGURES / "fig_ns_hist.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_page(pdf: Path, page: int, name: str, zoom: float = 2.0) -> None:
    doc = pymupdf.open(pdf)
    doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(FIGURES / name)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    copies = {
        "fig_fused_graph.png": ROOT / "outputs" / "gnn_benchmark_36m" / "fused_graph_example.png",
        "fig_horizon.png": ROOT / "outputs" / "gnn_benchmark_36m" / "horizon_sweep.png",
        "fig_36m_roc.png": ROOT / "outputs" / "gnn_benchmark_36m" / "graph_models_roc.png",
        "fig_gsl.png": ROOT / "outputs" / "gnn_benchmark_gsl" / "gsl_prior_comparison_bars.png",
        "fig_patient_graph.png": ROOT / "outputs" / "visualizations" / "TCGA-A2-A0SW.png",
    }
    for name, source in copies.items():
        shutil.copyfile(source, FIGURES / name)
    pipeline_figure()
    ctx = load_context(OUT)
    pred = ctx.predictions.reset_index()
    pfs_histogram(pred)
    ns_histogram(ctx)
    metrics = ctx.metrics.sort_values("harrell_c_fold_mean", ascending=False)
    save(F.model_comparison_figure(metrics.model.str.replace("_", " ").tolist(), metrics.harrell_c_fold_mean.tolist(), metrics.harrell_c_fold_std.tolist(), ctx.report_model.replace("_", " ")), "fig_model_comparison.png")
    progressors = pred[pred.event == 1]
    save(F.predicted_vs_recorded_figure(progressors.time.values, progressors.predicted_month.values, progressors.predicted_low.values, progressors.predicted_high.values, width=4.6), "fig_timing.png")
    paths = ctx.paths_for(EXAMPLE[1])
    save(F.path_figure(paths.to_dict("records")), "fig_paths_example.png")
    logic = pd.read_csv(OUT / "logic" / "metrics.csv").sort_values("c_mean", ascending=False)
    save(F.model_comparison_figure(logic["label"].tolist(), logic["c_mean"].tolist(), logic["c_std"].tolist(), ctx.logic_label), "fig_logic_comparison.png")
    weights = pd.read_csv(OUT / "logic" / f"{ctx.logic_engine}_weights.csv")
    proposed = weights[(weights["source"] == "rl") & (weights["support"] >= 40)].sort_values("weight_per_sd", key=np.abs, ascending=False)
    top = proposed.iloc[0]
    columns = [ctx.atom_frame["atom"].tolist().index(name) for name in str(top["atoms"]).split("|")]
    covered = np.all(ctx.atom_binary[:, columns], axis=1)
    frame = pred.set_index("patient_id").loc[list(ctx.patient_index)]
    usable = covered & frame["time"].notna().to_numpy()
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter().fit(frame["time"].to_numpy()[usable], frame["event"].to_numpy()[usable])
    curve = kmf.survival_function_.iloc[:, 0]
    save(F.rule_km_figure(curve.index.tolist(), curve.tolist(), *ctx.cohort_km, f"RL-proposed formula holds (n={int(usable.sum())})", float(top["weight_per_sd"]) > 0, width=4.6), "fig_rule_km.png")
    (FIGURES / "fig_rule_km.txt").write_text(f"{top['formula']}\nweight {top['weight_per_sd']:+.3f} fold {int(top['fold'])}\n", encoding="utf-8")
    pdf = REPORTS / EXAMPLE[0] / f"{EXAMPLE[1]}.pdf"
    for page in range(3):
        render_page(pdf, page, f"fig_report_p{page + 1}.png")
    render_page(REPORTS / EXAMPLE[0] / "00_group_A_summary.pdf", 0, "fig_group_summary.png")
    render_page(REPORTS / "N_no_progression" / "TCGA-BH-A18I.pdf", 0, "fig_report_n_p1.png")
    print(f"Wrote {len(list(FIGURES.glob('*.png')))} figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
