"""Plot the hop-prior GSL run against the fixed 2-hop graph and the first GSL run.

The fixed 2-hop metrics and the first GSL metrics are read from their own run
directories. The hop-prior run directory is read when it is present; otherwise
the transcribed 5-fold means and standard deviations in ``RECORDED_PRIOR`` are
used. That run directory is in no commit of this repository, so the hop-prior
column of the figure cannot be re-derived here. The per-depth weights of the
first GSL run are recomputed from ``outputs/gnn_benchmark_gsl/gsl_edges.csv``;
the hop-prior bars are that run's design starting values, not measurements.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIXED_RUN = OUTPUTS / "gnn_benchmark_36m"
FIRST_GSL_RUN = OUTPUTS / "gnn_benchmark_gsl"
PRIOR_RUN = OUTPUTS / "gnn_benchmark_gsl_prior"
EDGE_REPORT = FIRST_GSL_RUN / "gsl_edges.csv"
OUTPUT_PATH = FIRST_GSL_RUN / "gsl_prior_comparison_bars.png"

MODELS = ("gcn", "graphsage", "gatv2")
MODEL_LABELS = ("GCN", "GraphSAGE", "GATv2")
SETTINGS = (
    ("Fixed 2-hop graph", "#4C78A8"),
    ("First GSL, free weights", "#F58518"),
    ("GSL + hop prior", "#54A24B"),
)
DEPTHS = (0, 1, 2, 3)
DEPTH_LABELS = ("hop 0\nobserved", "hop 1", "hop 2", "hop 3")

# Transcribed 5-fold result of the hop-prior GSL run, as that run's own
# outputs/gnn_benchmark_gsl_prior/metrics_summary.csv reports it for pfs_36m:
# AUROC mean, AUROC std, AUPRC mean, AUPRC std. No commit or branch of this
# repository contains that directory, so these numbers are a transcription and
# neither this script nor the notebook can recompute them.
RECORDED_PRIOR = {
    "gcn": (0.7014, 0.0793, 0.4558, 0.1286),
    "graphsage": (0.7032, 0.0804, 0.4627, 0.1326),
    "gatv2": (0.7029, 0.0810, 0.4628, 0.1330),
}
# Design starting values, not measurements. Hops 0, 1, and 2 sit inside the
# fixed prior, where edge_factor = 2 * sigmoid(0) = 1 at initialisation; every
# one of those edges can still move afterwards through its own residual, so 1.0
# is only where they start. Hop 3 sits behind the hop gate, which has to pass
# 0.5 to contribute and starts at sigmoid(-2); that gate never opened, so about
# 0.12 is the ceiling on a hop-3 weight in that run. The hop-prior run's
# per-edge report was not kept, so none of these four bars is measured.
PRIOR_DEPTH_WEIGHTS = (1.0, 1.0, 1.0, 0.12)
OPEN_THRESHOLD = 0.5
# 89 of the 498 patients with a known 36-month window progressed by then.
CHANCE_AUPRC = 89 / 498


def read_summary(run_dir: Path) -> dict[str, tuple[float, float, float, float]]:
    """Read AUROC and AUPRC means and standard deviations for the graph models."""
    frame = pd.read_csv(run_dir / "metrics_summary.csv").set_index("model")
    return {
        model: (
            float(frame.loc[model, "auroc_mean"]),
            float(frame.loc[model, "auroc_std"]),
            float(frame.loc[model, "auprc_mean"]),
            float(frame.loc[model, "auprc_std"]),
        )
        for model in MODELS
    }


def load_runs() -> tuple[dict, dict, dict]:
    """Load the fixed 2-hop run, the first GSL run, and the hop-prior run."""
    fixed = read_summary(FIXED_RUN)
    first_gsl = read_summary(FIRST_GSL_RUN)
    if (PRIOR_RUN / "metrics_summary.csv").exists():
        prior = read_summary(PRIOR_RUN)
    else:
        print(f"{PRIOR_RUN} is not present; using the transcribed values in RECORDED_PRIOR.")
        prior = RECORDED_PRIOR
    return fixed, first_gsl, prior


def depth_weights(path: Path) -> tuple[float, ...]:
    """Mean learned weight of the stored edges at each depth of the first GSL run."""
    frame = pd.read_csv(path)
    return tuple(float(frame.loc[frame["hop"] == hop, "score"].mean()) for hop in DEPTHS)


def build_figure(fixed: dict, first_gsl: dict, prior: dict, depths: tuple[float, ...]) -> Path:
    """Draw AUROC, AUPRC, and the weight each depth was given."""
    runs = (fixed, first_gsl, prior)
    positions = np.arange(len(MODELS), dtype=float)
    width = 0.26
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    for axis, column, name in ((axes[0], 0, "AUROC"), (axes[1], 2, "AUPRC")):
        for offset, (label, colour), run in zip((-width, 0.0, width), SETTINGS, runs):
            means = [run[model][column] for model in MODELS]
            stds = [run[model][column + 1] for model in MODELS]
            axis.bar(positions + offset, means, width, yerr=stds, capsize=3, color=colour, label=label)
        chance = 0.5 if column == 0 else CHANCE_AUPRC
        chance_label = "chance 0.50" if column == 0 else "chance 0.179 (89/498)"
        axis.axhline(chance, color="black", linestyle=":", linewidth=1.2, label=chance_label)
        axis.set_xticks(positions, MODEL_LABELS)
        axis.set_ylim(0, 1)
        axis.set_ylabel("5-fold mean ± std")
        axis.set_title(f"{name}, 36-month progression")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=9, loc="upper left")

    hops = np.arange(len(DEPTHS), dtype=float)
    axes[2].bar(hops - 0.19, depths, 0.38, color=SETTINGS[1][1], label="First GSL, free weights (measured)")
    axes[2].bar(hops + 0.19, PRIOR_DEPTH_WEIGHTS, 0.38, color=SETTINGS[2][1], label="GSL + hop prior (design start)")
    axes[2].axhline(OPEN_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label="open threshold 0.50")
    axes[2].set_xticks(hops, DEPTH_LABELS)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Mean weight (measured / design start)")
    axes[2].set_title("Weight per PrimeKG depth (prior bars: design start)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle("Patient-conditioned GSL with a hop prior versus the fixed 2-hop graph", fontsize=14)
    fig.text(
        0.005,
        0.005,
        "Hop-prior hops 0-2: design start, not measured (that run's per-edge report was not kept).",
        ha="left",
        fontsize=8,
        color="#555555",
    )
    fig.text(
        0.995,
        0.005,
        "outputs/gnn_benchmark_gsl_prior/ is in no commit, so the hop-prior AUROC/AUPRC are transcribed, not re-derived.",
        ha="right",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH



def main() -> None:
    fixed, first_gsl, prior = load_runs()
    depths = depth_weights(EDGE_REPORT)
    print("model      fixed 2-hop   first GSL   hop prior   (AUROC/AUPRC; hop prior transcribed)")
    for index, model in enumerate(MODELS):
        print(
            f"{MODEL_LABELS[index]:<10}"
            f"{fixed[model][0]:.3f}/{fixed[model][2]:.3f}    "
            f"{first_gsl[model][0]:.3f}/{first_gsl[model][2]:.3f}    "
            f"{prior[model][0]:.3f}/{prior[model][2]:.3f}"
        )
    print("first GSL mean weight by depth:", [round(value, 3) for value in depths])
    print("hop-prior weight by depth (design start):", PRIOR_DEPTH_WEIGHTS)
    print("wrote", build_figure(fixed, first_gsl, prior, depths))


if __name__ == "__main__":
    main()
