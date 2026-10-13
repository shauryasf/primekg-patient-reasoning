#!/usr/bin/env python
"""Compare finished benchmark runs on one endpoint, as markdown tables.

Reads ``metrics_summary.csv`` from each run directory and prints AUROC, AUPRC, and
the C-index per model, so a quoted table can be regenerated before it is believed.
Nothing is written to disk.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    ("bce", ROOT / "outputs" / "gnn_benchmark_36m"),
    ("cindex", ROOT / "outputs" / "gnn_benchmark_36m_cindex"),
    ("cox", ROOT / "outputs" / "gnn_benchmark_36m_cox"),
)
METRICS = ("auroc", "auprc", "c_index")


def _cell(row: pd.Series | None, metric: str, available: bool) -> str:
    if row is None:
        return "-"
    if not available:
        # A run that predates the metric cannot report it. That is not the same
        # as a metric that was computable and came out undefined.
        return "not reported"
    mean, std = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
    if mean is None or not pd.notna(mean):
        return "undefined"
    return f"{mean:.3f} ± {std:.3f}"


def _table(frames: dict[str, pd.DataFrame], metric: str, runs: list[str]) -> list[str]:
    models: list[str] = []
    for run in runs:
        for model in frames[run]["model"].tolist():
            if pd.notna(model) and model not in models:
                models.append(str(model))
    lines = [f"| Model | {' | '.join(runs)} |", "| --- |" + " --- |" * len(runs)]
    for model in models:
        cells = []
        for run in runs:
            frame = frames[run]
            row = frame[frame["model"] == model]
            cells.append(_cell(row.iloc[0] if len(row) else None, metric, f"{metric}_mean" in frame.columns))
        lines.append(f"| {model} | {' | '.join(cells)} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="pfs_36m")
    parser.add_argument("--metric", choices=METRICS, default="auprc")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="name=directory pairs; the default lists the committed 36-month runs",
    )
    args = parser.parse_args()
    runs = list(DEFAULT_RUNS)
    if args.runs:
        runs = []
        for item in args.runs:
            name, _, directory = item.partition("=")
            if not directory:
                parser.error(f"--runs entries look like name=directory; received {item!r}")
            runs.append((name, Path(directory)))

    frames: dict[str, pd.DataFrame] = {}
    missing = []
    for name, path in runs:
        summary_path = path / "metrics_summary.csv"
        if not summary_path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(summary_path)
        frames[name] = frame[frame["endpoint"] == args.endpoint].copy()
    if not frames:
        print(f"No run reported endpoint {args.endpoint}. Missing: {', '.join(missing) or 'nothing matched'}")
        return 1
    names = list(frames)
    print(f"Endpoint {args.endpoint}. '-' means the model was not in that run; 'not reported' means that run predates the metric.")
    print()
    for metric in METRICS:
        print(f"### {metric}")
        print()
        print("\n".join(_table(frames, metric, names)))
        print()
    if missing:
        print("Missing run directories: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
