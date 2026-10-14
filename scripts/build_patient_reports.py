#!/usr/bin/env python
"""Write the PDF patient reports, binned by recorded progression timeline.

Reads the cached outputs of ``scripts/run_neurosymbolic_pipeline.py`` and writes::

    outputs/patient_reports/
      00_cohort_overview.pdf
      A_progression_within_12m/   00_group_A_summary.pdf, group_A_all_reports.pdf, <patient>.pdf ...
      B_progression_12-24m/       ...
      C_progression_24-36m/       ...
      D_progression_after_36m/    ...
      N_no_progression/           00_group_N_summary.pdf, <patient>.pdf ...

Group N has no combined bundle: at roughly 940 reports it would be too large to
open comfortably, so its summary carries the index instead.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time as clock
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypdf import PdfWriter  # noqa: E402

from src.neurosymbolic.report_pdf import build_cohort_overview, build_group_summary, build_patient_report, load_context  # noqa: E402

LOGGER = logging.getLogger("reports")
GROUP_DIRS = {
    "A": "A_progression_within_12m",
    "B": "B_progression_12-24m",
    "C": "C_progression_24-36m",
    "D": "D_progression_after_36m",
    "N": "N_no_progression",
}
BUNDLED = ("A", "B", "C", "D")
_CONTEXT = None


def _init(out_dir: str) -> None:
    global _CONTEXT
    import matplotlib

    matplotlib.use("Agg")
    _CONTEXT = load_context(Path(out_dir))


def _one(patient_id: str, path: str) -> str:
    build_patient_report(_CONTEXT, patient_id, Path(path))
    return patient_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-dir", type=Path, default=ROOT / "outputs" / "neurosymbolic")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "patient_reports")
    parser.add_argument("--patients", default=None, help="comma-separated patient ids; default is every patient")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    context = load_context(args.pipeline_dir)
    groups = context.predictions["recorded_group"].to_dict()
    patients = [item.strip() for item in args.patients.split(",")] if args.patients else list(context.patient_ids)
    targets = {}
    for patient_id in patients:
        folder = args.output_dir / GROUP_DIRS[groups[patient_id]]
        folder.mkdir(parents=True, exist_ok=True)
        targets[patient_id] = folder / f"{patient_id}.pdf"
    started = clock.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(args.pipeline_dir),)) as pool:
        futures = [pool.submit(_one, patient_id, str(path)) for patient_id, path in targets.items()]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done % 50 == 0 or done == len(futures):
                LOGGER.info("%d / %d reports (%.1f min)", done, len(futures), (clock.time() - started) / 60)
    if args.patients:
        return 0
    files = {patient_id: f"{GROUP_DIRS[groups[patient_id]]}/{patient_id}.pdf" for patient_id in context.patient_ids}
    for code, folder_name in GROUP_DIRS.items():
        folder = args.output_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        summary = build_group_summary(context, code, folder / f"00_group_{code}_summary.pdf", files)
        if code in BUNDLED:
            writer = PdfWriter()
            writer.append(str(summary))
            members = context.predictions[context.predictions["recorded_group"] == code].sort_values("time").index
            for patient_id in members:
                writer.append(str(folder / f"{patient_id}.pdf"))
            with (folder / f"group_{code}_all_reports.pdf").open("wb") as handle:
                writer.write(handle)
        LOGGER.info("Group %s summary written", code)
    build_cohort_overview(context, args.output_dir / "00_cohort_overview.pdf")
    LOGGER.info("Cohort overview written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
