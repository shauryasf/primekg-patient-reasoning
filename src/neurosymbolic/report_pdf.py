"""PDF patient reports, group summaries, and the cohort overview.

Everything a report prints is read from the pipeline's cached outputs under
``outputs/neurosymbolic``. Predictions, rules, and path scores are all
out-of-fold: they come from models that did not train on that patient.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from ..dataset_fusion import PrimeSlice
from . import report_figures as figures
from .paths import K_MAX, K_MIN, prominence_cutoff, rule_scores, score_paths, select_top_paths
from .timing import GROUP_NAMES, GROUPS, NO_PROGRESSION, window_label

PAGE_W, PAGE_H = A4
MARGIN = 14 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
NAVY = colors.HexColor("#0d366b")
INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
INK_3 = colors.HexColor("#8a8984")
RULE_GRAY = colors.HexColor("#e4e3df")
TILE = colors.HexColor("#f5f4f1")
RISK_UP = colors.HexColor("#c4302f")
RISK_DOWN = colors.HexColor("#1c5cab")
ISSUER = "PrimeKG Patient Reasoning · Precision Oncology Research"
DISCLAIMER = "Research prototype built on the public TCGA BRCA PanCancer Atlas cohort. Not a medical device and not for clinical decision-making."

STYLES = {
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=6, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=INK, spaceBefore=4, spaceAfter=2),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=8, leading=10.5, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=6.8, leading=8.6, textColor=INK_2),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.2, leading=8.8, textColor=INK),
    "cell_b": ParagraphStyle("cell_b", fontName="Helvetica-Bold", fontSize=7.2, leading=8.8, textColor=INK),
    "key": ParagraphStyle("key", fontName="Helvetica", fontSize=6.8, leading=8.6, textColor=INK_2),
    "tile_label": ParagraphStyle("tile_label", fontName="Helvetica", fontSize=6.8, leading=8, textColor=INK_2),
    "tile_value": ParagraphStyle("tile_value", fontName="Helvetica-Bold", fontSize=13, leading=15.5, textColor=INK),
    "tile_sub": ParagraphStyle("tile_sub", fontName="Helvetica", fontSize=6.6, leading=8.2, textColor=INK_2),
}


def _png(data: bytes, width: float) -> Image:
    image = Image(io.BytesIO(data))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return image


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ordinal(value: int) -> str:
    suffix = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _table(rows, widths, header: bool = True, zebra: bool = True, align_top: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP" if align_top else "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE_GRAY),
    ]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa"))]
    if zebra:
        for row in range(1 if header else 0, len(rows)):
            if row % 2 == 0:
                style.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#fafaf8")))
    table.setStyle(TableStyle(style))
    return table


def _tile(label: str, value: str, sub: str, accent: colors.Color | None = None, extra=None) -> Table:
    content = [[_p(label, "tile_label")], [_p(value, "tile_value")], [_p(sub, "tile_sub")]]
    if extra is not None:
        content.append([extra])
    tile = Table(content, colWidths=[CONTENT_W / 4 - 4])
    tile.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TILE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, accent or NAVY),
            ]
        )
    )
    return tile


def _badge(code: str) -> str:
    color = figures.GROUP_COLORS[code]
    return f'<font color="{color}"><b>■</b></font> <b>Group {code}</b> · {GROUP_NAMES[code]}'


@dataclass
class ReportContext:
    out_dir: Path
    patient_ids: list[str]
    predictions: pd.DataFrame
    curves: np.ndarray
    grid: np.ndarray
    record: pd.DataFrame
    outcomes: pd.DataFrame
    clinical: pd.DataFrame
    alterations: pd.DataFrame
    predicates: pd.DataFrame
    predicate_labels: dict[str, str]
    predicate_kinds: dict[str, str]
    rules: pd.DataFrame
    patient_rules: dict[str, list[str]]
    rule_km: dict[str, dict]
    scored_paths: pd.DataFrame
    cutoff: float
    gene_importance: dict[str, list[tuple[str, float]]]
    prime: PrimeSlice
    cohort_km: tuple[np.ndarray, np.ndarray]
    metrics: pd.DataFrame
    timing: dict
    report_model: str
    gnn_model: str
    logic_engine: str = "rulekit"
    logic_label: str = "RuleKit LR-Rules"
    logic_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    logic_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    atom_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    atom_binary: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    patient_index: dict[str, int] = field(default_factory=dict)
    similar: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    top_paths: dict[str, pd.DataFrame] = field(default_factory=dict)

    def paths_for(self, patient_id: str) -> pd.DataFrame:
        if patient_id not in self.top_paths:
            candidates = self.scored_paths[self.scored_paths["patient_id"] == patient_id]
            self.top_paths[patient_id] = select_top_paths(candidates, self.cutoff) if len(candidates) else pd.DataFrame()
        return self.top_paths[patient_id]


def kg_atom_genes(atom_frame: pd.DataFrame, prime: PrimeSlice) -> list[set[str]]:
    """Entrez ids of the genes behind each PrimeKG atom (its pathway members, drug targets, disease genes)."""
    drug_genes: dict[str, set[str]] = {}
    disease_genes: dict[str, set[str]] = {}
    for gene, drugs in prime.gene_drugs.items():
        for drug in drugs:
            drug_genes.setdefault(drug, set()).add(gene)
    for gene, diseases in prime.gene_diseases.items():
        for disease in diseases:
            disease_genes.setdefault(disease, set()).add(gene)
    lookup = {"pathway": prime.pathway_genes, "drug": drug_genes, "disease": disease_genes}
    genes = []
    for kind, members in zip(atom_frame["kind"], atom_frame["members"]):
        found: set[str] = set()
        if str(kind).startswith("kg_"):
            for member in str(members).split("|"):
                prefix, _, node = member.partition(":")
                found |= lookup.get(prefix, {}).get(node, set())
        genes.append(found)
    return genes


def _engine_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """One schema for every reasoner: ``atoms``, ``formula``, ``weight_per_sd``, ``fold``."""
    weights = weights.copy()
    if "weight_per_sd" not in weights.columns:
        weights["weight_per_sd"] = weights["net_weight"] if "net_weight" in weights.columns else weights["weight"]
    if "formula" not in weights.columns:
        weights["formula"] = weights["rule"]
    if "atoms" not in weights.columns:
        weights["atoms"] = weights["name"]
    return weights


def load_context(out_dir: Path) -> ReportContext:
    tables = out_dir / "tables"
    predictions = pd.read_csv(out_dir / "fusion" / "patient_predictions.csv").set_index("patient_id")
    patient_ids = list(predictions.index)
    meta = json.loads((tables / "predicate_meta.json").read_text(encoding="utf-8"))
    rules = pd.read_csv(out_dir / "symbolic" / "rules.csv").set_index("rule_id")
    coverage = pd.read_csv(out_dir / "symbolic" / "patient_rules.csv")
    candidates = pd.read_csv(out_dir / "explain" / "candidate_paths.csv")
    evidence = pd.read_csv(out_dir / "symbolic" / "kg_evidence.csv")
    engine = json.loads((out_dir / "logic" / "engine.json").read_text(encoding="utf-8"))
    atom_frame = pd.read_csv(out_dir / "logic" / "atoms.csv", keep_default_na=False)
    atom_binary = np.load(out_dir / "logic" / "atom_values.npy") > 0
    logic_weights = _engine_weights(pd.read_csv(out_dir / "logic" / f"{engine['engine']}_weights.csv"))
    logic_risk = pd.read_csv(out_dir / "logic" / "oof_risk.csv")
    patient_index = {pid: i for i, pid in enumerate(logic_risk["patient_id"])}
    predictions["fold"] = logic_risk.set_index("patient_id")["fold"].reindex(predictions.index).to_numpy()
    prime = PrimeSlice.from_frame(pd.read_csv(Path(out_dir).parents[1] / "data" / "primekg" / "primekg_panel_slice.csv", dtype=str, keep_default_na=False))
    rule_table = rule_scores(candidates, logic_weights, atom_frame, atom_binary, patient_index, kg_atom_genes(atom_frame, prime))
    scored = score_paths(candidates, evidence, rule_table)
    importance = pd.read_csv(out_dir / "explain" / "gene_importance.csv")
    gene_importance = {
        patient_id: list(zip(group.sort_values("importance", ascending=False)["gene"], group.sort_values("importance", ascending=False)["importance"]))
        for patient_id, group in importance.groupby("patient_id")
    }
    timed = predictions["time"].notna()
    kmf = KaplanMeierFitter().fit(predictions.loc[timed, "time"], predictions.loc[timed, "event"])
    curve = kmf.survival_function_.iloc[:, 0]
    timing = json.loads((out_dir / "fusion" / "timing_evaluation.json").read_text(encoding="utf-8"))
    predicates = pd.read_csv(tables / "predicates.csv", index_col=0)
    context = ReportContext(
        out_dir=out_dir,
        patient_ids=patient_ids,
        predictions=predictions,
        curves=np.load(out_dir / "fusion" / "report_curves.npy"),
        grid=np.arange(0.0, 241.0, 1.0),
        record=pd.read_csv(tables / "record.csv", index_col=0, dtype=str, keep_default_na=False),
        outcomes=pd.read_csv(tables / "outcomes.csv", index_col=0),
        clinical=pd.read_csv(tables / "clinical_extended.csv", index_col=0),
        alterations=pd.read_csv(tables / "alterations.csv", dtype={"entrez": str}, keep_default_na=False),
        predicates=predicates,
        predicate_labels=meta["labels"],
        predicate_kinds=meta["kinds"],
        rules=rules,
        patient_rules=coverage.groupby("patient_id")["rule_id"].agg(list).to_dict(),
        rule_km=json.loads((out_dir / "symbolic" / "rule_km.json").read_text(encoding="utf-8")),
        scored_paths=scored,
        cutoff=prominence_cutoff(scored),
        gene_importance=gene_importance,
        prime=prime,
        cohort_km=(curve.index.to_numpy(), curve.to_numpy()),
        metrics=pd.read_csv(out_dir / "fusion" / "metrics.csv"),
        timing=timing,
        report_model=timing["report_model"],
        gnn_model=timing["best_gnn"],
        logic_engine=engine["engine"],
        logic_label=engine["label"],
        logic_weights=logic_weights,
        logic_metrics=pd.read_csv(out_dir / "logic" / "metrics.csv"),
        atom_frame=atom_frame,
        atom_binary=atom_binary,
        patient_index=patient_index,
    )
    context.similar = _similar_patients(context)
    return context


def _similar_patients(context: ReportContext, k: int = 5) -> dict[str, list[tuple[str, float]]]:
    """Case-based neighbours: cosine similarity over clinical one-hots and molecular/KG predicates."""
    frame = context.predicates
    binary_cols = [column for column in frame.columns if context.predicate_kinds.get(column) != "clinical"]
    binary = (frame[binary_cols] == "yes").astype(float)
    clinical = pd.get_dummies(context.clinical[["t_stage", "n_stage", "m_stage", "ajcc_stage", "subtype", "histology"]].astype(str), dtype=float) * 2.0
    age = ((context.clinical[["age"]] - context.clinical["age"].mean()) / context.clinical["age"].std()).fillna(0)
    matrix = pd.concat([clinical, age, binary], axis=1).loc[context.patient_ids].to_numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0, 1, norms)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -1)
    result = {}
    for row, patient_id in enumerate(context.patient_ids):
        best = np.argsort(-similarity[row])[:k]
        result[patient_id] = [(context.patient_ids[index], float(similarity[row, index])) for index in best]
    return result


class _Doc(BaseDocTemplate):
    def __init__(self, path: Path, header_right: str, title: str, **kwargs):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=27 * mm, bottomMargin=15 * mm, title=title, author=ISSUER, **kwargs)
        self.header_right = header_right
        self.report_title = title
        frame = Frame(MARGIN, 15 * mm, CONTENT_W, PAGE_H - 42 * mm, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 21 * mm, PAGE_W, 21 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(MARGIN, PAGE_H - 10 * mm, self.report_title)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, PAGE_H - 15.5 * mm, ISSUER)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 10 * mm, self.header_right)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 15.5 * mm, f"Report date {date.today():%d %b %Y}")
        canvas.setStrokeColor(RULE_GRAY)
        canvas.setLineWidth(0.6)
        canvas.line(MARGIN, 11 * mm, PAGE_W - MARGIN, 11 * mm)
        canvas.setFillColor(INK_3)
        canvas.setFont("Helvetica", 6.3)
        canvas.drawString(MARGIN, 7.5 * mm, DISCLAIMER)
        canvas.drawRightString(PAGE_W - MARGIN, 7.5 * mm, f"Page {doc.page}")
        canvas.restoreState()


def _round(value: object) -> str:
    """Print long decimals from the clinical tables (TMB 4.266666667) at two decimals."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.2f}".rstrip("0").rstrip(".") if not number.is_integer() else f"{number:.0f}"


def _fmt_months(value: float) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.1f} mo"


def _record_section(context: ReportContext, patient_id: str) -> list:
    record = context.record.loc[patient_id]
    left = ["Age at diagnosis", "Sex", "Race", "Ethnicity", "Genetic ancestry", "Prior diagnosis", "Radiation therapy", "Neoadjuvant therapy"]
    right = ["Tumour type", "OncoTree code", "PAM50 subtype", "AJCC pathologic stage", "Pathologic T", "Pathologic N", "Pathologic M", "ICD-O-3 histology"]
    extra = ["Sample", "Aneuploidy score", "TMB (nonsyn., /Mb)", "MSIsensor score", "AJCC edition"]
    rows = []
    for index in range(max(len(left), len(right))):
        cells = []
        for column in (left, right):
            if index < len(column):
                key = column[index]
                value = record.get(key, "—")
                if key == "PAM50 subtype":
                    value = str(value).replace("BRCA_", "")
                cells += [_p(key, "key"), _p(_esc(value), "cell_b")]
            else:
                cells += ["", ""]
        rows.append(cells)
    rows.append([_p(" · ".join(f"{key}: <b>{_esc(_round(record.get(key, '—')))}</b>" for key in extra), "small"), "", "", ""])
    table = Table(rows, colWidths=[CONTENT_W * 0.17, CONTENT_W * 0.33, CONTENT_W * 0.17, CONTENT_W * 0.33])
    table.setStyle(
        TableStyle(
            [
                ("SPAN", (0, len(rows) - 1), (-1, len(rows) - 1)),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE_GRAY),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return [_p("1 · Patient and tumour record", "h1"), table]


def _assessment(context: ReportContext, patient_id: str) -> tuple[list, dict]:
    row = context.predictions.loc[patient_id]
    index = context.patient_ids.index(patient_id)
    percentile = float(row["risk_percentile"])
    tier = "High" if percentile >= 2 / 3 else "Intermediate" if percentile >= 1 / 3 else "Low"
    progressed = int(row["event"]) == 1
    recorded = float(row["time"]) if np.isfinite(row["time"]) else float("nan")
    window = str(row["predicted_window"]) if isinstance(row["predicted_window"], str) and row["predicted_window"] else ""
    month, low, high = float(row["predicted_month"]), float(row["predicted_low"]), float(row["predicted_high"])
    group = str(row["recorded_group"])
    gauge = _png(figures.risk_gauge_figure(percentile), CONTENT_W / 4 - 16)
    tiles = [
        _tile("Model risk (cohort percentile)", f"{_ordinal(int(round(percentile * 100)))}", f"{tier} risk tier · fused neuro-symbolic model", RISK_UP if tier == "High" else RISK_DOWN if tier == "Low" else colors.HexColor("#c98500"), gauge),
        _tile(
            "Probability of progression",
            f"{row['p_progress_36m'] * 100:.0f}% by 36 mo",
            f"12 mo {row['p_progress_12m'] * 100:.0f}% · 24 mo {row['p_progress_24m'] * 100:.0f}% · 60 mo {row['p_progress_60m'] * 100:.0f}%",
        ),
        _tile(
            "Model-estimated progression time",
            f"≈ {month:.0f} months" if np.isfinite(month) else "—",
            (f"Window {window_label(window)} · IQR {low:.0f}–{high:.0f} mo" if window else "") + "<br/>if progression occurs",
        ),
    ]
    if progressed:
        agree = window == group
        verdict = f'<font color="#1b7f3b"><b>✓ estimate in recorded window</b></font>' if agree else f'<font color="#c4302f"><b>✗ recorded window differs</b></font>'
        tiles.append(_tile("Recorded progression", f"{recorded:.1f} months", f"{_badge(group)}<br/>{verdict} · |error| {abs(month - recorded):.0f} mo", colors.HexColor(figures.GROUP_COLORS[group])))
    else:
        short = np.isfinite(recorded) and recorded < 24
        note = "Short follow-up: absence of progression is weakly informative." if short else "Censored at last follow-up; not proof of lifelong remission."
        tiles.append(_tile("Recorded progression", "None recorded", f"Followed {_fmt_months(recorded)} · {_badge('N')}<br/>{note}", colors.HexColor(figures.GROUP_COLORS["N"])))
    tile_row = Table([tiles], colWidths=[CONTENT_W / 4] * 4)
    tile_row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    figure = _png(
        figures.survival_figure(
            context.grid,
            context.curves[index],
            *context.cohort_km,
            recorded,
            progressed,
            (month, low, high) if np.isfinite(month) else None,
        ),
        CONTENT_W,
    )
    facts = {"percentile": percentile, "tier": tier, "month": month, "window": window, "progressed": progressed, "recorded": recorded, "group": group}
    caption = _p(
        "Blue: this patient's progression-free survival curve from the fused model, fitted on the four folds that exclude the patient. "
        "Grey: Kaplan–Meier curve of the whole cohort. Shaded bands mark the report groups A–C; the bar on the time axis is the model's "
        "interquartile range for the month of progression, given that progression occurs.",
        "small",
    )
    return [_p("2 · Progression assessment", "h1"), tile_row, Spacer(1, 4), figure, caption], facts


def _formulas_for(context: ReportContext, patient_id: str) -> pd.DataFrame:
    """Weighted formulas of the logic engine that fire for this patient, from the fold that held them out."""
    if context.logic_weights.empty or patient_id not in context.patient_index:
        return pd.DataFrame()
    fold = int(context.predictions.loc[patient_id, "fold"])
    weights = context.logic_weights[context.logic_weights["fold"] == fold]
    position = {name: index for index, name in enumerate(context.atom_frame["atom"])}
    truth = context.atom_binary[context.patient_index[patient_id]]
    fired = []
    for item in weights.to_dict("records"):
        atoms = [position[name] for name in str(item["atoms"]).split("|") if name in position]
        if atoms and all(truth[a] for a in atoms) and abs(float(item["weight_per_sd"])) > 1e-6:
            item["atom_index"] = atoms
            fired.append(item)
    frame = pd.DataFrame(fired)
    if frame.empty:
        return frame
    return frame.sort_values("weight_per_sd", key=np.abs, ascending=False)


def _formula_km(context: ReportContext, patient_id: str, atoms: list[int]):
    """Kaplan-Meier curve of the training-fold patients that satisfy a formula."""
    fold = int(context.predictions.loc[patient_id, "fold"])
    frame = context.predictions.loc[list(context.patient_index)]
    covered = np.all(context.atom_binary[:, atoms], axis=1) & (frame["fold"].to_numpy() != fold) & frame["time"].notna().to_numpy()
    if covered.sum() < 5:
        return None
    kmf = KaplanMeierFitter().fit(frame["time"].to_numpy()[covered], frame["event"].to_numpy()[covered])
    curve = kmf.survival_function_.iloc[:, 0]
    return curve.index.tolist(), curve.tolist(), int(covered.sum())


def _path_text(path: dict) -> str:
    nodes = [node for node in json.loads(path["nodes"]) if node["type"] != "sample"]
    relations = json.loads(path["relations"])[1:]
    parts = [f"<b>{_esc(nodes[0]['name'])}</b>"]
    for relation, node in zip(relations, nodes[1:]):
        parts.append(f"<font color='#8a8984'>{figures.RELATION_TEXT.get(relation, relation)}</font> → <b>{_esc(node['name'])}</b>")
    return " ".join(parts)


def _reasoning_section(context: ReportContext, patient_id: str) -> tuple[list, dict]:
    story: list = [_p("3 · Neuro-symbolic reasoning", "h1")]
    paths = context.paths_for(patient_id)
    candidates = int((context.scored_paths["patient_id"] == patient_id).sum())
    facts: dict = {"paths": paths}
    story.append(_p("3a · Top knowledge-graph paths (GNN-scored, PrimeKG-grounded)", "h2"))
    if paths.empty:
        story.append(
            _p(
                "No explanation path is available: none of this patient's altered panel genes has a stored PrimeKG pathway, drug, or breast-disease link "
                "(or the tumour carries no qualifying alteration). The risk estimate therefore rests on the clinical block and the symbolic rules below.",
                "body",
            )
        )
    else:
        prominent = int(paths["prominent"].sum())
        if prominent >= len(paths):
            note = f"{len(paths)} of {candidates} candidate paths shown; all clear the prominence cutoff (NS ≥ {context.cutoff:.2f}, the cohort's 80th percentile), capped at K = {K_MAX}."
        else:
            lead = "None" if prominent == 0 else f"Only {prominent}"
            note = (
                f"{lead} of the {candidates} candidate paths clear{'s' if prominent == 1 else ''} the prominence cutoff (NS ≥ {context.cutoff:.2f}, the cohort's 80th percentile); "
                f"the best remaining paths are shown so that {min(K_MIN, len(paths))} appear, and they are not marked ★."
            )
        story = [KeepTogether(story + [_png(figures.path_figure(paths.to_dict("records")), CONTENT_W)])]
        story.append(_p(note + f" NS score = 0.6 × neural percentile (GNNExplainer importance from the held-out fold model) + 0.4 × rule strength (summed weight of the {_esc(context.logic_label)} formulas that fire for this patient and mention the path's gene or end node).", "small"))
        rows = [[_p("#", "cell_b"), _p("Path", "cell_b"), _p("Neural pct", "cell_b"), _p("Rule evidence (learned logic formulas)", "cell_b"), _p("NS", "cell_b")]]
        for path in paths.to_dict("records"):
            score = float(path.get("rule_score", 0.0) or 0.0)
            if int(path.get("rule_count", 0) or 0) > 0:
                arrow = "↑" if score > 0 else "↓"
                color = "#c4302f" if score > 0 else "#1c5cab"
                evidence = (
                    f"<font color='{color}'><b>{arrow} {path['evidence_direction']}</b></font> · Σw {score:+.3f} over {int(path['rule_count'])} formula(s)"
                    f"<br/><font size='6.4'>{_esc(path['rule_formula'])}</font>"
                )
            else:
                evidence = "No learned formula fires on this path; its score is neural only."
            if np.isfinite(path.get("fact_hazard_ratio", np.nan)):
                evidence += f"<br/><font color='#8a8984' size='6.4'>cohort: HR {path['fact_hazard_ratio']:.2f}, n = {int(path['fact_carriers'])} carriers</font>"
            rows.append(
                [
                    _p(f"{path['rank']}{' ★' if path['prominent'] else ''}", "cell_b"),
                    _p(_path_text(path), "cell"),
                    _p(f"{path['neural'] * 100:.0f}", "cell"),
                    _p(evidence, "cell"),
                    _p(f"<b>{path['ns_score']:.2f}</b>", "cell"),
                ]
            )
        story.append(_table(rows, [CONTENT_W * w for w in (0.05, 0.43, 0.08, 0.36, 0.08)]))

    formulas = _formulas_for(context, patient_id)
    facts["formulas"] = formulas
    block: list = [_p(f"3b · Weighted logic formulas that fire for this patient ({_esc(context.logic_label)})", "h2")]
    if formulas.empty:
        block.append(_p("No learned formula with a non-zero weight fires for this patient; the symbolic risk is the baseline of the logic model.", "body"))
        story.append(KeepTogether(block))
        return story, facts
    source_text = {"rl": "RL-proposed", "mined": "mined pair", "unary": "single atom"}
    rows = [[_p("Weight", "cell_b"), _p("Formula (first-order, grounded on this patient)", "cell_b"), _p("Source", "cell_b"), _p("Support", "cell_b")]]
    for item in formulas.head(6).to_dict("records"):
        weight = float(item["weight_per_sd"])
        up = weight > 0
        pill = f"<font color='{'#c4302f' if up else '#1c5cab'}'><b>{weight:+.3f}</b><br/>{'↑ risk' if up else '↓ risk'}</font>"
        kg = " <font color='#1baf7a'><b>[KG]</b></font>" if bool(item.get("uses_kg")) else ""
        source = str(item.get("source", "") or "")
        rows.append(
            [
                _p(pill, "cell"),
                _p(_esc(item["formula"]) + kg, "cell"),
                _p(source_text.get(source, source or "—"), "cell"),
                _p(f"{int(item.get('support', 0))} pts", "cell"),
            ]
        )
    table = _table(rows, [CONTENT_W * w for w in (0.1, 0.44, 0.1, 0.08)])
    top = formulas.iloc[0]
    km = _formula_km(context, patient_id, list(top["atom_index"]))
    side = _png(figures.rule_km_figure(km[0], km[1], *context.cohort_km, f"top formula holds (n={km[2]})", float(top["weight_per_sd"]) > 0), CONTENT_W * 0.27) if km else ""
    layout = Table([[table, side]], colWidths=[CONTENT_W * 0.72, CONTENT_W * 0.28])
    layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    block.append(layout)
    block.append(
        _p(
            "Each formula reads Body(p) ⇒ Progress(p) and carries a weight learned with the Cox partial likelihood on the four folds that exclude this patient "
            "(per standard deviation of its grounding count; positive raises the progression hazard). Multi-atom formulas were proposed by a reinforcement-learning "
            "rule generator rewarded for replicated log-rank separation, or mined as pairs. [KG] marks a formula with a PrimeKG relation such as "
            "Altered(p, g) ∧ InPathway(g, w). The curve shows the training patients for whom the top formula holds.",
            "small",
        )
    )
    story.append(KeepTogether(block))
    return story, facts


def _molecular_section(context: ReportContext, patient_id: str) -> list:
    heading = _p("4 · Molecular profile and knowledge-graph context", "h1")
    story: list = [heading]
    alterations = context.alterations[context.alterations["patient_id"] == patient_id]
    importance = dict(context.gene_importance.get(patient_id, []))
    if alterations.empty:
        story.append(_p("No mutation or high-level CNA on the 500-gene panel was recorded for this tumour.", "body"))
    else:
        grouped = (
            alterations.groupby(["gene", "kind"])
            .agg(detail=("detail", lambda values: ", ".join(sorted({str(v) for v in values if v})[:3])), driver=("driver", "max"), entrez=("entrez", "first"))
            .reset_index()
        )
        grouped["importance"] = grouped["gene"].map(importance).fillna(0.0)
        grouped["kg_links"] = [
            len(context.prime.gene_pathways.get(str(e), ())) + len(context.prime.gene_drugs.get(str(e), ())) + len(context.prime.gene_diseases.get(str(e), ()))
            for e in grouped["entrez"]
        ]
        grouped = grouped.sort_values(["driver", "importance"], ascending=[False, False])
        n_total = len(grouped)
        shown = grouped.head(12)
        kind_text = {"mut": "Mutation", "amp": "High-level amplification", "del": "Homozygous deletion"}
        rows = [[_p("Gene", "cell_b"), _p("Alteration", "cell_b"), _p("Detail", "cell_b"), _p("PrimeKG links", "cell_b"), _p("GNN imp.", "cell_b")]]
        for item in shown.itertuples(index=False):
            rows.append(
                [
                    _p(f"<b>{_esc(item.gene)}</b>{' · driver' if item.driver else ''}", "cell"),
                    _p(kind_text[item.kind], "cell"),
                    _p(_esc(item.detail if item.kind == "mut" else "GISTIC ±2"), "cell"),
                    _p(str(item.kg_links), "cell"),
                    _p(f"{item.importance:.2f}" if item.importance else "—", "cell"),
                ]
            )
        table = _table(rows, [CONTENT_W * w for w in (0.15, 0.18, 0.14, 0.08, 0.07)])
        genes = [(gene, value, alterations[alterations["gene"] == gene]["kind"].iloc[0]) for gene, value in context.gene_importance.get(patient_id, [])]
        side = _png(figures.gene_bar_figure(genes), CONTENT_W * 0.36) if genes else _p("No gene importance available.", "small")
        layout = Table([[table, side]], colWidths=[CONTENT_W * 0.63, CONTENT_W * 0.37])
        layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        story[-1] = KeepTogether([heading, layout])
        story.append(_p(f"{n_total} altered panel genes in total; known drivers first, then by GNN importance. CNA follows the graph policy: only homozygous deletion (−2) and high-level amplification (+2).", "small"))
        drugs = []
        for item in grouped.itertuples(index=False):
            for drug in sorted(context.prime.gene_drugs.get(str(item.entrez), ())):
                drugs.append((item.gene, context.prime.drug_names.get(drug, drug), drug))
        if drugs:
            text = "; ".join(f"{_esc(gene)} → {_esc(name)} ({drug})" for gene, name, drug in drugs[:10])
            more = f" … and {len(drugs) - 10} more" if len(drugs) > 10 else ""
            story.append(_p("PrimeKG drug–protein links of the altered genes", "h2"))
            story.append(_p(text + more + ". These are knowledge-graph links (target, enzyme, carrier, or transporter), shown for research context only; they are not treatment recommendations.", "small"))

    neighbours = context.similar.get(patient_id, [])
    if neighbours:
        rows = [[_p("Most similar patients", "cell_b"), _p("Age", "cell_b"), _p("Stage", "cell_b"), _p("Subtype", "cell_b"), _p("Similarity", "cell_b"), _p("Recorded outcome", "cell_b")]]
        for other, score in neighbours:
            clinical = context.clinical.loc[other]
            outcome = context.predictions.loc[other]
            text = f"progressed at {outcome['time']:.1f} mo (group {outcome['recorded_group']})" if int(outcome["event"]) == 1 else f"no progression, followed {_fmt_months(outcome['time'])}"
            rows.append([_p(other, "cell"), _p(f"{clinical['age']:.0f}", "cell"), _p(str(clinical["ajcc_stage"]), "cell"), _p(str(clinical["subtype"]), "cell"), _p(f"{score:.2f}", "cell"), _p(text, "cell")])
        story.append(_p("Case-based context", "h2"))
        story.append(_table(rows, [CONTENT_W * w for w in (0.2, 0.08, 0.09, 0.1, 0.1, 0.43)]))
        story.append(_p("Cosine similarity over stage, subtype, histology, age, driver alterations, and PrimeKG pathway facts. Outcomes are recorded follow-up, shown as context, not as a prediction.", "small"))
    story.append(_p("How to read this report", "h2"))
    story.append(
        _p(
            f"The risk and the survival curve come from a Cox model on the clinical block (age, sex, PAM50 subtype, AJCC stage, M stage) plus the GNN risk and the logic-engine risk ({context.logic_label}), "
            "all fitted on the four folds that exclude this patient. In cross-validation the GNN and logic terms carry close to zero weight once the clinical block is present, "
            "so the risk mostly reflects clinical stage, nodal status, age, and subtype. The paths show which PrimeKG facts the graph model attended to, and the weighted "
            "logic formulas state which facts, alone or combined, shift the progression hazard in the training folds; they are evidence, not causal claims. "
            f"The progression-time estimate ({context.timing['chosen_estimator'].replace('_', ' ')}) is within 12 months of the recorded month for "
            f"{context.timing['chosen']['within_12_months'] * 100:.0f}% of held-out progressors and in the recorded window for {context.timing['chosen']['window_accuracy'] * 100:.0f}%; "
            "the recorded month is always shown beside it.",
            "small",
        )
    )
    return story


def _summary_paragraph(context: ReportContext, patient_id: str, assessment: dict, reasoning: dict) -> Paragraph:
    clinical = context.clinical.loc[patient_id]
    parts = [
        f"This {clinical['age']:.0f}-year-old patient has a {str(clinical['subtype']).replace('missing', 'subtype-unassigned')} tumour, "
        f"AJCC stage {clinical['ajcc_stage']} ({clinical['t_stage']}, {clinical['n_stage']}, {clinical['m_stage']}). "
        f"The fused neuro-symbolic model places them in the <b>{_ordinal(int(round(assessment['percentile'] * 100)))} risk percentile</b> ({assessment['tier'].lower()} tier)."
    ]
    formulas = reasoning.get("formulas")
    if formulas is not None and not formulas.empty:
        top = formulas.iloc[0]
        direction = "raises" if float(top["weight_per_sd"]) > 0 else "lowers"
        parts.append(f" The strongest learned formula that holds for them (<i>{_esc(top['formula'])}</i>) {direction} the progression hazard (weight {float(top['weight_per_sd']):+.3f} per SD).")
    paths = reasoning.get("paths")
    if paths is not None and not paths.empty:
        best = paths.iloc[0]
        evidence = ""
        if int(best.get("rule_count", 0) or 0) > 0:
            evidence = f"; the learned formulas it instantiates point to {best['evidence_direction']} (Σw {float(best['rule_score']):+.3f})"
        parts.append(f" The highest-scoring knowledge-graph path runs from <b>{_esc(best['altered_gene'])}</b> to <b>{_esc(best['terminal_name'])}</b>{evidence}.")
    if assessment["progressed"] and np.isfinite(assessment["month"]):
        parts.append(f" Progression was recorded at {assessment['recorded']:.1f} months; the model's estimate given progression was {assessment['month']:.0f} months.")
    elif not assessment["progressed"]:
        parts.append(f" No progression was recorded during {_fmt_months(assessment['recorded'])} of follow-up.")
    return _p("".join(parts), "body")


def build_patient_report(context: ReportContext, patient_id: str, path: Path) -> Path:
    group = str(context.predictions.loc[patient_id, "recorded_group"])
    doc = _Doc(path, f"Patient {patient_id} · Group {group}", "Neuro-Symbolic Progression Report")
    story: list = [_p(_badge(group) + f" &nbsp;&nbsp;<font color='#8a8984'>Breast invasive carcinoma · TCGA PanCancer Atlas 2018 · model: {context.report_model.replace('_', ' ')}</font>", "body"), Spacer(1, 2)]
    story += _record_section(context, patient_id)
    assessment, facts = _assessment(context, patient_id)
    story += assessment
    reasoning, reasoning_facts = _reasoning_section(context, patient_id)
    story.insert(len(story), _p("Summary", "h2"))
    story.append(_summary_paragraph(context, patient_id, facts, reasoning_facts))
    story += reasoning
    story += _molecular_section(context, patient_id)
    doc.build(story)
    return path


def _group_patients(context: ReportContext, code: str) -> pd.DataFrame:
    frame = context.predictions[context.predictions["recorded_group"] == code].copy()
    return frame.sort_values("time" if code != NO_PROGRESSION[0] else "risk_percentile", ascending=code != NO_PROGRESSION[0])


def build_group_summary(context: ReportContext, code: str, path: Path, report_files: dict[str, str]) -> Path:
    members = _group_patients(context, code)
    doc = _Doc(path, f"Group {code} · {len(members)} patients", f"Group {code}: {GROUP_NAMES[code]}")
    story: list = []
    range_text = window_label(code) if code != NO_PROGRESSION[0] else "censored, no progression recorded"
    story.append(_p(f"{_badge(code)} &nbsp; <font color='#52514e'>recorded progression-free survival: {range_text}</font>", "body"))
    progressed = code != NO_PROGRESSION[0]
    tiles = [
        _tile("Patients", f"{len(members)}", f"{len(members) / len(context.predictions) * 100:.1f}% of the cohort"),
        _tile("Median model risk percentile", f"{members['risk_percentile'].median() * 100:.0f}", "cohort-wide risk ranking, out-of-fold"),
    ]
    if progressed:
        hit = (members["predicted_window"] == code).mean()
        error = (members["predicted_month"] - members["time"]).abs().median()
        tiles.append(_tile("Recorded progression", f"{members['time'].median():.1f} mo", f"median; range {members['time'].min():.1f}–{members['time'].max():.1f} mo"))
        tiles.append(_tile("Model window agreement", f"{hit * 100:.0f}%", f"estimate in this window · median |error| {error:.0f} mo"))
    else:
        short = (members["time"] < 24).mean()
        tiles.append(_tile("Median follow-up", f"{members['time'].median():.1f} mo", f"{short * 100:.0f}% followed for under 24 months"))
        tiles.append(_tile("Predicted P(progression by 60 mo)", f"{members['p_progress_60m'].median() * 100:.0f}%", "median over the group"))
    row = Table([tiles], colWidths=[CONTENT_W / 4] * 4)
    row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story += [Spacer(1, 4), row, Spacer(1, 6)]
    if progressed and len(members) > 2:
        scatter = _png(figures.predicted_vs_recorded_figure(members["time"].to_numpy(), members["predicted_month"].to_numpy(), members["predicted_low"].to_numpy(), members["predicted_high"].to_numpy()), CONTENT_W * 0.48)
    else:
        scatter = _p("", "small")
    ends = []
    for patient_id in members.index:
        selected = context.paths_for(patient_id)
        if not selected.empty:
            ends.extend(selected["terminal_name"].tolist())
    common = pd.Series(ends).value_counts().head(8) if ends else pd.Series(dtype=int)
    rows = [[_p("Most frequent path end-points in this group", "cell_b"), _p("Patients", "cell_b")]]
    rows += [[_p(_esc(name), "cell"), _p(str(count), "cell")] for name, count in common.items()]
    endpoints = _table(rows, [CONTENT_W * 0.38, CONTENT_W * 0.1])
    layout = Table([[scatter, endpoints]], colWidths=[CONTENT_W * 0.5, CONTENT_W * 0.5])
    layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(layout)
    fired = []
    for patient_id in members.index:
        formulas = _formulas_for(context, patient_id)
        if not formulas.empty:
            fired.extend((row["formula"], float(row["weight_per_sd"])) for row in formulas.head(3).to_dict("records"))
    if fired:
        frame = pd.DataFrame(fired, columns=["formula", "weight"])
        top = frame.groupby("formula").agg(patients=("weight", "size"), weight=("weight", "mean")).sort_values("patients", ascending=False).head(6)
        rows = [[_p("Logic formulas firing most often in this group", "cell_b"), _p("Mean weight", "cell_b"), _p("Patients", "cell_b")]]
        for formula, item in top.iterrows():
            rows.append([_p(_esc(formula), "cell"), _p(f"{item['weight']:+.3f}", "cell"), _p(str(int(item["patients"])), "cell")])
        story += [Spacer(1, 6), _table(rows, [CONTENT_W * 0.72, CONTENT_W * 0.13, CONTENT_W * 0.15])]
    drivers = context.alterations[context.alterations["patient_id"].isin(members.index) & context.alterations["driver"].astype(bool)]
    if not drivers.empty:
        counts = drivers.groupby(["gene", "kind"])["patient_id"].nunique().sort_values(ascending=False).head(10)
        text = ", ".join(f"{gene} {figures.RELATION_TEXT['has_' + {'mut': 'mutation', 'amp': 'high_level_amplification', 'del': 'homozygous_deletion'}[kind]]} ({count / len(members) * 100:.0f}%)" for (gene, kind), count in counts.items())
        story += [_p("Most common driver alterations", "h2"), _p(text, "body")]
    story.append(PageBreak())
    story.append(_p("Patient index", "h1"))
    rows = [[_p(h, "cell_b") for h in ("Patient", "Age", "Stage", "Subtype", "Recorded", "Model estimate", "Window", "Risk pct", "Report file")]]
    for patient_id, item in members.iterrows():
        clinical = context.clinical.loc[patient_id]
        recorded = f"{item['time']:.1f} mo" + ("" if int(item["event"]) == 1 else " (cens.)")
        estimate = f"{item['predicted_month']:.0f} mo" if np.isfinite(item["predicted_month"]) else "—"
        mark = ("✓" if item["predicted_window"] == code else "✗") if progressed else str(item["predicted_window"])
        rows.append(
            [
                _p(patient_id, "cell"),
                _p(f"{clinical['age']:.0f}", "cell"),
                _p(str(clinical["ajcc_stage"]), "cell"),
                _p(str(clinical["subtype"]), "cell"),
                _p(recorded, "cell"),
                _p(estimate, "cell"),
                _p(mark, "cell"),
                _p(f"{item['risk_percentile'] * 100:.0f}", "cell"),
                _p(report_files.get(patient_id, ""), "small"),
            ]
        )
    story.append(_table(rows, [CONTENT_W * w for w in (0.14, 0.05, 0.06, 0.08, 0.11, 0.1, 0.07, 0.07, 0.32)]))
    doc.build(story)
    return path


def build_cohort_overview(context: ReportContext, path: Path) -> Path:
    doc = _Doc(path, f"{len(context.predictions)} patients", "Cohort Overview: Neuro-Symbolic Progression Model")
    metrics = context.metrics.sort_values("harrell_c_fold_mean", ascending=False)
    story: list = [_p("Model comparison (all predictions out-of-fold, same five folds)", "h1")]
    story.append(_png(figures.model_comparison_figure(metrics["model"].str.replace("_", " ").tolist(), metrics["harrell_c_fold_mean"].tolist(), metrics["harrell_c_fold_std"].tolist(), context.report_model.replace("_", " ")), CONTENT_W))
    columns = ["harrell_c", "uno_c_120", "td_auc_12", "td_auc_36", "td_auc_60", "integrated_brier_120"]
    rows = [[_p(h, "cell_b") for h in ("Model", "Harrell C", "Uno C (120 mo)", "AUC 12 mo", "AUC 36 mo", "AUC 60 mo", "IBS 120 mo")]]
    for item in metrics.itertuples(index=False):
        values = [getattr(item, column, float("nan")) for column in columns]
        rows.append([_p(item.model.replace("_", " "), "cell_b" if item.model == context.report_model else "cell")] + [_p("—" if not np.isfinite(v) else f"{v:.3f}", "cell") for v in values])
    story.append(_table(rows, [CONTENT_W * w for w in (0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12)]))
    story.append(_p("Lower integrated Brier score (IBS) is better; it is reported only for models that produce a survival curve.", "small"))
    if not context.logic_metrics.empty:
        story.append(_p("Symbolic reasoners compared", "h1"))
        rows = [[_p(h, "cell_b") for h in ("Reasoner", "Harrell C (mean ± sd)", "Uno C", "Formulas / fold", "KG formulas")]]
        for item in context.logic_metrics.sort_values("c_mean", ascending=False).itertuples(index=False):
            style = "cell_b" if item.method == context.logic_engine else "cell"
            rows.append(
                [
                    _p(_esc(item.label) + (" (engine)" if item.method == context.logic_engine else ""), style),
                    _p(f"{item.c_mean:.3f} ± {item.c_std:.3f}", "cell"),
                    _p(f"{item.uno_c_120:.3f}", "cell"),
                    _p(f"{item.rules_per_fold:.1f}", "cell"),
                    _p(f"{item.kg_rule_share * 100:.0f}%", "cell"),
                ]
            )
        story.append(_table(rows, [CONTENT_W * w for w in (0.4, 0.18, 0.12, 0.15, 0.15)]))
        story.append(_p("The engine is chosen with the one-standard-error rule: among reasoners within one standard error of the best fold-mean C-index, the one with the richest reasoning is preferred (RL-structured MLN, then MLN, PSL, RL rules, RuleKit).", "small"))
    story.append(_p("Predicting the month of progression", "h1"))
    rows = [[_p(h, "cell_b") for h in ("Estimator (held-out progressors)", "MAE", "Median |error|", "Within 12 mo", "Window accuracy", "IQR coverage")]]
    for name, row in context.timing["estimators"].items():
        bold = "cell_b" if name == context.timing["chosen_estimator"] else "cell"
        rows.append([_p(name.replace("_", " "), bold)] + [_p(f"{row[k]:.1f}" if k.endswith("error_months") or k == "mae_months" else f"{row[k] * 100:.0f}%", "cell") for k in ("mae_months", "median_abs_error_months", "within_12_months", "window_accuracy")] + [_p(f"{row.get('interval_coverage', float('nan')) * 100:.0f}%", "cell")])
    story.append(_table(rows, [CONTENT_W * w for w in (0.36, 0.11, 0.14, 0.13, 0.14, 0.12)]))
    story.append(_p("The bold estimator is the report's “model-estimated progression time”. The training-progressor median is a constant baseline that ignores the patient.", "small"))
    progressors = context.predictions[context.predictions["event"] == 1]
    curves = {}
    for code, _, _, _ in GROUPS:
        members = context.predictions[context.predictions["recorded_group"] == code]
        if len(members):
            kmf = KaplanMeierFitter().fit(members["time"], members["event"])
            curves[code] = (kmf.survival_function_.index.to_numpy(), kmf.survival_function_.iloc[:, 0].to_numpy())
    members = context.predictions[(context.predictions["recorded_group"] == "N") & context.predictions["time"].notna()]
    kmf = KaplanMeierFitter().fit(members["time"], members["event"])
    curves["N"] = (kmf.survival_function_.index.to_numpy(), kmf.survival_function_.iloc[:, 0].to_numpy())
    story.append(_p("Report groups", "h1"))
    rows = [[_p(h, "cell_b") for h in ("Group", "Recorded PFS", "Patients", "Median model risk pct", "Model window agreement")]]
    for code in [g[0] for g in GROUPS] + ["N"]:
        members = context.predictions[context.predictions["recorded_group"] == code]
        agreement = f"{(members['predicted_window'] == code).mean() * 100:.0f}%" if code != "N" else "—"
        rows.append([_p(_badge(code), "cell"), _p(window_label(code) if code != "N" else "censored", "cell"), _p(str(len(members)), "cell"), _p(f"{members['risk_percentile'].median() * 100:.0f}", "cell"), _p(agreement, "cell")])
    story.append(_table(rows, [CONTENT_W * w for w in (0.36, 0.16, 0.1, 0.18, 0.2)]))
    scatter = _png(figures.predicted_vs_recorded_figure(progressors["time"].to_numpy(), progressors["predicted_month"].to_numpy(), progressors["predicted_low"].to_numpy(), progressors["predicted_high"].to_numpy(), width=4.2), CONTENT_W * 0.55)
    story.append(
        KeepTogether(
            [
                _p("Model-estimated against recorded progression month (held-out progressors)", "h1"),
                scatter,
                _p("Each dot is one of the 145 progressors; the vertical bar is the model's interquartile range and the diagonal is perfect agreement. "
                   "Estimates are compressed towards 15–45 months: early progressions are over-estimated and progressions after five years are under-estimated.", "small"),
            ]
        )
    )
    doc.build(story)
    return path
