"""Survival rules learned with RuleKit (LR-Rules) over readable predicates.

RuleKit grows each rule with the log-rank test as its quality measure, so a rule
body picks out a group of patients whose progression-free survival differs from
everyone else's. Each rule therefore carries its own Kaplan-Meier curve. A
prediction for a new patient is the average curve of the rules that cover them.

Binary predicates enter RuleKit as "yes" or missing, never as "no". A rule can
then only say "PIK3CA mutated" or "altered gene in pathway P"; it cannot be
built from the absence of a rare passenger mutation, which is what the raw
yes/no coding produced.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from .features import NUMERIC_CLINICAL, PredicateTable

TIME_COLUMN = "survival_time"
LABEL_NAME = "survival_status"
RULE_PARAMS = {"minsupp_new": 0.05, "max_growing": 3, "ignore_missing": False}
INTERVAL = re.compile(r"^([<(])\s*([-\w.]+)\s*,\s*([-\w.]+)\s*([)>])$")
_STARTED = False


def start_rulekit() -> None:
    """Start the JVM once per process."""
    global _STARTED
    if not _STARTED:
        from rulekit import RuleKit

        RuleKit.init()
        _STARTED = True


def rulekit_frame(table: PredicateTable) -> pd.DataFrame:
    """Predicates in RuleKit form: binary "no" and clinical "missing" become missing values."""
    frame = table.frame.copy()
    for column in frame.columns:
        if column in NUMERIC_CLINICAL or column == "n_altered_genes":
            frame[column] = frame[column].astype(float)
            continue
        values = frame[column].astype(object)
        if table.kinds[column] == "clinical":
            frame[column] = values.where(values != "missing", np.nan)
        else:
            frame[column] = values.where(values == "yes", np.nan)
    return frame


@dataclass
class Condition:
    column: str
    raw: str
    text: str


@dataclass
class LearnedRule:
    rule_id: str
    fold: int
    raw: str
    conditions: list[Condition]
    coverage: int
    events: int
    pvalue: float
    hazard_ratio: float
    median_months: float
    km_times: list[float] = field(default_factory=list)
    km_survival: list[float] = field(default_factory=list)
    members: set[str] = field(default_factory=set)

    @property
    def text(self) -> str:
        return " AND ".join(condition.text for condition in self.conditions)

    @property
    def direction(self) -> str:
        return "higher risk" if self.hazard_ratio > 1 else "lower risk"

    def as_row(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "fold": self.fold,
            "rule": self.text,
            "raw": self.raw,
            "coverage": self.coverage,
            "events": self.events,
            "pvalue": self.pvalue,
            "hazard_ratio": self.hazard_ratio,
            "median_months": self.median_months,
            "direction": self.direction,
            "members": "|".join(sorted(self.members)),
            "n_kg_conditions": sum(condition.column.startswith("kg_") for condition in self.conditions),
        }


def _number(text: str) -> float:
    return float("inf") if text == "inf" else float("-inf") if text == "-inf" else float(text)


def readable_condition(column: str, value: str, table: PredicateTable) -> str:
    """Turn ``age = <41.50, inf)`` into ``Age ≥ 41.5`` and ``kg_x = {yes}`` into its label."""
    label = table.labels.get(column, column)
    value = value.strip()
    if value.startswith("{"):
        level = value.strip("{}")
        if table.kinds.get(column) != "clinical" and level == "yes":
            return label
        return f"{label} = {level}"
    match = INTERVAL.match(value)
    if not match:
        return f"{label} {value}"
    low, high = _number(match.group(2)), _number(match.group(3))
    digits = 1 if column != "tmb_nonsynonymous" else 2
    if np.isinf(low):
        return f"{label} < {high:.{digits}f}"
    if np.isinf(high):
        return f"{label} ≥ {low:.{digits}f}"
    return f"{low:.{digits}f} ≤ {label} < {high:.{digits}f}"


def parse_rule(raw: str, table: PredicateTable) -> list[Condition]:
    body = raw.strip()
    body = re.sub(r"^IF\s+", "", body)
    body = re.sub(r"\s+THEN.*$", "", body)
    conditions = []
    for part in body.split(" AND "):
        if " = " not in part:
            continue
        column, value = part.split(" = ", 1)
        column = column.strip()
        conditions.append(Condition(column=column, raw=part.strip(), text=readable_condition(column, value, table)))
    return conditions


def _rule_statistics(covered: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[float, float, list[float], list[float]]:
    """Hazard ratio of covered vs. the rest, and the covered group's Kaplan-Meier curve."""
    kmf = KaplanMeierFitter().fit(time[covered], event[covered])
    median = float(kmf.median_survival_time_)
    curve = kmf.survival_function_.iloc[:, 0]
    hazard_ratio = float("nan")
    if 0 < covered.sum() < len(covered):
        frame = pd.DataFrame({"T": time, "E": event, "covered": covered.astype(float)})
        try:
            fitter = CoxPHFitter(penalizer=1e-3).fit(frame, duration_col="T", event_col="E")
            hazard_ratio = float(np.exp(fitter.params_["covered"]))
        except Exception:  # noqa: BLE001 - a degenerate split keeps nan
            pass
    return hazard_ratio, median, curve.index.tolist(), curve.tolist()


def coverage_matrix(model, frame: pd.DataFrame) -> np.ndarray:
    """Boolean ``[patients, rules]`` matrix of which rules cover which patient."""
    matrix = np.asarray(model.get_coverage_matrix(frame))
    return matrix.astype(bool)


def fit_rules(
    frame: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    table: PredicateTable,
    fold: int,
    params: dict | None = None,
):
    """Fit RuleKit on one training fold and describe every rule on that fold's data."""
    from rulekit.survival import SurvivalRules

    start_rulekit()
    training = frame.copy()
    training[TIME_COLUMN] = time
    model = SurvivalRules(survival_time_attr=TIME_COLUMN, **(params or RULE_PARAMS))
    model.fit(training, pd.Series(event.astype(int).astype(str), name=LABEL_NAME))
    covered = coverage_matrix(model, training)
    rules: list[LearnedRule] = []
    for index, rule in enumerate(model.model.rules):
        mask = covered[:, index]
        hazard_ratio, median, km_times, km_survival = _rule_statistics(mask, time, event)
        conditions = parse_rule(str(rule), table)
        members: set[str] = set()
        for condition in conditions:
            members |= table.members.get(condition.column, set())
        rules.append(
            LearnedRule(
                rule_id=f"F{fold + 1}-R{index + 1}",
                fold=fold,
                raw=str(rule),
                conditions=conditions,
                coverage=int(mask.sum()),
                events=int(event[mask].sum()),
                pvalue=float(rule.pvalue),
                hazard_ratio=hazard_ratio,
                median_months=median,
                km_times=km_times,
                km_survival=km_survival,
                members=members,
            )
        )
    return model, rules


def predict_curves(model, frame: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """RuleKit's per-patient survival curve, read off as a step function on ``grid``."""
    scoring = frame.copy()
    scoring[TIME_COLUMN] = 0.0
    curves = np.ones((len(scoring), len(grid)))
    for row, estimate in enumerate(model.predict(scoring)):
        times = np.asarray(estimate["times"], dtype=float)
        probabilities = np.asarray(estimate["probabilities"], dtype=float)
        positions = np.searchsorted(times, grid, side="right") - 1
        curves[row] = np.where(positions >= 0, probabilities[np.clip(positions, 0, None)], 1.0)
    return curves


def restricted_mean(curves: np.ndarray, grid: np.ndarray, tau: float = 120.0) -> np.ndarray:
    """Restricted mean survival time up to ``tau`` months from step curves on ``grid``."""
    keep = grid <= tau
    widths = np.diff(np.r_[grid[keep], tau])
    return (curves[:, keep] * widths).sum(axis=1)
