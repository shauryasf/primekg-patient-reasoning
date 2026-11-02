"""Ground atoms shared by the three reasoners.

Every atom is a closed formula about one patient ``p``. Its value is the number
of true groundings, which is what a Markov logic network counts:

- clinical atoms such as ``Stage(p, III)`` or ``AgeAtLeast(p, 65)`` (0 or 1);
- genomic atoms such as ``Mutated(p, TP53)`` or ``Amplified(p, MYC)`` (0 or 1);
- PrimeKG atoms that are relational: ``Altered(p, g) AND InPathway(g, w)`` counts
  the altered genes of ``p`` that PrimeKG places in pathway ``w``, and likewise
  ``TargetedBy(g, d)`` for drugs and ``LinkedTo(g, d)`` for breast diseases.

Atom selection uses prevalence only, never an outcome, so the table can be built
once for the whole cohort. Knowledge-graph atoms that hold for exactly the same
patients are merged, and the merged atom keeps every node it stands for, so a
report path ending at any of those nodes can be matched to the atom's weight.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...dataset_fusion import PrimeSlice

MIN_SHARE = 0.03
MAX_SHARE = 0.90
CLINICAL_LEVELS = {
    "t_stage": ("T1", "T2", "T3", "T4"),
    "n_stage": ("N0", "N1", "N2", "N3"),
    "m_stage": ("M1", "MX"),
    "ajcc_stage": ("I", "II", "III", "IV"),
    "subtype": ("LumA", "LumB", "Basal", "Her2", "Normal"),
    "histology": ("lobular", "other"),
    "radiation_therapy": ("no",),
}
CLINICAL_NAMES = {
    "t_stage": "TStage",
    "n_stage": "NStage",
    "m_stage": "MStage",
    "ajcc_stage": "Stage",
    "subtype": "Subtype",
    "histology": "Histology",
    "radiation_therapy": "Radiation",
}


@dataclass
class AtomTable:
    """``values[p, a]`` is the number of true groundings of atom ``a`` for patient ``p``."""

    patient_ids: list[str]
    values: np.ndarray
    names: list[str]
    labels: list[str]
    kinds: list[str]
    formulas: list[str]
    variables: list[str]
    members: list[set[str]] = field(default_factory=list)

    @property
    def binary(self) -> np.ndarray:
        return self.values > 0

    def index(self, name: str) -> int:
        return self.names.index(name)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "atom": self.names,
                "label": self.labels,
                "kind": self.kinds,
                "formula": self.formulas,
                "variable": self.variables,
                "members": ["|".join(sorted(m)) for m in self.members],
                "carriers": self.binary.sum(axis=0),
            }
        )


def _add(columns, name, label, kind, formula, variable, members, values):
    columns.append((name, label, kind, formula, variable, set(members), np.asarray(values, dtype=np.float32)))


def build_atoms(
    clinical: pd.DataFrame,
    predicates: pd.DataFrame,
    predicate_labels: dict[str, str],
    alterations: pd.DataFrame,
    prime: PrimeSlice,
) -> AtomTable:
    patient_ids = list(clinical.index)
    n = len(patient_ids)
    columns: list = []

    for column, levels in CLINICAL_LEVELS.items():
        for level in levels:
            values = (clinical[column].astype(str) == level).to_numpy()
            _add(columns, f"{column}={level}", f"{column.replace('_', ' ').replace('ajcc', 'AJCC')} = {level}", "clinical",
                 f"{CLINICAL_NAMES[column]}(p, {level})", column, [], values)
    age = clinical["age"].to_numpy(dtype=float)
    for name, label, mask in (
        ("age<40", "age < 40", age < 40),
        ("age>=65", "age ≥ 65", age >= 65),
    ):
        _add(columns, name, label, "clinical", f"Age(p) {label[4:]}", "age", [], np.nan_to_num(mask, nan=0))
    for column, label in (("aneuploidy_score", "aneuploidy"), ("tmb_nonsynonymous", "TMB")):
        values = clinical[column].to_numpy(dtype=float)
        high, low = np.nanquantile(values, 0.75), np.nanquantile(values, 0.25)
        _add(columns, f"{column}_high", f"high {label} (top quarter)", "clinical", f"High{label.title().replace(' ', '')}(p)", column, [], np.nan_to_num(values >= high))
        _add(columns, f"{column}_low", f"low {label} (bottom quarter)", "clinical", f"Low{label.title().replace(' ', '')}(p)", column, [], np.nan_to_num(values <= low))

    verbs = {"mut": "Mutated", "amp": "Amplified", "del": "Deleted"}
    for column in predicates.columns:
        if not column.startswith(("mut_", "amp_", "del_")):
            continue
        kind, gene = column.split("_", 1)
        values = (predicates[column] == "yes").to_numpy()
        _add(columns, column, predicate_labels.get(column, column), "genomic", f"{verbs[kind]}(p, {gene})", column, [gene], values)

    altered = alterations.groupby("patient_id")["entrez"].agg(lambda items: set(map(str, items))).to_dict()
    kg_groups = (
        ("pathway", prime.pathway_genes, prime.pathway_names, "InPathway", "altered gene in pathway"),
        ("disease", _invert(prime.gene_diseases), prime.disease_names, "LinkedTo", "altered gene linked to"),
        ("drug", _invert(prime.gene_drugs), prime.drug_names, "TargetedBy", "altered gene targeted by"),
    )
    kg_columns: dict[bytes, int] = {}
    for prefix, members_of, names, relation, phrase in kg_groups:
        # Sorted, so atom columns (and every index-based learner) do not depend on hash order.
        for node, genes in sorted(members_of.items()):
            counts = np.array([len(altered.get(pid, set()) & genes) for pid in patient_ids], dtype=np.float32)
            share = float((counts > 0).mean())
            if not (MIN_SHARE <= share <= MAX_SHARE):
                continue
            key = (counts > 0).tobytes()
            node_id = f"{prefix}:{node}"
            title = names.get(node, node)
            if key in kg_columns:
                columns[kg_columns[key]][5].add(node_id)
                continue
            kg_columns[key] = len(columns)
            _add(columns, f"kg_{prefix}:{node}", f"{phrase} '{title}'", f"kg_{prefix}",
                 f"Altered(p, g) ∧ {relation}(g, '{title}')", f"kg_{prefix}:{node}", [node_id], counts)

    kept = [c for c in columns if MIN_SHARE <= float((c[6] > 0).mean()) <= MAX_SHARE]
    return AtomTable(
        patient_ids=patient_ids,
        values=np.vstack([c[6] for c in kept]).T if kept else np.zeros((n, 0), dtype=np.float32),
        names=[c[0] for c in kept],
        labels=[c[1] for c in kept],
        kinds=[c[2] for c in kept],
        formulas=[c[3] for c in kept],
        variables=[c[4] for c in kept],
        members=[c[5] for c in kept],
    )


def _invert(gene_to_nodes: dict[str, set[str]]) -> dict[str, set[str]]:
    nodes: dict[str, set[str]] = {}
    for gene, items in sorted(gene_to_nodes.items()):
        for item in sorted(items):
            nodes.setdefault(item, set()).add(gene)
    return nodes


def soft_targets(time: np.ndarray, event: np.ndarray, train: np.ndarray, horizon: float = 36.0) -> np.ndarray:
    """Probability of progression by ``horizon``, with censoring handled by Kaplan-Meier.

    Progression at or before the horizon is 1 and event-free follow-up past it is 0.
    A patient censored at ``c < horizon`` gets ``(S(c) - S(horizon)) / S(c)`` from the
    training-fold Kaplan-Meier curve: the chance they progress before the horizon
    given that they were event-free at ``c``. These are the soft truth values PSL and
    the reports' calibration use; the survival losses read time and event directly.
    """
    from lifelines import KaplanMeierFitter

    timed = train[np.isfinite(time[train])]
    kmf = KaplanMeierFitter().fit(time[timed], event[timed])
    survival = lambda t: float(kmf.survival_function_at_times(t).iloc[0])  # noqa: E731
    at_horizon = survival(horizon)
    targets = np.zeros(len(time), dtype=np.float32)
    for index, (t, e) in enumerate(zip(time, event)):
        if not np.isfinite(t):
            targets[index] = 1.0 - at_horizon
        elif e == 1 and t <= horizon:
            targets[index] = 1.0
        elif t >= horizon:
            targets[index] = 0.0
        else:
            start = survival(t)
            targets[index] = 0.0 if start <= 0 else (start - at_horizon) / start
    return targets
