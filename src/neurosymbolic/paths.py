"""Neuro-symbolic path scoring and top-K selection for the patient reports.

Every candidate path is a real walk over stored edges of the patient's fused
graph: ``patient → sample → altered gene → (protein partner) → pathway / drug /
breast disease``. Two scores are attached to it:

- **neural**: the mean GNNExplainer importance of the path's edges, taken from
  the fold model that never trained on this patient, and expressed as a
  percentile over every candidate path in the cohort;
- **symbolic (rule)**: the summed weight of the learned logic formulas that the
  path instantiates. A formula counts when it fires for this patient (all its
  atoms are true) and the path is one of its groundings: one of its atoms
  mentions the path's end node or altered gene, or is a relational PrimeKG atom
  such as ``Altered(p, g) AND InPathway(g, w)`` whose gene ``g`` lies on the path. Weights come from the chosen logic reasoner
  (by default the RL-structured survival Markov logic network) of the fold that
  held the patient out. ``symbolic = min(|sum of weights| / S, 1)`` with ``S`` the
  95th percentile over all rule-matched paths, and the sign gives the direction.

The log-rank evidence for the path's end fact is still attached for the report,
but it no longer drives the score. The neuro-symbolic score is
``0.6 * neural + 0.4 * rule``. A path is
*prominent* when its score clears a cohort-wide cutoff. A report shows
``min(K_MAX, prominent paths)`` paths and, when fewer than ``K_MIN`` are
prominent, tops up with the best remaining paths and says so.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from ..dataset_fusion import relations_from_codes
from ..explain import enumerate_explanation_paths, score_edges

NEURAL_WEIGHT = 0.6
SYMBOLIC_WEIGHT = 0.4
Z_CAP = 3.0
K_MAX = 5
K_MIN = 3
MAX_PER_GENE = 2
PROMINENT_QUANTILE = 0.8
CANDIDATES_PER_PATIENT = 60


def explain_patient(model, data, meta: dict) -> tuple[list[dict], dict[str, float], str]:
    """Candidate paths and per-gene importance for one patient graph."""
    altered = [node for node, flag in enumerate(meta["altered"]) if flag]
    torch.manual_seed(0)
    scores, source = score_edges(model, data, altered)
    relations = relations_from_codes(data.rel_code)
    found = enumerate_explanation_paths(
        meta["node_ids"],
        meta["node_types"],
        meta["node_names"],
        meta["altered"],
        data.edge_index,
        data.is_reverse,
        relations,
        scores,
        limit=CANDIDATES_PER_PATIENT,
    )
    paths = []
    for item in found:
        nodes = [
            {"id": meta["node_ids"][index], "type": meta["node_types"][index], "name": meta["node_names"][index]}
            for index in item["nodes"]
        ]
        edge_scores = _edge_scores_along(item["nodes"], data, scores)
        paths.append(
            {
                "nodes": nodes,
                "relations": list(item["relations"]),
                "edge_scores": edge_scores,
                "neural_raw": float(item["path_score"]),
                "text": item["path"],
            }
        )
    gene_importance: dict[str, float] = {}
    sources = data.edge_index[0].tolist()
    targets = data.edge_index[1].tolist()
    values = scores.detach().cpu().float().tolist()
    incident: dict[int, list[float]] = defaultdict(list)
    for src, dst, value, reverse in zip(sources, targets, values, data.is_reverse.tolist()):
        if reverse:
            continue
        incident[src].append(value)
        incident[dst].append(value)
    for node in altered:
        if incident.get(node):
            gene_importance[meta["node_names"][node]] = float(np.mean(incident[node]))
    return paths, gene_importance, source


def _edge_scores_along(node_indices: list[int], data, scores: torch.Tensor) -> list[float]:
    """Score of each forward edge on the path, in order."""
    edge_index = data.edge_index
    reverse = data.is_reverse
    values = []
    for src, dst in zip(node_indices[:-1], node_indices[1:]):
        hit = ((edge_index[0] == src) & (edge_index[1] == dst) & ~reverse).nonzero(as_tuple=True)[0]
        values.append(float(scores[hit[0]]) if len(hit) else float("nan"))
    return values


def paths_frame(patient_id: str, fold: int, paths: list[dict]) -> pd.DataFrame:
    rows = []
    for rank, path in enumerate(paths, start=1):
        terminal = path["nodes"][-1]
        genes = [node["name"] for node in path["nodes"] if node["type"] == "gene"]
        rows.append(
            {
                "patient_id": patient_id,
                "fold": fold,
                "candidate_rank": rank,
                "neural_raw": path["neural_raw"],
                "terminal_id": terminal["id"],
                "terminal_type": terminal["type"],
                "terminal_name": terminal["name"],
                "altered_gene": genes[0] if genes else "",
                "genes": "|".join(genes),
                "nodes": json.dumps(path["nodes"]),
                "relations": json.dumps(path["relations"]),
                "edge_scores": json.dumps(path["edge_scores"]),
                "text": path["text"],
            }
        )
    return pd.DataFrame(rows)


def rule_scores(
    candidates: pd.DataFrame,
    weights: pd.DataFrame,
    atom_frame: pd.DataFrame,
    atom_binary: np.ndarray,
    patient_index: dict[str, int],
    atom_genes: list[set[str]] | None = None,
) -> pd.DataFrame:
    """Signed weight of the fired logic formulas each candidate path instantiates.

    A path instantiates a formula when it is one of the formula's true groundings
    for this patient: the formula fires, and one of its atoms mentions the path's
    end node, its altered gene, or (through ``atom_genes``) a gene on the path, as
    in ``Altered(p, g) AND InPathway(g, w)`` with ``g`` bound to the path's gene.
    """
    atom_position = {name: index for index, name in enumerate(atom_frame["atom"])}
    members = [set(str(m).split("|")) if isinstance(m, str) and m else set() for m in atom_frame["members"]]
    by_member: dict[str, set[int]] = defaultdict(set)
    for index, items in enumerate(members):
        for item in items:
            by_member[item].add(index)
    by_gene: dict[str, set[int]] = defaultdict(set)
    for index, genes in enumerate(atom_genes or []):
        for gene in genes:
            by_gene[gene].add(index)
    formulas_by_fold: dict[int, list[tuple[list[int], float, str]]] = defaultdict(list)
    for row in weights.itertuples(index=False):
        atoms = [atom_position[name] for name in str(row.atoms).split("|") if name in atom_position]
        formulas_by_fold[int(row.fold)].append((atoms, float(row.weight_per_sd), str(row.formula)))
    scores, texts, counts = [], [], []
    for row in candidates.itertuples(index=False):
        truth = atom_binary[patient_index[row.patient_id]]
        relevant = by_member.get(str(row.terminal_id), set()) | by_member.get(str(row.altered_gene), set())
        for node in json.loads(row.nodes):
            if node["type"] == "gene":
                relevant = relevant | by_gene.get(str(node["id"]).rsplit(":", 1)[-1], set())
        fired = [
            (weight, text)
            for atoms, weight, text in formulas_by_fold[int(row.fold)]
            if atoms and relevant.intersection(atoms) and all(truth[a] for a in atoms)
        ]
        scores.append(sum(weight for weight, _ in fired))
        texts.append(max(fired, key=lambda item: abs(item[0]))[1] if fired else "")
        counts.append(len(fired))
    return pd.DataFrame({"rule_score": scores, "rule_formula": texts, "rule_count": counts}, index=candidates.index)


def score_paths(candidates: pd.DataFrame, evidence: pd.DataFrame, rules: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the neural percentile, the rule score, cohort fact evidence, and the fused score.

    Without ``rules`` the symbolic half falls back to the log-rank evidence of the
    path's end fact, which is how the first version of the reports scored paths.
    """
    scored = candidates.copy()
    scored["neural"] = scored["neural_raw"].rank(pct=True)
    lookup = evidence.set_index(["fold", "fact"])[["hazard_ratio", "z", "carriers"]]
    keys = list(zip(scored["fold"], scored["terminal_id"]))
    matched = lookup.reindex(keys)
    scored["fact_hazard_ratio"] = matched["hazard_ratio"].to_numpy()
    scored["fact_z"] = matched["z"].to_numpy()
    scored["fact_carriers"] = matched["carriers"].to_numpy()
    if rules is None:
        strength = np.clip(np.abs(scored["fact_z"].fillna(0.0).to_numpy()) / Z_CAP, 0, 1)
        signed = scored["fact_z"].to_numpy()
    else:
        scored = scored.join(rules)
        magnitude = np.abs(scored["rule_score"].to_numpy())
        matched = magnitude[magnitude > 0]
        scale = float(np.quantile(matched, 0.95)) if matched.size else 1.0
        strength = np.clip(magnitude / max(scale, 1e-9), 0, 1)
        signed = np.where(magnitude > 0, scored["rule_score"].to_numpy(), np.nan)
    scored["symbolic"] = strength
    scored["ns_score"] = NEURAL_WEIGHT * scored["neural"] + SYMBOLIC_WEIGHT * scored["symbolic"]
    scored["evidence_direction"] = np.where(
        np.isnan(signed), "no rule fires" if rules is not None else "no cohort evidence", np.where(signed > 0, "faster progression", "slower progression")
    )
    return scored


def prominence_cutoff(scored: pd.DataFrame, quantile: float = PROMINENT_QUANTILE) -> float:
    return float(scored["ns_score"].quantile(quantile))


def select_top_paths(patient_paths: pd.DataFrame, cutoff: float, k_max: int = K_MAX, k_min: int = K_MIN) -> pd.DataFrame:
    """``min(k_max, prominent)`` paths, topped up to ``k_min`` with the best of the rest.

    At most ``MAX_PER_GENE`` paths go through the same altered gene, each gene
    route and each end node is shown once, so five paths do not repeat one
    amplicon or one protein partner five times.
    """
    ordered = patient_paths.sort_values("ns_score", ascending=False)
    chosen, per_gene, ends, routes = [], defaultdict(int), set(), set()
    has_routes = "genes" in patient_paths.columns
    for row in ordered.itertuples(index=False):
        # One path per gene route: PARP1 → ESR1 → drug X and PARP1 → ESR1 → drug Y say the same thing twice.
        route = row.genes if has_routes else row.altered_gene + row.terminal_id
        if per_gene[row.altered_gene] >= MAX_PER_GENE or row.terminal_id in ends or route in routes:
            continue
        chosen.append(row._asdict())
        per_gene[row.altered_gene] += 1
        ends.add(row.terminal_id)
        routes.add(route)
        if len(chosen) >= k_max:
            break
    frame = pd.DataFrame(chosen, columns=patient_paths.columns)
    if frame.empty:
        frame["prominent"] = []
        return frame
    frame["prominent"] = frame["ns_score"] >= cutoff
    prominent = int(frame["prominent"].sum())
    keep = max(min(k_max, prominent), min(k_min, len(frame)))
    frame = frame.head(keep).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame
