"""Patient-conditioned scores for candidate PrimeKG edges.

The scorer does not retrieve PrimeKG. It only weights edges that graph
construction already stored. Observed patient-sample-gene edges stay at 1.
Outcome labels are not inputs.
"""
from __future__ import annotations

import torch
from torch import nn

from .dataset_fusion import (
    AGE_FEATURE_INDEX,
    CLINICAL_DIM,
    FEATURE_DIM,
    OBSERVED_RELATIONS,
    RELATION_NAMES,
    RELATION_TO_CODE,
    FusionConfig,
)


class EdgeStructureLearner(nn.Module):
    """Patient-specific hop gate times an edge residual.

    Hops through ``fixed_hops`` start at weight 1, which is the fixed-neighborhood
    baseline. Farther candidate hops start at ``sigmoid(gsl_closed_logit)`` and a
    patient head can open them. The edge and relation residual is ``2 * sigmoid(logit)``.
    That is 1 at a zero start, and the slope there is not zero.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.gsl_hidden)
        self.observed = {RELATION_TO_CODE[name] for name in OBSERVED_RELATIONS}
        self.n_hops = 8
        self.node_encoder = nn.Linear(FEATURE_DIM, hidden)
        self.relation_embedding = nn.Embedding(len(RELATION_NAMES), int(config.gsl_relation_dim))
        self.patient_encoder = nn.Linear(CLINICAL_DIM, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 3 + int(config.gsl_relation_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)
        self.relation_logit = nn.Parameter(torch.zeros(len(RELATION_NAMES)))
        hop_bias = torch.full((self.n_hops,), float(config.gsl_closed_logit))
        keep = min(max(int(config.fixed_hops), 0), self.n_hops - 1)
        if keep > 0:
            hop_bias[1 : keep + 1] = float(config.gsl_open_logit)
        self.hop_bias = nn.Parameter(hop_bias)
        self.patient_hop = nn.Linear(hidden, self.n_hops)
        nn.init.zeros_(self.patient_hop.weight)
        nn.init.zeros_(self.patient_hop.bias)
        self.node_mlp = nn.Linear(hidden, 1) if config.gsl_node_scores else None
        self.last_hop_gate: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        rel_code: torch.Tensor,
        is_patient: torch.Tensor,
        hop: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            self.last_hop_gate = x.new_zeros(0)
            return x.new_zeros(0)
        hidden = self.node_encoder(x)
        source = edge_index[0]
        destination = edge_index[1]
        relation_codes = rel_code.clamp(min=0, max=len(RELATION_NAMES) - 1)
        relation = self.relation_embedding(relation_codes)
        hop_values = hop if hop is not None else torch.zeros(rel_code.shape[0], dtype=torch.long, device=x.device)
        hop_values = hop_values.clamp(min=0, max=self.n_hops - 1)
        patient_hidden = self._patient_context(x, hidden, is_patient, source, batch)
        delta = self.edge_mlp(torch.cat([hidden[source], hidden[destination], relation, patient_hidden], dim=-1)).view(-1)
        temperature = max(float(self.config.gsl_temperature), 1e-3)
        # 2 * sigmoid(0) is 1, and the slope there is not zero, unlike clamp(..., max=1).
        edge_factor = 2.0 * torch.sigmoid((delta + self.relation_logit[relation_codes]) / temperature)
        hop_logit = self.hop_bias[hop_values] + self.patient_hop(patient_hidden).gather(1, hop_values.view(-1, 1)).view(-1)
        hop_gate = torch.sigmoid(hop_logit)
        self.last_hop_gate = hop_gate
        keep = min(max(int(self.config.fixed_hops), 0), self.n_hops - 1)
        inside_prior = (hop_values >= 1) & (hop_values <= keep)
        learned = torch.where(inside_prior, edge_factor, hop_gate * edge_factor)
        if self.node_mlp is not None:
            node_score = torch.sigmoid(self.node_mlp(hidden).view(-1))
            node_score = torch.where(is_patient.bool(), torch.ones_like(node_score), node_score)
            learned = learned * node_score[destination]
        observed = torch.tensor([int(code) in self.observed for code in rel_code.tolist()], dtype=torch.bool, device=x.device)
        weight = torch.where(observed, torch.ones_like(learned), learned)
        if self.config.gsl_hard_prune and not self.training:
            keep_edge = weight >= float(self.config.gsl_threshold)
            weight = torch.where(observed | keep_edge, weight, torch.zeros_like(weight))
        return weight

    def _patient_context(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        is_patient: torch.Tensor,
        source: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.config.gsl_patient_context:
            return hidden.new_zeros(source.shape[0], hidden.shape[1])
        patient_rows = is_patient.bool().nonzero(as_tuple=False).view(-1)
        clinical_width = CLINICAL_DIM
        if patient_rows.numel() == 0:
            return self.patient_encoder(x.new_zeros(source.shape[0], clinical_width))
        patient_clinical = x[patient_rows, AGE_FEATURE_INDEX : AGE_FEATURE_INDEX + clinical_width]
        if batch is None:
            return self.patient_encoder(patient_clinical[:1].expand(source.shape[0], -1))
        table = x.new_zeros(int(batch.max().item()) + 1, clinical_width)
        table[batch[patient_rows]] = patient_clinical
        return self.patient_encoder(table[batch[source]])


def edge_report(
    data,
    meta: dict[str, object],
    weight: torch.Tensor,
    threshold: float,
    hop_gate: torch.Tensor | None = None,
) -> list[dict[str, object]]:
    """One row per stored forward edge, including the learned weight and the original hop."""
    sources = data.edge_index[0].detach().cpu().tolist()
    destinations = data.edge_index[1].detach().cpu().tolist()
    scores = weight.detach().cpu().tolist()
    hops = data.hop.detach().cpu().tolist() if getattr(data, "hop", None) is not None else [0] * len(sources)
    gates = hop_gate.detach().cpu().tolist() if hop_gate is not None else [None] * len(sources)
    relations = [RELATION_NAMES[int(code)] for code in data.rel_code.detach().cpu().tolist()]
    reverses = data.is_reverse.detach().cpu().tolist()
    node_ids = list(meta.get("node_ids", []))
    rows = []
    for src, dst, relation, hop, score, gate, reverse in zip(sources, destinations, relations, hops, scores, gates, reverses):
        if reverse:
            continue
        kept = bool(score >= threshold) or relation in OBSERVED_RELATIONS
        rows.append(
            {
                "source": node_ids[src] if node_ids else int(src),
                "relation": relation,
                "destination": node_ids[dst] if node_ids else int(dst),
                "hop": int(hop),
                "score": float(score),
                "hop_gate": None if gate is None else float(gate),
                "retained": kept,
                "knowledge": relation not in OBSERVED_RELATIONS,
            }
        )
    return rows
