"""Patient-graph classifiers: MLP baseline, GCN, GATv2, and GraphSAGE."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_max_pool, global_mean_pool

from .dataset_fusion import AGE_FEATURE_INDEX, CLINICAL_DIM, FusionConfig, load_fusion_config
from .gsl import EdgeStructureLearner


class WeightedSAGEConv(SAGEConv):
    """GraphSAGE whose neighbor messages are multiplied by a per-edge weight."""

    def forward(self, x, edge_index, edge_weight=None, size=None):
        self._gsl_weight = edge_weight
        return SAGEConv.forward(self, x, edge_index, size=size)

    def message(self, x_j):
        weight = getattr(self, "_gsl_weight", None)
        if weight is None:
            return x_j
        return x_j * weight.view(-1, 1)


class WeightedGATv2Conv(GATv2Conv):
    """GATv2 whose messages are scaled by a learned PrimeKG edge weight."""

    def forward(self, x, edge_index, edge_weight=None, return_attention_weights=None):
        self._gsl_weight = edge_weight
        return super().forward(x, edge_index, return_attention_weights=return_attention_weights)

    def message(self, x_j, alpha):
        message = x_j * alpha.unsqueeze(-1)
        weight = getattr(self, "_gsl_weight", None)
        if weight is None:
            return message
        return message * weight.view(-1, *([1] * (message.dim() - 1)))

MODEL_NAMES = ("mlp", "gcn", "gatv2", "graphsage")


class TabularMLP(nn.Module):
    """Clinical attributes plus a 500-gene alteration multi-hot. No graph edges.

    When ``clinical_dim`` is set, a linear clinical head is added to an alteration
    network whose last layer starts at zero. The clinical head can keep the
    stage and subtype signal that a wide alteration vector would otherwise swamp.
    """

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3, clinical_dim: int = 0) -> None:
        super().__init__()
        self.clinical_dim = clinical_dim
        if clinical_dim < 0 or clinical_dim > in_dim:
            raise ValueError(f"clinical_dim {clinical_dim} is outside feature width {in_dim}")
        self.clinical_head = nn.Linear(clinical_dim, 1) if clinical_dim else None
        self.residual_gate = nn.Parameter(torch.tensor(-2.0)) if clinical_dim else None
        alteration_dim = in_dim - clinical_dim
        self.net = nn.Sequential(
            nn.Linear(alteration_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if self.clinical_head is not None:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clinical_head is None:
            return self.net(x).view(-1)
        clinical = self.clinical_head(x[:, : self.clinical_dim]).view(-1)
        alteration = self.net(x[:, self.clinical_dim :]).view(-1)
        return clinical + torch.sigmoid(self.residual_gate) * alteration


class PatientGNN(nn.Module):
    """Three message-passing layers, a patient + altered-gene readout, and a clinical residual.

    Mean and max pooling use the patient node and altered genes when ``focus`` is set.
    Unaltered pathway neighbors stay in the graph for message passing and explanations,
    and they are left out of the pool so they do not average away the patient signal.
    A linear head on the patient clinical block is added to the graph logit. The graph
    classifier starts at zero, so training begins from the clinical score.

    GATv2 does not add self-loops, so attention weights line up with ``edge_index``.
    Reverse edges are already in the fused graph, which lets pathway context reach
    the patient node in three hops.
    """

    def __init__(
        self,
        architecture: str,
        in_dim: int,
        num_embeddings: int,
        hidden: int = 64,
        emb_dim: int = 32,
        layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        fusion: FusionConfig | None = None,
    ) -> None:
        super().__init__()
        if architecture not in {"gcn", "gatv2", "graphsage"}:
            raise ValueError(f"Unknown architecture {architecture}")
        self.architecture = architecture
        self.dropout = dropout
        self.embedding = nn.Embedding(max(num_embeddings, 1), emb_dim)
        convs: list[nn.Module] = []
        input_dim = in_dim + emb_dim
        for layer in range(layers):
            dim_in = input_dim if layer == 0 else hidden
            if architecture == "gcn":
                convs.append(GCNConv(dim_in, hidden))
            elif architecture == "graphsage":
                convs.append(WeightedSAGEConv(dim_in, hidden))
            else:
                convs.append(WeightedGATv2Conv(dim_in, hidden, heads=heads, concat=False, dropout=dropout, add_self_loops=False))
        self.convs = nn.ModuleList(convs)
        self.gsl = EdgeStructureLearner(fusion) if fusion is not None and (fusion.gsl_enabled or fusion.mode == "gsl") else None
        self.classifier = nn.Linear(hidden * 3, 1)
        self.clinical_head = nn.Linear(CLINICAL_DIM, 1)
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def edge_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        rel_code: torch.Tensor | None,
        is_patient: torch.Tensor,
        hop: torch.Tensor | None,
        batch: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.gsl is None or rel_code is None:
            return None
        return self.gsl(x, edge_index, rel_code, is_patient, hop, batch)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_emb_index: torch.Tensor,
        return_attention: bool = False,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = torch.cat([self.embedding(node_emb_index), x], dim=-1)
        attentions: list[torch.Tensor] = []
        for layer, conv in enumerate(self.convs):
            if return_attention and isinstance(conv, GATv2Conv):
                hidden, (_, alpha) = conv(hidden, edge_index, edge_weight=edge_weight, return_attention_weights=True)
                attentions.append(alpha.mean(dim=-1))
            elif isinstance(conv, GCNConv):
                hidden = conv(hidden, edge_index, edge_weight=edge_weight)
            else:
                hidden = conv(hidden, edge_index, edge_weight=edge_weight)
            if layer + 1 < len(self.convs):
                hidden = torch.relu(hidden)
                hidden = torch.nn.functional.dropout(hidden, p=self.dropout, training=self.training)
        if return_attention:
            return hidden, attentions
        return hidden

    def readout(
        self,
        hidden: torch.Tensor,
        is_patient: torch.Tensor,
        batch: torch.Tensor | None,
        focus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = hidden.new_zeros(hidden.size(0), dtype=torch.long)
        if focus is None:
            focus = torch.ones(hidden.size(0), dtype=torch.bool, device=hidden.device)
        else:
            focus = focus.bool()
        patient = hidden[is_patient.bool()]
        pooled = torch.cat(
            [
                patient,
                global_mean_pool(hidden[focus], batch[focus]),
                global_max_pool(hidden[focus], batch[focus]),
            ],
            dim=-1,
        )
        return pooled

    def _logits(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        is_patient: torch.Tensor,
        batch: torch.Tensor | None,
        focus: torch.Tensor | None,
    ) -> torch.Tensor:
        graph_logit = self.classifier(self.readout(hidden, is_patient, batch, focus)).view(-1)
        clinical = x[is_patient.bool(), AGE_FEATURE_INDEX : AGE_FEATURE_INDEX + CLINICAL_DIM]
        return self.clinical_head(clinical).view(-1) + torch.sigmoid(self.residual_gate) * graph_logit

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_emb_index: torch.Tensor,
        is_patient: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_attention: bool = False,
        focus: torch.Tensor | None = None,
        rel_code: torch.Tensor | None = None,
        hop: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        edge_weight = self.edge_weights(x, edge_index, rel_code, is_patient, hop, batch)
        if return_attention:
            hidden, attentions = self.encode(x, edge_index, node_emb_index, return_attention=True, edge_weight=edge_weight)
            return self._logits(x, hidden, is_patient, batch, focus), attentions
        hidden = self.encode(x, edge_index, node_emb_index, return_attention=False, edge_weight=edge_weight)
        return self._logits(x, hidden, is_patient, batch, focus)


def build_model(
    name: str,
    feature_dim: int,
    num_embeddings: int,
    mlp_in_dim: int,
    mlp_clinical_dim: int = 0,
    fusion: FusionConfig | None = None,
) -> nn.Module:
    if name == "mlp":
        return TabularMLP(mlp_in_dim, clinical_dim=mlp_clinical_dim)
    return PatientGNN(name, feature_dim, num_embeddings, fusion=fusion if fusion is not None else load_fusion_config())
