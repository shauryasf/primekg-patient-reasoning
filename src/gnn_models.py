"""Patient-graph classifiers: MLP baseline plus seven message-passing architectures."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINConv,
    RGATConv,
    RGCNConv,
    SAGEConv,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)

from .dataset_fusion import AGE_FEATURE_INDEX, CLINICAL_DIM, RELATION_NAMES, FusionConfig, load_fusion_config
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

MODEL_NAMES = ("mlp", "gcn", "gatv2", "graphsage", "rgcn", "rgat", "gin", "transformer")

# Which message-passing architectures read which part of the stored graph.
# ``rel_code`` is the relation id of every edge, already on the fused graph.
GRAPH_ARCHITECTURES = MODEL_NAMES[1:]
RELATION_ARCHITECTURES = ("rgcn", "rgat")
EDGE_FEATURE_ARCHITECTURES = ("transformer",)
ATTENTION_ARCHITECTURES = ("gatv2", "rgat", "transformer")
# Only these convolutions take the learned graph-structure weight. PyG's RGCNConv,
# RGATConv, GINConv, and TransformerConv do not accept ``edge_weight``, so a GSL run
# is refused for them instead of silently training on unweighted messages.
EDGE_WEIGHT_ARCHITECTURES = ("gcn", "graphsage", "gatv2")
RELATION_EMB_DIM = 8


def _build_conv(architecture: str, dim_in: int, hidden: int, heads: int, dropout: float) -> nn.Module:
    """One message-passing layer for ``architecture``.

    ``rgcn`` and ``rgat`` are relation-aware: they index their weights by the
    relation of each edge, which is the classic treatment of a heterogeneous
    knowledge graph. ``gin`` is relation-agnostic and only sees connectivity.
    ``transformer`` receives a learned relation embedding as its edge feature.
    """
    if architecture == "gcn":
        return GCNConv(dim_in, hidden)
    if architecture == "graphsage":
        return WeightedSAGEConv(dim_in, hidden)
    if architecture == "gatv2":
        return WeightedGATv2Conv(dim_in, hidden, heads=heads, concat=False, dropout=dropout, add_self_loops=False)
    if architecture == "rgcn":
        return RGCNConv(dim_in, hidden, num_relations=len(RELATION_NAMES), num_bases=len(RELATION_NAMES))
    if architecture == "rgat":
        # RGATConv materializes a per-edge weight tensor of size
        # ``[num_edges, in_channels, heads * out_channels]``. With four heads on a
        # batch of fused graphs that is several gigabytes per layer, which thrashes
        # memory, so this one architecture runs a single head.
        return RGATConv(
            dim_in,
            hidden,
            num_relations=len(RELATION_NAMES),
            num_bases=len(RELATION_NAMES),
            heads=1,
            concat=False,
            dropout=dropout,
        )
    if architecture == "gin":
        return GINConv(nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
    if architecture == "transformer":
        return TransformerConv(dim_in, hidden, heads=heads, concat=False, dropout=dropout, edge_dim=RELATION_EMB_DIM)
    raise ValueError(f"Unknown architecture {architecture!r}; expected one of {GRAPH_ARCHITECTURES}")


def _attention_scores(alpha: torch.Tensor) -> torch.Tensor:
    """One edge score per edge, averaged over heads."""
    return alpha.mean(dim=-1) if alpha.dim() > 1 else alpha.reshape(-1)


class TabularMLP(nn.Module):
    """Clinical attributes plus a 500-gene alteration multi-hot. No graph edges.

    When ``clinical_dim`` is set, a linear clinical head is added to an alteration
    network whose last layer starts at zero. The clinical head can keep the
    stage and subtype signal that a wide alteration vector would otherwise swamp.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        dropout: float = 0.3,
        clinical_dim: int = 0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.clinical_dim = clinical_dim
        if clinical_dim < 0 or clinical_dim > in_dim:
            raise ValueError(f"clinical_dim {clinical_dim} is outside feature width {in_dim}")
        self.clinical_head = nn.Linear(clinical_dim, 1) if clinical_dim else None
        self.residual_gate = nn.Parameter(torch.tensor(gate_init)) if clinical_dim else None
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

    ``architecture`` selects the convolution: GCN and GraphSAGE aggregate uniformly,
    GATv2, RGAT, and the graph transformer learn attention, R-GCN and RGAT index their
    weights by relation, and GIN sees only connectivity. GATv2 does not add self-loops,
    so attention weights line up with ``edge_index``. Reverse edges are already in the
    fused graph, which lets pathway context reach the patient node in three hops.
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
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if architecture not in GRAPH_ARCHITECTURES:
            raise ValueError(f"Unknown architecture {architecture}; expected one of {GRAPH_ARCHITECTURES}")
        self.architecture = architecture
        self.dropout = dropout
        self.embedding = nn.Embedding(max(num_embeddings, 1), emb_dim)
        self.relation_embedding = (
            nn.Embedding(len(RELATION_NAMES), RELATION_EMB_DIM) if architecture in EDGE_FEATURE_ARCHITECTURES else None
        )
        convs: list[nn.Module] = []
        input_dim = in_dim + emb_dim
        for layer in range(layers):
            dim_in = input_dim if layer == 0 else hidden
            convs.append(_build_conv(architecture, dim_in, hidden, heads, dropout))
        self.convs = nn.ModuleList(convs)
        self.gsl = EdgeStructureLearner(fusion) if fusion is not None and (fusion.gsl_enabled or fusion.mode == "gsl") else None
        if self.gsl is not None and architecture not in EDGE_WEIGHT_ARCHITECTURES:
            raise ValueError(
                f"{architecture} cannot take the learned graph-structure edge weight. "
                f"Use --fusion-mode fixed, or one of {EDGE_WEIGHT_ARCHITECTURES} with GSL."
            )
        self.classifier = nn.Linear(hidden * 3, 1)
        self.clinical_head = nn.Linear(CLINICAL_DIM, 1)
        self.residual_gate = nn.Parameter(torch.tensor(gate_init))
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

    def _conv_kwargs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        rel_code: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        """Layer inputs that this architecture reads. Every other argument stays default."""
        kwargs: dict[str, torch.Tensor | None] = {}
        if self.architecture in EDGE_WEIGHT_ARCHITECTURES:
            kwargs["edge_weight"] = edge_weight
        if self.architecture in RELATION_ARCHITECTURES:
            kwargs["edge_type"] = self._relation_codes(edge_index, rel_code)
        if self.architecture in EDGE_FEATURE_ARCHITECTURES:
            relation = self._relation_codes(edge_index, rel_code)
            kwargs["edge_attr"] = self.relation_embedding(relation)
        return kwargs

    def _relation_codes(self, edge_index: torch.Tensor, rel_code: torch.Tensor | None) -> torch.Tensor:
        """Relation id per edge, defaulting to relation 0 when the graph carries none."""
        if rel_code is None:
            return torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        return rel_code.clamp(min=0, max=len(RELATION_NAMES) - 1)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_emb_index: torch.Tensor,
        return_attention: bool = False,
        edge_weight: torch.Tensor | None = None,
        rel_code: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = torch.cat([self.embedding(node_emb_index), x], dim=-1)
        conv_kwargs = self._conv_kwargs(x, edge_index, edge_weight, rel_code)
        attentions: list[torch.Tensor] = []
        for layer, conv in enumerate(self.convs):
            if return_attention and self.architecture in ATTENTION_ARCHITECTURES:
                hidden, (_, alpha) = conv(hidden, edge_index, return_attention_weights=True, **conv_kwargs)
                attentions.append(_attention_scores(alpha))
            else:
                hidden = conv(hidden, edge_index, **conv_kwargs)
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
            hidden, attentions = self.encode(
                x, edge_index, node_emb_index, return_attention=True, edge_weight=edge_weight, rel_code=rel_code
            )
            return self._logits(x, hidden, is_patient, batch, focus), attentions
        hidden = self.encode(
            x, edge_index, node_emb_index, return_attention=False, edge_weight=edge_weight, rel_code=rel_code
        )
        return self._logits(x, hidden, is_patient, batch, focus)


def build_model(
    name: str,
    feature_dim: int,
    num_embeddings: int,
    mlp_in_dim: int,
    mlp_clinical_dim: int = 0,
    fusion: FusionConfig | None = None,
    gate_init: float = 0.0,
) -> nn.Module:
    if name == "mlp":
        return TabularMLP(mlp_in_dim, clinical_dim=mlp_clinical_dim, gate_init=gate_init)
    return PatientGNN(
        name,
        feature_dim,
        num_embeddings,
        fusion=fusion if fusion is not None else load_fusion_config(),
        gate_init=gate_init,
    )
