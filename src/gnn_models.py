"""Patient-graph classifiers: MLP baseline, GCN, GATv2, and GraphSAGE."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_max_pool, global_mean_pool

MODEL_NAMES = ("mlp", "gcn", "gatv2", "graphsage")


class TabularMLP(nn.Module):
    """Clinical attributes plus a 500-gene alteration multi-hot. No graph edges."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class PatientGNN(nn.Module):
    """Three message-passing layers and a patient + global mean/max readout.

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
                convs.append(SAGEConv(dim_in, hidden))
            else:
                convs.append(GATv2Conv(dim_in, hidden, heads=heads, concat=False, dropout=dropout, add_self_loops=False))
        self.convs = nn.ModuleList(convs)
        self.classifier = nn.Linear(hidden * 3, 1)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_emb_index: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = torch.cat([self.embedding(node_emb_index), x], dim=-1)
        attentions: list[torch.Tensor] = []
        for layer, conv in enumerate(self.convs):
            if return_attention and isinstance(conv, GATv2Conv):
                hidden, (_, alpha) = conv(hidden, edge_index, return_attention_weights=True)
                attentions.append(alpha.mean(dim=-1))
            else:
                hidden = conv(hidden, edge_index)
            if layer + 1 < len(self.convs):
                hidden = torch.relu(hidden)
                hidden = torch.nn.functional.dropout(hidden, p=self.dropout, training=self.training)
        if return_attention:
            return hidden, attentions
        return hidden

    def readout(self, hidden: torch.Tensor, is_patient: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        if batch is None:
            batch = hidden.new_zeros(hidden.size(0), dtype=torch.long)
        patient = hidden[is_patient.bool()]
        pooled = torch.cat([patient, global_mean_pool(hidden, batch), global_max_pool(hidden, batch)], dim=-1)
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_emb_index: torch.Tensor,
        is_patient: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if return_attention:
            hidden, attentions = self.encode(x, edge_index, node_emb_index, return_attention=True)
            logits = self.classifier(self.readout(hidden, is_patient, batch)).view(-1)
            return logits, attentions
        hidden = self.encode(x, edge_index, node_emb_index, return_attention=False)
        return self.classifier(self.readout(hidden, is_patient, batch)).view(-1)


def build_model(name: str, feature_dim: int, num_embeddings: int, mlp_in_dim: int) -> nn.Module:
    if name == "mlp":
        return TabularMLP(mlp_in_dim)
    return PatientGNN(name, feature_dim, num_embeddings)
