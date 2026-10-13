"""Path explanations over a fused patient graph.

Paths use only edges that exist in the graph. A drug or pathway is mentioned
only when PrimeKG (or the patient alteration) actually connects it.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from .gnn_models import ATTENTION_ARCHITECTURES, PatientGNN

TARGET_TYPES = {"pathway", "disease", "drug"}


def enumerate_explanation_paths(
    node_ids: list[str],
    node_types: list[str],
    node_names: list[str],
    altered: list[bool],
    edge_index: list[tuple[int, int]] | torch.Tensor,
    is_reverse: list[bool] | torch.Tensor,
    relations: list[str],
    scores: list[float] | torch.Tensor,
    limit: int = 3,
) -> list[dict[str, object]]:
    """Rank real patient → altered gene → pathway/drug/disease paths by edge score."""
    if isinstance(edge_index, torch.Tensor):
        pairs = edge_index.detach().cpu().t().tolist()
    else:
        pairs = list(edge_index)
    if isinstance(is_reverse, torch.Tensor):
        reverse_flags = is_reverse.detach().cpu().bool().tolist()
    else:
        reverse_flags = list(is_reverse)
    if isinstance(scores, torch.Tensor):
        score_values = scores.detach().cpu().float().tolist()
    else:
        score_values = [float(value) for value in scores]

    outgoing: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    for (src, dst), reverse, relation, score in zip(pairs, reverse_flags, relations, score_values):
        if reverse:
            continue
        outgoing[int(src)].append((int(dst), relation, float(score)))

    patient_nodes = [index for index, kind in enumerate(node_types) if kind == "patient"]
    if len(patient_nodes) != 1:
        return []
    patient = patient_nodes[0]
    found: list[dict[str, object]] = []
    for gene in [index for index, flag in enumerate(altered) if flag]:
        prefixes = []
        for sample, sample_rel, sample_gene_score in outgoing.get(patient, []):
            if node_types[sample] != "sample":
                continue
            for nxt, gene_rel, gene_score in outgoing.get(sample, []):
                if nxt == gene:
                    prefixes.append(([patient, sample, gene], [sample_rel, gene_rel], [sample_gene_score, gene_score]))
        for nodes, rels, rel_scores in prefixes:
            _extend(nodes, rels, rel_scores, outgoing, node_types, found, depth_left=2)
    found.sort(key=lambda item: float(item["path_score"]), reverse=True)
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in found:
        text = format_path(item["nodes"], item["relations"], node_ids, node_names)
        if text in seen:
            continue
        seen.add(text)
        item["path"] = text
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _extend(nodes, rels, rel_scores, outgoing, node_types, found, depth_left: int) -> None:
    current = nodes[-1]
    if node_types[current] in TARGET_TYPES and len(nodes) >= 4:
        found.append(
            {
                "nodes": list(nodes),
                "relations": list(rels),
                "path_score": sum(rel_scores) / len(rel_scores),
            }
        )
    if depth_left == 0:
        return
    for nxt, relation, score in outgoing.get(current, []):
        if nxt in nodes:
            continue
        kind = node_types[nxt]
        if kind not in TARGET_TYPES and kind != "gene":
            continue
        nodes.append(nxt)
        rels.append(relation)
        rel_scores.append(score)
        _extend(nodes, rels, rel_scores, outgoing, node_types, found, depth_left - 1)
        nodes.pop()
        rels.pop()
        rel_scores.pop()


def format_path(nodes: list[int], relations: list[str], node_ids: list[str], node_names: list[str]) -> str:
    parts = [node_ids[nodes[0]]]
    for relation, node in zip(relations, nodes[1:]):
        name = node_names[node]
        ident = node_ids[node]
        label = ident if not name or name == ident else f"{ident} ({name})"
        parts.append(f"-{relation}-> {label}")
    return " ".join(parts)


def attention_edge_scores(model: nn.Module, data) -> tuple[torch.Tensor, str] | None:
    """Average attention over layers. Returns None when the architecture has no attention."""
    if not isinstance(model, PatientGNN) or model.architecture not in ATTENTION_ARCHITECTURES:
        return None
    model.eval()
    with torch.no_grad():
        _, attentions = model(
            data.x,
            data.edge_index,
            data.emb_id,
            data.is_patient,
            return_attention=True,
            focus=getattr(data, "focus", None),
            rel_code=getattr(data, "rel_code", None),
            hop=getattr(data, "hop", None),
        )
    if not attentions:
        return None
    stacked = torch.stack([layer[: data.edge_index.size(1)] for layer in attentions], dim=0)
    return stacked.mean(dim=0), "gatv2_attention"


def gnnexplainer_edge_scores(model: nn.Module, data, epochs: int = 80) -> torch.Tensor | None:
    """Edge mask from PyG GNNExplainer for a single graph. None if the API rejects the model."""
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
    except ImportError:
        return None

    class _Logits(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x, edge_index, node_emb_index=None, is_patient=None, batch=None, focus=None, rel_code=None, hop=None):
            return self.inner(x, edge_index, node_emb_index, is_patient, batch, focus=focus, rel_code=rel_code, hop=hop).view(-1, 1)

    wrapper = _Logits(model)
    wrapper.eval()
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={"mode": "binary_classification", "task_level": "graph", "return_type": "raw"},
    )
    try:
        explanation = explainer(
            data.x,
            data.edge_index,
            node_emb_index=data.emb_id,
            is_patient=data.is_patient,
            focus=getattr(data, "focus", None),
            rel_code=getattr(data, "rel_code", None),
            hop=getattr(data, "hop", None),
        )
    except Exception:
        return None
    mask = explanation.edge_mask
    if mask is None:
        return None
    return mask.detach().float().view(-1)


def score_edges(model: nn.Module, data, altered: list[int] | None = None) -> tuple[torch.Tensor, str]:
    """Prefer GATv2 attention, then a GNNExplainer edge mask, then an alteration prior."""
    attended = attention_edge_scores(model, data)
    if attended is not None:
        return attended
    mask = gnnexplainer_edge_scores(model, data)
    if mask is not None and mask.numel() == data.edge_index.size(1):
        return mask, "gnnexplainer"
    scores = torch.ones(data.edge_index.size(1), dtype=torch.float)
    altered_nodes = set(altered or [])
    if altered_nodes:
        sources = data.edge_index[0].tolist()
        targets = data.edge_index[1].tolist()
        for index, (src, dst) in enumerate(zip(sources, targets)):
            scores[index] = 1.0 if src in altered_nodes or dst in altered_nodes else 0.2
    return scores, "altered_gene_prior"
