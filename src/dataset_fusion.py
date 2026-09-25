"""Fuse 500-gene patient graphs with a filtered PrimeKG neighborhood.

PrimeKG proteins join patient genes on NCBI Entrez id. PPI edges are the induced
subgraph of the 500-gene panel. Pathway, drug, and breast/mammary disease edges
are the 1-hop biomedical context. Unaltered panel genes that share a pathway
with an altered gene are kept so alterations can meet on a pathway node.
Two-hop PPI expansion is intentionally not used: it pulls in most of PrimeKG.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .clinical import link_patients_to_samples, load_clinical_data
from .cna import load_cna_for_samples
from .config import load_feature_policy, repository_root
from .io_utils import normalize_identifier
from .labels import CLINICAL_FEATURE_FIELDS, OUTCOME_FIELDS, assert_no_outcome_leakage
from .mutations import load_mutations

KEPT_RELATIONS = ("protein_protein", "pathway_protein", "disease_protein", "drug_protein")
NODE_TYPES = ("patient", "sample", "gene", "pathway", "disease", "drug")
SUBTYPES = ("BRCA_Basal", "BRCA_Her2", "BRCA_LumA", "BRCA_LumB", "BRCA_Normal", "missing")
STAGES = ("I", "II", "III", "IV", "X", "missing")
M_STAGES = ("M0", "M1", "MX", "CM0", "missing")
RELATION_NAMES = (
    "has_sample",
    "has_mutation",
    "has_high_level_amplification",
    "has_homozygous_deletion",
    "protein_protein",
    "pathway_protein",
    "disease_protein",
    "drug_protein",
)
RELATION_TO_CODE = {name: index for index, name in enumerate(RELATION_NAMES)}
OBSERVED_RELATIONS = RELATION_NAMES[:4]
PRIMEKG_RELATIONS = RELATION_NAMES[4:]
TYPE_DIM = len(NODE_TYPES)
AGE_FEATURE_INDEX = TYPE_DIM
CLINICAL_DIM = 1 + 1 + len(SUBTYPES) + len(STAGES) + len(M_STAGES)
ALTERATION_DIM = 5
FEATURE_DIM = TYPE_DIM + CLINICAL_DIM + ALTERATION_DIM
MLP_AGE_INDEX = 0
SLICE_COLUMNS = [
    "relation",
    "display_relation",
    "x_id",
    "x_type",
    "x_name",
    "y_id",
    "y_type",
    "y_name",
    "x_entrez",
    "y_entrez",
]


def feature_names() -> list[str]:
    names = [f"type_{kind}" for kind in NODE_TYPES]
    names.append("age")
    names.append("sex")
    names.extend(f"subtype_{value}" for value in SUBTYPES)
    names.extend(f"stage_{value}" for value in STAGES)
    names.extend(f"m_stage_{value}" for value in M_STAGES)
    names.extend(
        [
            "alt_mutated",
            "alt_high_amplification",
            "alt_homozygous_deletion",
            "alt_log1p_mutation_count",
            "alt_known_driver",
        ]
    )
    if len(names) != FEATURE_DIM:
        raise RuntimeError(f"Feature width {len(names)} does not match FEATURE_DIM {FEATURE_DIM}")
    assert_no_outcome_leakage(names)
    return names


def normalize_entrez(value: object) -> str | None:
    """Normalize panel floats such as ``5290.0`` and PrimeKG ids such as ``5290``."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "null", "inf", "infinity"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return str(int(text))
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-6:
        return None
    return str(rounded)


def collapse_stage(value: object) -> str:
    text = normalize_identifier(value)
    if text is None:
        return "missing"
    compact = re.sub(r"\s+", "", text.upper())
    for label in ("IV", "III", "II", "I"):
        if re.search(rf"STAGE{label}(?!I)", compact):
            return label
    if "STAGEX" in compact or compact.endswith("X"):
        return "X"
    return "missing"


def collapse_m_stage(value: object) -> str:
    text = normalize_identifier(value)
    if text is None:
        return "missing"
    compact = re.sub(r"[^A-Z0-9]", "", text.upper())
    if "CM0" in compact:
        return "CM0"
    if compact.startswith("M0"):
        return "M0"
    if compact.startswith("M1"):
        return "M1"
    if compact.startswith("MX"):
        return "MX"
    return "missing"


def parse_age(value: object) -> float:
    text = normalize_identifier(value)
    if text is None:
        return math.nan
    try:
        age = float(text)
    except ValueError:
        return math.nan
    if age < 0 or age > 120:
        return math.nan
    return age


def encode_sex(value: object) -> float:
    text = normalize_identifier(value)
    if text is None:
        return 0.5
    upper = text.upper()
    if upper in {"FEMALE", "F"}:
        return 1.0
    if upper in {"MALE", "M"}:
        return 0.0
    return 0.5


def _one_hot(value: str, categories: tuple[str, ...]) -> list[float]:
    folded = {category.casefold(): category for category in categories}
    key = value if value in categories else folded.get(str(value).casefold(), "missing")
    if key not in categories:
        key = "missing"
    return [1.0 if category == key else 0.0 for category in categories]


def clinical_block(row: dict | pd.Series) -> np.ndarray:
    """19 clinical features. Outcome columns are never read."""
    getter = row.get if isinstance(row, dict) else lambda key, default=None: row[key] if key in row.index else default
    block = np.zeros(CLINICAL_DIM, dtype=np.float32)
    block[0] = parse_age(getter("AGE"))
    block[1] = encode_sex(getter("SEX"))
    subtype = normalize_identifier(getter("SUBTYPE")) or "missing"
    stage = collapse_stage(getter("AJCC_PATHOLOGIC_TUMOR_STAGE"))
    m_stage = collapse_m_stage(getter("PATH_M_STAGE"))
    cursor = 2
    subtype_vec = _one_hot(subtype, SUBTYPES)
    block[cursor : cursor + len(subtype_vec)] = subtype_vec
    cursor += len(subtype_vec)
    stage_vec = _one_hot(stage, STAGES)
    block[cursor : cursor + len(stage_vec)] = stage_vec
    cursor += len(stage_vec)
    block[cursor:] = _one_hot(m_stage, M_STAGES)
    return block


def node_feature_vector(node_type: str, clinical: np.ndarray | None = None, alteration: np.ndarray | None = None) -> np.ndarray:
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    vector[NODE_TYPES.index(node_type)] = 1.0
    if clinical is not None:
        vector[AGE_FEATURE_INDEX : AGE_FEATURE_INDEX + CLINICAL_DIM] = clinical
    if alteration is not None:
        vector[AGE_FEATURE_INDEX + CLINICAL_DIM :] = alteration
    return vector


def load_gene_panel(path: Path | None = None) -> dict[str, dict[str, object]]:
    panel_path = path or repository_root() / load_feature_policy().get("gene_panel", "config/gene_panel_500.json")
    raw = json.loads(panel_path.read_text(encoding="utf-8"))
    panel: dict[str, dict[str, object]] = {}
    for symbol, info in raw.items():
        panel[str(symbol).upper()] = {
            "entrez": normalize_entrez(info.get("entrez_id")) or "",
            "is_known_driver": bool(info.get("is_known_driver", False)),
        }
    return panel


def filter_primekg_frame(frame: pd.DataFrame, panel_entrez: set[str]) -> pd.DataFrame:
    """Keep panel PPI plus pathway, drug, and breast/mammary disease edges."""
    if frame.empty or "relation" not in frame.columns:
        return pd.DataFrame(columns=SLICE_COLUMNS)
    working = frame[frame["relation"].isin(KEPT_RELATIONS)].copy()
    if working.empty:
        return pd.DataFrame(columns=SLICE_COLUMNS)
    for column in ("x_id", "x_type", "x_name", "y_id", "y_type", "y_name"):
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].astype(str)
    if "display_relation" not in working.columns:
        working["display_relation"] = ""
    x_entrez = working["x_id"].map(normalize_entrez)
    y_entrez = working["y_id"].map(normalize_entrez)
    x_gene = working["x_type"].str.contains("gene", case=False) & x_entrez.isin(panel_entrez)
    y_gene = working["y_type"].str.contains("gene", case=False) & y_entrez.isin(panel_entrez)
    relation = working["relation"]
    pathway = relation.eq("pathway_protein") & (
        (x_gene & working["y_type"].str.contains("pathway", case=False))
        | (y_gene & working["x_type"].str.contains("pathway", case=False))
    )
    drug = relation.eq("drug_protein") & (
        (x_gene & working["y_type"].str.contains("drug", case=False))
        | (y_gene & working["x_type"].str.contains("drug", case=False))
    )
    disease_name = working["x_name"].where(working["x_type"].str.contains("disease", case=False), working["y_name"])
    disease_ok = disease_name.str.contains(r"breast|mammary", case=False, na=False)
    disease = relation.eq("disease_protein") & disease_ok & (
        (x_gene & working["y_type"].str.contains("disease", case=False))
        | (y_gene & working["x_type"].str.contains("disease", case=False))
    )
    ppi = relation.eq("protein_protein") & x_gene & y_gene & x_entrez.ne(y_entrez)
    kept = working.loc[ppi | pathway | drug | disease].copy()
    if kept.empty:
        return pd.DataFrame(columns=SLICE_COLUMNS)
    kept["x_entrez"] = x_entrez.loc[kept.index].fillna("")
    kept["y_entrez"] = y_entrez.loc[kept.index].fillna("")
    kept["display_relation"] = kept["display_relation"].astype(str)
    return kept.loc[:, SLICE_COLUMNS].drop_duplicates(subset=["relation", "x_id", "x_type", "y_id", "y_type"])


def load_primekg_slice(kg_csv: Path, panel_entrez: set[str], cache_path: Path | None = None) -> pd.DataFrame:
    """Stream PrimeKG and cache the BRCA-panel slice."""
    cache = cache_path or kg_csv.with_name("primekg_panel_slice.csv")
    if cache.exists() and cache.stat().st_size > 0:
        return pd.read_csv(cache, dtype=str, keep_default_na=False)
    usecols = ["relation", "display_relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(kg_csv, usecols=lambda column: column in usecols, dtype=str, chunksize=250_000, low_memory=False):
        filtered = filter_primekg_frame(chunk, panel_entrez)
        if not filtered.empty:
            parts.append(filtered)
    slice_frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SLICE_COLUMNS)
    if not slice_frame.empty:
        slice_frame = slice_frame.drop_duplicates(subset=["relation", "x_id", "x_type", "y_id", "y_type"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    slice_frame.to_csv(cache, index=False)
    return slice_frame


def _semantic_type(prime_type: str) -> str:
    text = prime_type.lower()
    if "pathway" in text:
        return "pathway"
    if "drug" in text:
        return "drug"
    if "disease" in text:
        return "disease"
    if "gene" in text or "protein" in text:
        return "gene"
    return "other"


@dataclass
class PrimeSlice:
    ppi: dict[str, set[str]] = field(default_factory=dict)
    gene_pathways: dict[str, set[str]] = field(default_factory=dict)
    pathway_genes: dict[str, set[str]] = field(default_factory=dict)
    pathway_names: dict[str, str] = field(default_factory=dict)
    gene_drugs: dict[str, set[str]] = field(default_factory=dict)
    drug_names: dict[str, str] = field(default_factory=dict)
    gene_diseases: dict[str, set[str]] = field(default_factory=dict)
    disease_names: dict[str, str] = field(default_factory=dict)
    protein_ids: set[str] = field(default_factory=set)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "PrimeSlice":
        slice_ = cls()
        if frame.empty:
            return slice_
        for record in frame.itertuples(index=False):
            relation = str(record.relation)
            x_type = _semantic_type(str(record.x_type))
            y_type = _semantic_type(str(record.y_type))
            x_entrez = normalize_entrez(record.x_entrez) or ""
            y_entrez = normalize_entrez(record.y_entrez) or ""
            if x_type == "gene" and x_entrez:
                slice_.protein_ids.add(x_entrez)
            if y_type == "gene" and y_entrez:
                slice_.protein_ids.add(y_entrez)
            if relation == "protein_protein" and x_entrez and y_entrez:
                slice_.ppi.setdefault(x_entrez, set()).add(y_entrez)
                slice_.ppi.setdefault(y_entrez, set()).add(x_entrez)
                continue
            gene_entrez, other_type, other_id, other_name = _gene_and_other(
                x_type, x_entrez, str(record.x_id), str(record.x_name), y_type, y_entrez, str(record.y_id), str(record.y_name)
            )
            if not gene_entrez or other_type == "other":
                continue
            if relation == "pathway_protein" and other_type == "pathway":
                slice_.gene_pathways.setdefault(gene_entrez, set()).add(other_id)
                slice_.pathway_genes.setdefault(other_id, set()).add(gene_entrez)
                slice_.pathway_names.setdefault(other_id, other_name)
            elif relation == "drug_protein" and other_type == "drug":
                slice_.gene_drugs.setdefault(gene_entrez, set()).add(other_id)
                slice_.drug_names.setdefault(other_id, other_name)
            elif relation == "disease_protein" and other_type == "disease":
                slice_.gene_diseases.setdefault(gene_entrez, set()).add(other_id)
                slice_.disease_names.setdefault(other_id, other_name)
        return slice_


@dataclass
class FusionConfig:
    """Candidate PrimeKG depth and graph-structure-learning switches.

    ``fixed_hops`` is the baseline neighborhood. ``max_candidate_hops`` is only
    the upper bound GSL is allowed to see. Neither number is the GNN depth.
    """

    mode: str = "fixed"
    fixed_hops: int = 2
    max_candidate_hops: int = 3
    gsl_enabled: bool = False
    gsl_threshold: float = 0.5
    gsl_hard_prune: bool = False
    gsl_patient_context: bool = True
    gsl_node_scores: bool = False
    gsl_hidden: int = 32
    gsl_relation_dim: int = 8
    gsl_temperature: float = 1.0

    @property
    def candidate_hops(self) -> int:
        if self.gsl_enabled or self.mode == "gsl":
            return int(self.max_candidate_hops)
        return int(self.fixed_hops)


def load_fusion_config(policy: dict | None = None) -> FusionConfig:
    raw = dict((policy if policy is not None else load_feature_policy()).get("primekg_fusion") or {})
    allowed = set(FusionConfig.__dataclass_fields__)
    return FusionConfig(**{key: value for key, value in raw.items() if key in allowed})


@dataclass
class PrimeNeighborhood:
    gene_hops: dict[str, int]
    pathway_hops: dict[str, int]
    drug_hops: dict[str, int]
    disease_hops: dict[str, int]
    ppi_partners: set[str]


def expand_primekg_neighborhood(
    seeds: set[str],
    prime: PrimeSlice,
    panel_entrez: set[str],
    max_hops: int,
) -> PrimeNeighborhood:
    """Bounded PrimeKG candidate search.

    Protein-protein genes are added only as direct panel neighbors of an altered
    gene. A partner's own protein neighbors are not crawled. Pathway, drug, and
    disease links can continue out to ``max_hops``.
    """
    hops = max(int(max_hops), 0)
    gene_hops = {entrez: 0 for entrez in seeds}
    ppi_partners: set[str] = set()
    if hops >= 1:
        for seed in seeds:
            for partner in prime.ppi.get(seed, set()):
                if partner not in panel_entrez or partner in seeds:
                    continue
                ppi_partners.add(partner)
                gene_hops[partner] = min(gene_hops.get(partner, hops + 1), 1)
    pathway_hops: dict[str, int] = {}
    drug_hops: dict[str, int] = {}
    disease_hops: dict[str, int] = {}
    if hops == 0:
        return PrimeNeighborhood(gene_hops, pathway_hops, drug_hops, disease_hops, ppi_partners)
    changed = True
    while changed:
        changed = False
        for gene, gene_hop in list(gene_hops.items()):
            next_hop = gene_hop + 1
            if next_hop > hops:
                continue
            for pathway_id in prime.gene_pathways.get(gene, set()):
                if pathway_hops.get(pathway_id, hops + 1) > next_hop:
                    pathway_hops[pathway_id] = next_hop
                    changed = True
            for drug_id in prime.gene_drugs.get(gene, set()):
                if drug_hops.get(drug_id, hops + 1) > next_hop:
                    drug_hops[drug_id] = next_hop
                    changed = True
            for disease_id in prime.gene_diseases.get(gene, set()):
                if disease_hops.get(disease_id, hops + 1) > next_hop:
                    disease_hops[disease_id] = next_hop
                    changed = True
        for pathway_id, pathway_hop in list(pathway_hops.items()):
            next_hop = pathway_hop + 1
            if next_hop > hops:
                continue
            for gene in prime.pathway_genes.get(pathway_id, set()):
                if gene not in panel_entrez:
                    continue
                if gene_hops.get(gene, hops + 1) > next_hop:
                    gene_hops[gene] = next_hop
                    changed = True
    return PrimeNeighborhood(gene_hops, pathway_hops, drug_hops, disease_hops, ppi_partners)


def _gene_and_other(x_type, x_entrez, x_id, x_name, y_type, y_entrez, y_id, y_name):
    if x_type == "gene" and x_entrez:
        return x_entrez, y_type, y_id, y_name
    if y_type == "gene" and y_entrez:
        return y_entrez, x_type, x_id, x_name
    return "", "other", "", ""


def build_catalog(panel: dict[str, dict[str, object]], prime: PrimeSlice) -> dict[str, int]:
    keys = [f"gene:{info['entrez']}" for info in panel.values() if info.get("entrez")]
    keys.extend(f"pathway:{identifier}" for identifier in prime.pathway_names)
    keys.extend(f"disease:{identifier}" for identifier in prime.disease_names)
    keys.extend(f"drug:{identifier}" for identifier in prime.drug_names)
    return {key: index + 1 for index, key in enumerate(sorted(set(keys)))}


def validate_fused_graph(meta: dict[str, object], protein_ids: set[str]) -> list[str]:
    errors: list[str] = []
    node_types = list(meta["node_types"])
    if sum(kind == "patient" for kind in node_types) != 1:
        errors.append("fused graph must contain exactly one patient")
    leaked = set(meta.get("stored_fields", [])).intersection(OUTCOME_FIELDS)
    if leaked:
        errors.append(f"outcome fields stored on the fused graph: {sorted(leaked)}")
    for entrez in meta.get("grafted_entrez", []):
        if entrez not in protein_ids:
            errors.append(f"grafted protein {entrez} is not in the PrimeKG slice")
    return errors


def assemble_fused_graph(
    patient_id: str,
    clinical_row: dict | pd.Series,
    sample_genes: dict[str, dict[str, dict[str, object]]],
    prime: PrimeSlice,
    panel: dict[str, dict[str, object]],
    catalog: dict[str, int],
    label: int,
    max_hops: int = 2,
) -> tuple[Data, dict[str, object]]:
    """Build one homogeneous PyG graph. ``label`` is stored as ``data.y`` only."""
    altered: dict[str, dict[str, object]] = {}
    sample_edges: list[tuple[str, str, str]] = []
    for sample_id, genes in sample_genes.items():
        for entrez, info in genes.items():
            current = altered.setdefault(
                entrez,
                {"symbol": info["symbol"], "mut_count": 0, "amp": False, "homdel": False},
            )
            current["mut_count"] = int(current["mut_count"]) + int(info["mut_count"])
            current["amp"] = bool(current["amp"] or info["amp"])
            current["homdel"] = bool(current["homdel"] or info["homdel"])
            if int(info["mut_count"]) > 0:
                sample_edges.append((sample_id, entrez, "has_mutation"))
            if info["amp"]:
                sample_edges.append((sample_id, entrez, "has_high_level_amplification"))
            if info["homdel"]:
                sample_edges.append((sample_id, entrez, "has_homozygous_deletion"))

    seeds = set(altered)
    panel_entrez = {str(info["entrez"]) for info in panel.values() if info.get("entrez")}
    neighborhood = expand_primekg_neighborhood(seeds, prime, panel_entrez, max_hops)
    gene_ids = sorted(neighborhood.gene_hops, key=lambda entrez: (_symbol_for(entrez, altered, panel), entrez))
    pathway_ids = set(neighborhood.pathway_hops)
    drug_ids = set(neighborhood.drug_hops)
    disease_ids = set(neighborhood.disease_hops)
    context = seeds | neighborhood.ppi_partners

    nodes: list[dict[str, object]] = [
        {
            "id": f"patient:{patient_id}",
            "type": "patient",
            "name": patient_id,
            "emb": 0,
            "altered": False,
            "entrez": "",
        }
    ]
    for sample_id in sorted(sample_genes):
        nodes.append({"id": f"sample:{sample_id}", "type": "sample", "name": sample_id, "emb": 0, "altered": False, "entrez": ""})
    for entrez in gene_ids:
        symbol = _symbol_for(entrez, altered, panel)
        info = altered.get(entrez, {"mut_count": 0, "amp": False, "homdel": False})
        nodes.append(
            {
                "id": f"gene:{symbol}:{entrez}",
                "type": "gene",
                "name": symbol,
                "emb": catalog.get(f"gene:{entrez}", 0),
                "altered": entrez in seeds,
                "entrez": entrez,
                "info": info,
                "symbol": symbol,
            }
        )
    for pathway_id in sorted(pathway_ids, key=lambda identifier: prime.pathway_names.get(identifier, identifier)):
        nodes.append(
            {
                "id": f"pathway:{pathway_id}",
                "type": "pathway",
                "name": prime.pathway_names.get(pathway_id, pathway_id),
                "emb": catalog.get(f"pathway:{pathway_id}", 0),
                "altered": False,
                "entrez": "",
            }
        )
    for disease_id in sorted(disease_ids, key=lambda identifier: prime.disease_names.get(identifier, identifier)):
        nodes.append(
            {
                "id": f"disease:{disease_id}",
                "type": "disease",
                "name": prime.disease_names.get(disease_id, disease_id),
                "emb": catalog.get(f"disease:{disease_id}", 0),
                "altered": False,
                "entrez": "",
            }
        )
    for drug_id in sorted(drug_ids, key=lambda identifier: prime.drug_names.get(identifier, identifier)):
        nodes.append(
            {
                "id": f"drug:{drug_id}",
                "type": "drug",
                "name": prime.drug_names.get(drug_id, drug_id),
                "emb": catalog.get(f"drug:{drug_id}", 0),
                "altered": False,
                "entrez": "",
            }
        )

    index = {str(node["id"]): position for position, node in enumerate(nodes)}
    clinical = clinical_block(clinical_row)
    features = []
    for node in nodes:
        alteration = None
        block = clinical if node["type"] == "patient" else None
        if node["type"] == "gene":
            info = node["info"]
            symbol = str(node["symbol"])
            driver = 1.0 if panel.get(symbol, {}).get("is_known_driver") else 0.0
            alteration = np.array(
                [
                    1.0 if int(info["mut_count"]) > 0 else 0.0,
                    1.0 if info["amp"] else 0.0,
                    1.0 if info["homdel"] else 0.0,
                    math.log1p(int(info["mut_count"])),
                    driver,
                ],
                dtype=np.float32,
            )
        features.append(node_feature_vector(str(node["type"]), block, alteration))

    sources: list[int] = []
    targets: list[int] = []
    relations: list[str] = []
    reverses: list[bool] = []
    hops: list[int] = []

    def add_edge(src: int, dst: int, relation: str, hop: int) -> None:
        sources.extend((src, dst))
        targets.extend((dst, src))
        relations.extend((relation, relation))
        reverses.extend((False, True))
        hops.extend((int(hop), int(hop)))

    patient_index = 0
    for sample_id in sorted(sample_genes):
        add_edge(patient_index, index[f"sample:{sample_id}"], "has_sample", 0)
    for sample_id, entrez, relation in sample_edges:
        symbol = _symbol_for(entrez, altered, panel)
        add_edge(index[f"sample:{sample_id}"], index[f"gene:{symbol}:{entrez}"], relation, 0)
    gene_set = set(gene_ids)
    seen_ppi: set[tuple[str, str]] = set()
    for entrez in gene_ids:
        for partner in prime.ppi.get(entrez, set()):
            if partner not in gene_set:
                continue
            pair = tuple(sorted((entrez, partner)))
            if pair in seen_ppi:
                continue
            seen_ppi.add(pair)
            left = index[f"gene:{_symbol_for(pair[0], altered, panel)}:{pair[0]}"]
            right = index[f"gene:{_symbol_for(pair[1], altered, panel)}:{pair[1]}"]
            edge_hop = max(neighborhood.gene_hops.get(pair[0], 0), neighborhood.gene_hops.get(pair[1], 0), 1)
            add_edge(left, right, "protein_protein", edge_hop)
    for entrez in gene_ids:
        symbol = _symbol_for(entrez, altered, panel)
        gene_index = index[f"gene:{symbol}:{entrez}"]
        gene_hop = neighborhood.gene_hops.get(entrez, 0)
        for pathway_id in prime.gene_pathways.get(entrez, set()):
            if pathway_id in pathway_ids:
                add_edge(gene_index, index[f"pathway:{pathway_id}"], "pathway_protein", max(gene_hop, neighborhood.pathway_hops[pathway_id]))
        for drug_id in prime.gene_drugs.get(entrez, set()):
            if drug_id in drug_ids:
                add_edge(gene_index, index[f"drug:{drug_id}"], "drug_protein", max(gene_hop, neighborhood.drug_hops[drug_id]))
        for disease_id in prime.gene_diseases.get(entrez, set()):
            if disease_id in disease_ids:
                add_edge(gene_index, index[f"disease:{disease_id}"], "disease_protein", max(gene_hop, neighborhood.disease_hops[disease_id]))

    if not sources:
        # A labeled patient with no sample still needs a valid empty edge index.
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
    age = float(clinical[0])
    data = Data(
        x=torch.tensor(np.vstack(features), dtype=torch.float),
        edge_index=edge_index,
        emb_id=torch.tensor([int(node["emb"]) for node in nodes], dtype=torch.long),
        is_patient=torch.tensor([node["type"] == "patient" for node in nodes], dtype=torch.bool),
        focus=torch.tensor(
            [node["type"] == "patient" or bool(node["altered"]) for node in nodes],
            dtype=torch.bool,
        ),
        rel_code=torch.tensor([RELATION_TO_CODE[relation] for relation in relations], dtype=torch.long),
        hop=torch.tensor(hops, dtype=torch.long),
        is_reverse=torch.tensor(reverses, dtype=torch.bool),
        y=torch.tensor([float(label)], dtype=torch.float),
        raw_age=torch.tensor([age], dtype=torch.float),
        raw_clinical=torch.tensor(clinical, dtype=torch.float),
    )
    data.num_nodes = len(nodes)
    grafted = sorted(entrez for entrez in gene_ids if entrez not in seeds)
    meta = {
        "patient_id": patient_id,
        "node_ids": [str(node["id"]) for node in nodes],
        "node_types": [str(node["type"]) for node in nodes],
        "node_names": [str(node["name"]) for node in nodes],
        "altered": [bool(node["altered"]) for node in nodes],
        "relations": relations,
        "edge_hops": hops,
        "max_hops": int(max_hops),
        "grafted_entrez": grafted,
        "stored_fields": list(CLINICAL_FEATURE_FIELDS),
        "y": int(label),
    }
    errors = validate_fused_graph(meta, prime.protein_ids)
    if errors:
        raise ValueError(f"Invalid fused graph for {patient_id}: {errors}")
    return data, meta


def _symbol_for(entrez: str, altered: dict[str, dict[str, object]], panel: dict[str, dict[str, object]]) -> str:
    if entrez in altered:
        return str(altered[entrez]["symbol"])
    for symbol, info in panel.items():
        if info.get("entrez") == entrez:
            return symbol
    return entrez


def relations_from_codes(rel_code: torch.Tensor) -> list[str]:
    return [RELATION_NAMES[int(code)] for code in rel_code.tolist()]


def mlp_feature_matrix(clinical_rows: list[dict | pd.Series], sample_genes_by_patient: list[dict[str, dict[str, dict[str, object]]]], panel_symbols: list[str]) -> np.ndarray:
    rows = []
    for clinical_row, sample_genes in zip(clinical_rows, sample_genes_by_patient):
        flags = {symbol: [0.0, 0.0, 0.0] for symbol in panel_symbols}
        for genes in sample_genes.values():
            for info in genes.values():
                symbol = str(info["symbol"])
                if symbol not in flags:
                    continue
                if int(info["mut_count"]) > 0:
                    flags[symbol][0] = 1.0
                if info["amp"]:
                    flags[symbol][1] = 1.0
                if info["homdel"]:
                    flags[symbol][2] = 1.0
        multi_hot = np.array([bit for symbol in panel_symbols for bit in flags[symbol]], dtype=np.float32)
        rows.append(np.concatenate([clinical_block(clinical_row), multi_hot]))
    return np.vstack(rows) if rows else np.zeros((0, CLINICAL_DIM + 3 * len(panel_symbols)), dtype=np.float32)


def standardize_ages(graphs: list[Data], mlp_x: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    """Z-score the clinical block from the training fold only.

    Age is read from ``raw_age`` / the original MLP matrix, and the other clinical
    fields are read from ``raw_clinical``. Message-passing nodes other than the
    patient keep a zero clinical block. Outcome columns are not in this block.
    """
    if graphs and hasattr(graphs[0], "raw_clinical"):
        raw = np.vstack([graph.raw_clinical.detach().cpu().numpy() for graph in graphs]).astype(np.float64)
    else:
        raw = np.zeros((len(graphs), CLINICAL_DIM), dtype=np.float64)
        for index, graph in enumerate(graphs):
            age = float(graph.raw_age.view(-1)[0])
            raw[index, 0] = np.nan if math.isnan(age) else age
            raw[index, 1:] = graph.x[0, AGE_FEATURE_INDEX + 1 : AGE_FEATURE_INDEX + CLINICAL_DIM].detach().cpu().numpy()
    train = raw[np.asarray(train_indices)]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    filled = np.where(np.isnan(raw), mean, raw)
    scaled_clinical = ((filled - mean) / std).astype(np.float32)
    for graph, row in zip(graphs, scaled_clinical):
        graph.x[0, AGE_FEATURE_INDEX : AGE_FEATURE_INDEX + CLINICAL_DIM] = torch.tensor(row)
    scaled = mlp_x.copy()
    clinical = scaled[:, :CLINICAL_DIM].astype(np.float64)
    clinical_mean = np.nanmean(clinical[np.asarray(train_indices)], axis=0)
    clinical_std = np.nanstd(clinical[np.asarray(train_indices)], axis=0)
    clinical_mean = np.where(np.isfinite(clinical_mean), clinical_mean, 0.0)
    clinical_std = np.where(np.isfinite(clinical_std) & (clinical_std > 0), clinical_std, 1.0)
    clinical = np.where(np.isnan(clinical), clinical_mean, clinical)
    scaled[:, :CLINICAL_DIM] = ((clinical - clinical_mean) / clinical_std).astype(np.float32)
    return scaled


@dataclass
class FusedCohort:
    patient_ids: list[str]
    y: np.ndarray
    graphs: list[Data]
    metas: list[dict[str, object]]
    mlp_x: np.ndarray
    endpoint: str
    catalog_size: int


def _sample_gene_table(mutations: pd.DataFrame, cna: pd.DataFrame, panel: dict[str, dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    by_sample: dict[str, dict[str, dict[str, object]]] = {}

    def slot(sample_id: str, symbol: str) -> dict[str, object] | None:
        info = panel.get(symbol)
        if info is None or not info.get("entrez"):
            return None
        genes = by_sample.setdefault(sample_id, {})
        return genes.setdefault(
            str(info["entrez"]),
            {"symbol": symbol, "mut_count": 0, "amp": False, "homdel": False},
        )

    if not mutations.empty:
        for record in mutations.itertuples(index=False):
            symbol = str(record.Hugo_Symbol).upper()
            sample_id = str(record.Tumor_Sample_Barcode)
            gene = slot(sample_id, symbol)
            if gene is not None:
                gene["mut_count"] = int(gene["mut_count"]) + 1
    if not cna.empty:
        for record in cna.itertuples(index=False):
            symbol = str(record.Hugo_Symbol).upper()
            gene = slot(str(record.SAMPLE_ID), symbol)
            if gene is None:
                continue
            if int(record.cna_value) == 2:
                gene["amp"] = True
            elif int(record.cna_value) == -2:
                gene["homdel"] = True
    return by_sample


def build_cohort(
    dataset_dir: Path,
    labels: pd.DataFrame,
    prime_frame: pd.DataFrame,
    panel: dict[str, dict[str, object]] | None = None,
    fusion: FusionConfig | None = None,
) -> FusedCohort:
    panel = panel or load_gene_panel()
    fusion = fusion or load_fusion_config()
    panel_symbols = sorted(panel)
    prime = PrimeSlice.from_frame(prime_frame)
    catalog = build_catalog(panel, prime)
    clinical = load_clinical_data(dataset_dir)
    linked, _ = link_patients_to_samples(clinical)
    labeled_ids = set(labels["PATIENT_ID"].astype(str))
    linked = linked[linked["PATIENT_ID"].isin(labeled_ids)].copy()
    sample_ids = linked["SAMPLE_ID"].astype(str).tolist()
    mutations, _ = load_mutations(dataset_dir, gene_panel=set(panel))
    mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(sample_ids)].copy()
    policy = load_feature_policy().get("cna_policy", {})
    allowed = policy.get("include_values", [-2, 2])
    cna, _ = load_cna_for_samples(dataset_dir, sample_ids, gene_panel=set(panel), allowed_values=allowed)
    by_sample = _sample_gene_table(mutations, cna, panel)
    patients = clinical.patients.set_index("PATIENT_ID")
    label_map = labels.set_index("PATIENT_ID")["y"].astype(int).to_dict()
    graphs: list[Data] = []
    metas: list[dict[str, object]] = []
    clinical_rows = []
    sample_gene_rows = []
    patient_ids: list[str] = []
    targets: list[int] = []
    for patient_id in sorted(labeled_ids):
        samples = linked.loc[linked["PATIENT_ID"] == patient_id, "SAMPLE_ID"].astype(str).tolist()
        if patient_id not in patients.index or not samples:
            continue
        row = patients.loc[patient_id]
        sample_genes = {sample_id: by_sample.get(sample_id, {}) for sample_id in samples}
        graph, meta = assemble_fused_graph(
            patient_id,
            row,
            sample_genes,
            prime,
            panel,
            catalog,
            int(label_map[patient_id]),
            max_hops=fusion.candidate_hops,
        )
        graphs.append(graph)
        metas.append(meta)
        clinical_rows.append(row)
        sample_gene_rows.append(sample_genes)
        patient_ids.append(patient_id)
        targets.append(int(label_map[patient_id]))
    mlp_x = mlp_feature_matrix(clinical_rows, sample_gene_rows, panel_symbols)
    endpoint = str(labels["endpoint"].iloc[0]) if len(labels) else "pfs"
    return FusedCohort(
        patient_ids=patient_ids,
        y=np.array(targets, dtype=np.int64),
        graphs=graphs,
        metas=metas,
        mlp_x=mlp_x,
        endpoint=endpoint,
        catalog_size=len(catalog) + 1,
    )
