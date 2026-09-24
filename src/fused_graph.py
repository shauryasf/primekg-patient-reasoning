"""Construct Fused Knowledge Graphs (Patient Subgraph + PrimeKG Biomedical Network)
and convert them into PyTorch Geometric Data objects for GNN training.
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .target_labels import load_target_outcomes

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "brca_tcga_pan_can_atlas_2018"
OUTPUTS_DIR = ROOT / "outputs"
PRIMEKG_PATH = ROOT / "data" / "primekg_cancer_subnetwork.csv"

# Pre-defined vocabularies for standard node and edge categorical features
NODE_TYPES = ["patient", "sample", "gene", "pathway", "drug", "disease", "clinical_concept"]
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

EDGE_RELATIONS = [
    "has_sample",
    "has_mutation",
    "has_high_level_amplification",
    "has_homozygous_deletion",
    "has_subtype",
    "has_tumor_stage",
    "has_path_m_stage",
    "has_oncotree_code",
    "has_sample_type",
    "protein_protein",
    "pathway_protein",
    "drug_protein",
    "disease_protein"
]
EDGE_REL_TO_IDX = {r: i for i, r in enumerate(EDGE_RELATIONS)}

SUBTYPES = ["BRCA_Basal", "BRCA_Her2", "BRCA_LumA", "BRCA_LumB", "BRCA_Normal", "Unknown"]
STAGES = ["STAGE I", "STAGE II", "STAGE III", "STAGE IV", "STAGE X", "Unknown"]

def load_primekg_lookup(primekg_csv: Path) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, str, str]]]]:
    """Load PrimeKG subnetwork and build gene-indexed lookup table for fast graph fusion."""
    if not primekg_csv.exists():
        raise FileNotFoundError(f"PrimeKG subnetwork file not found at {primekg_csv}")
        
    df = pd.read_csv(primekg_csv)
    # Standardize gene names
    gene_to_edges: Dict[str, List[Tuple[str, str, str]]] = {}
    
    for _, row in df.iterrows():
        rel = str(row["relation"]).strip()
        x_name = str(row["x_name"]).strip()
        y_name = str(row["y_name"]).strip()
        x_type = str(row["x_type"]).strip()
        y_type = str(row["y_type"]).strip()
        
        # We index by gene
        if x_type == "gene/protein":
            if x_name not in gene_to_edges:
                gene_to_edges[x_name] = []
            gene_to_edges[x_name].append((x_name, rel, y_name, y_type))
            
        if y_type == "gene/protein":
            if y_name not in gene_to_edges:
                gene_to_edges[y_name] = []
            gene_to_edges[y_name].append((x_name, rel, y_name, x_type))
            
    return df, gene_to_edges

def build_fused_patient_graphs(
    all_tuples_csv: Path,
    primekg_csv: Path,
    target_df: pd.DataFrame,
    max_biomedical_hops: int = 1,
    max_primekg_edges_per_gene: int = 10
) -> List[Data]:
    """Fuse patient-specific graphs with PrimeKG biomedical knowledge into PyTorch Geometric Data objects."""
    print("Loading patient tuples...")
    tuples_df = pd.read_csv(all_tuples_csv)
    
    print("Loading PrimeKG biomedical subnetwork...")
    _, primekg_gene_lookup = load_primekg_lookup(primekg_csv)
    
    # Map patient targets
    target_map = {}
    for _, row in target_df.iterrows():
        if pd.notna(row["progression_event"]):
            target_map[row["patient_id"]] = int(row["progression_event"])
            
    # Group patient tuples by patient
    patient_groups = {}
    for _, row in tuples_df.iterrows():
        # Head ID usually has prefix 'patient:TCGA-...'
        h_id = str(row["head_id"])
        if h_id.startswith("patient:"):
            pid = h_id.replace("patient:", "")
        elif str(row["head_type"]) == "sample":
            # Extract patient from sample ID (e.g. TCGA-A2-A0YE-01 -> TCGA-A2-A0YE)
            pid = "-".join(h_id.replace("sample:", "").split("-")[:3])
        else:
            continue
            
        if pid not in patient_groups:
            patient_groups[pid] = []
        patient_groups[pid].append(row)
        
    print(f"Loaded {len(patient_groups)} patient subgraphs. Fusing with PrimeKG...")
    
    pyg_data_list = []
    
    for pid, p_tuples in patient_groups.items():
        if pid not in target_map:
            continue  # Skip if target is missing
            
        y_label = target_map[pid]
        
        # Build local NetworkX graph for this patient
        G = nx.MultiDiGraph()
        patient_genes = set()
        
        # Track clinical properties for node feature vector
        patient_age = 60.0  # default median
        patient_subtype = "Unknown"
        patient_stage = "Unknown"
        
        for t in p_tuples:
            h = str(t["head_id"])
            h_type = str(t["head_type"]).lower()
            rel = str(t["relation"]).strip()
            v = str(t["tail_id"])
            v_type = str(t["tail_type"]).lower()
            v_name = str(t.get("tail_name", v))
            
            G.add_node(h, node_type=h_type, name=h)
            G.add_node(v, node_type=v_type, name=v_name)
            G.add_edge(h, v, relation=rel)
            
            if v_type == "gene":
                gene_symbol = v_name
                patient_genes.add(gene_symbol)
                
            if rel == "has_subtype":
                patient_subtype = v_name
            elif rel == "has_tumor_stage":
                patient_stage = v_name
                
        # Now FUSE with PrimeKG around patient_genes
        for gene in patient_genes:
            if gene in primekg_gene_lookup:
                # Add top biological interactions from PrimeKG
                prime_edges = primekg_gene_lookup[gene][:max_primekg_edges_per_gene]
                for src_name, rel, tgt_name, other_type in prime_edges:
                    src_node = f"gene:{src_name}" if src_name == gene else f"biomedical:{src_name}"
                    tgt_node = f"gene:{tgt_name}" if tgt_name == gene else f"biomedical:{tgt_name}"
                    
                    b_type = "pathway" if "pathway" in other_type else ("drug" if "drug" in other_type else ("disease" if "disease" in other_type else "gene"))
                    
                    if not G.has_node(src_node):
                        G.add_node(src_node, node_type="gene" if src_name == gene else b_type, name=src_name)
                    if not G.has_node(tgt_node):
                        G.add_node(tgt_node, node_type="gene" if tgt_name == gene else b_type, name=tgt_name)
                        
                    norm_rel = "protein_protein" if rel in ["protein_protein", "ppi"] else (
                        "pathway_protein" if "pathway" in rel else (
                        "drug_protein" if "drug" in rel else (
                        "disease_protein" if "disease" in rel else "protein_protein")))
                        
                    G.add_edge(src_node, tgt_node, relation=norm_rel)
                    
        # Convert G to PyG Data object
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)
        
        # Build node feature matrix X
        # Node Type One-Hot (7 dim) + Clinical Subtype One-Hot (6 dim) + Stage One-Hot (6 dim) = 19 dim
        feat_dim = len(NODE_TYPES) + len(SUBTYPES) + len(STAGES)
        X = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        
        for n, idx in node_to_idx.items():
            ntype = G.nodes[n].get("node_type", "gene").lower()
            if ntype not in NODE_TYPE_TO_IDX:
                ntype = "gene"
            X[idx, NODE_TYPE_TO_IDX[ntype]] = 1.0
            
            # If patient node, embed subtype and stage
            if ntype == "patient":
                st_idx = SUBTYPES.index(patient_subtype) if patient_subtype in SUBTYPES else SUBTYPES.index("Unknown")
                X[idx, len(NODE_TYPES) + st_idx] = 1.0
                
                stage_idx = 0
                for s_i, s_name in enumerate(STAGES):
                    if s_name in patient_stage:
                        stage_idx = s_i
                        break
                X[idx, len(NODE_TYPES) + len(SUBTYPES) + stage_idx] = 1.0
                
        # Build edge index and edge attributes
        edge_indices = []
        edge_types = []
        for u, v, data in G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            rel = data.get("relation", "protein_protein")
            rel_idx = EDGE_REL_TO_IDX.get(rel, EDGE_REL_TO_IDX["protein_protein"])
            
            edge_indices.append([u_idx, v_idx])
            edge_types.append(rel_idx)
            
            # Add reverse edge for undirected message passing
            edge_indices.append([v_idx, u_idx])
            edge_types.append(rel_idx)
            
        if len(edge_indices) == 0:
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
            edge_type_tensor = torch.zeros(0, dtype=torch.long)
        else:
            edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
            
        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=edge_index_tensor,
            edge_type=edge_type_tensor,
            y=torch.tensor([y_label], dtype=torch.long),
            patient_id=pid,
            num_nodes=num_nodes
        )
        pyg_data_list.append(data)
        
    print(f"Successfully constructed {len(pyg_data_list)} Fused PyG patient graph objects.")
    return pyg_data_list

if __name__ == "__main__":
    from .target_labels import load_target_outcomes
    targets = load_target_outcomes(DATA_DIR)
    fused_graphs = build_fused_patient_graphs(
        all_tuples_csv=OUTPUTS_DIR / "all_patients_tuples.csv",
        primekg_csv=PRIMEKG_PATH,
        target_df=targets
    )
    print(f"Sample graph: {fused_graphs[0]}")
    print(f"Sample node features shape: {fused_graphs[0].x.shape}")
    print(f"Sample edge index shape: {fused_graphs[0].edge_index.shape}")
    print(f"Sample label: {fused_graphs[0].y}")
