import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool

class FastRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, num_relations=1):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden, num_relations)
        self.conv2 = RGCNConv(hidden, hidden, num_relations)
        self.lin = torch.nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, edge_type, batch):
        x = x.float()
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)
        return self.lin(x)

def run():
    print("Loading Data for Ultra-Fast R-GCN...")
    expr = pd.read_csv('data/processed/gse2034/expression.csv', index_col=0)
    labels = pd.read_csv('data/processed/gse2034/labels.csv', index_col=0)
    
    expr.index = expr.index.astype(str)
    
    kg = pd.read_csv('data/primekg_cancer_subnetwork.csv')
    kg_genes = kg[(kg['x_type'] == 'gene/protein') & (kg['y_type'] == 'gene/protein')].copy()
    
    kg_genes['x_id'] = kg_genes['x_index'].astype(str)
    kg_genes['y_id'] = kg_genes['y_index'].astype(str)
    
    panel = set(kg_genes['x_id']).union(set(kg_genes['y_id']))
    overlap = list(set(expr.index).intersection(panel))
    
    expr = expr.loc[overlap]
    expr = expr[~expr.index.duplicated(keep='first')]
    node_to_idx = {n: i for i, n in enumerate(overlap)}
    valid_edges = kg_genes[(kg_genes['x_id'].isin(overlap)) & (kg_genes['y_id'].isin(overlap))]
    
    edge_index = torch.tensor([
        valid_edges['x_id'].map(node_to_idx).values,
        valid_edges['y_id'].map(node_to_idx).values
    ], dtype=torch.long)
    
    # All edges are 'ppi' in this subgraph, so edge type is 0
    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
    
    X = expr.T.values
    y = labels['label'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    oof_preds = np.zeros(len(y))
    print("Training R-GCN (15 Epochs)...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        train_dataset = []
        for i in range(X_train_res.shape[0]):
            x_feats = torch.tensor(X_train_res[i].reshape(-1, 1), dtype=torch.float)
            # attach edge_type to data
            data = Data(x=x_feats, edge_index=edge_index, y=torch.tensor([y_train_res[i]], dtype=torch.float))
            data.edge_type = edge_type
            train_dataset.append(data)
            
        test_dataset = []
        for i in range(X_test.shape[0]):
            x_feats = torch.tensor(X_test[i].reshape(-1, 1), dtype=torch.float)
            data = Data(x=x_feats, edge_index=edge_index, y=torch.tensor([y_test[i]], dtype=torch.float))
            data.edge_type = edge_type
            test_dataset.append(data)
            
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        torch.manual_seed(42)
        model = FastRGCN(1, 16, 1, num_relations=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        model.train()
        for epoch in range(15):  # Super fast training
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                loss = criterion(out.view(-1), batch.y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            fold_preds = []
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                fold_preds.extend(torch.sigmoid(out).view(-1).numpy())
        
        p_test = np.array(fold_preds)
        p_bin = (p_test > 0.5).astype(int)
        
        fold_metrics.append({
            'Model': 'rgcn_SMOTE',
            'Fold': fold + 1,
            'AUROC': roc_auc_score(y_test, p_test),
            'F1_Positive': f1_score(y_test, p_bin, pos_label=1, zero_division=0),
            'F1_Weighted': f1_score(y_test, p_bin, average='weighted', zero_division=0)
        })
        
    new_metrics = pd.DataFrame(fold_metrics)
    agg_new = new_metrics.groupby('Model').agg(['mean', 'std']).reset_index()
    agg_new.columns = ['_'.join(col).strip('_') for col in agg_new.columns.values]
    
    # Load old metrics and filter out tabular
    old = pd.read_csv('outputs/gse2034_final/smote_aggregate_metrics.csv')
    old_graph = old[old['Model'].isin(['gcn_SMOTE', 'graphsage_SMOTE'])]
    
    # Combine
    final_df = pd.concat([old_graph, agg_new], ignore_index=True)
    final_df.to_csv('outputs/gse2034_final/graph_only_metrics.csv', index=False)
    print("Successfully built graph-only metrics with R-GCN included!")

if __name__ == "__main__":
    run()
