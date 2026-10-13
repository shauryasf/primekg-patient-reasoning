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
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, LayerNorm

class AdvancedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, arch='gcn'):
        super().__init__()
        self.arch = arch
        if arch == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif arch == 'graphsage':
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        elif arch == 'gat':
            self.conv1 = GATConv(in_channels, hidden, heads=2, concat=False)
            self.conv2 = GATConv(hidden, hidden, heads=2, concat=False)
            
        self.norm = LayerNorm(hidden)
        self.lin1 = torch.nn.Linear(hidden, hidden // 2)
        self.lin2 = torch.nn.Linear(hidden // 2, out_channels)

    def forward(self, x, edge_index, batch):
        x = x.float()
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.lin2(x)

def run():
    print("Loading Preprocessed Data...")
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
    
    X = expr.T.values
    y = labels['label'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_oof_preds = []
    fold_metrics = []
    
    # WE ONLY WANT GRAPH MODELS FUSED WITH PRIME_KG!
    models = ['gcn_PrimeKG', 'graphsage_PrimeKG', 'gat_PrimeKG']
    
    for model_name in models:
        print(f"Training {model_name}...")
        
        oof_preds = np.zeros(len(y))
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Apply SMOTE to the training fold
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            # Build PyG datasets dynamically for the SMOTE augmented data
            train_dataset = []
            for i in range(X_train_res.shape[0]):
                x_feats = torch.tensor(X_train_res[i].reshape(-1, 1), dtype=torch.float)
                train_dataset.append(Data(x=x_feats, edge_index=edge_index, y=torch.tensor([y_train_res[i]], dtype=torch.float)))
            
            test_dataset = []
            for i in range(X_test.shape[0]):
                x_feats = torch.tensor(X_test[i].reshape(-1, 1), dtype=torch.float)
                test_dataset.append(Data(x=x_feats, edge_index=edge_index, y=torch.tensor([y_test[i]], dtype=torch.float)))
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            torch.manual_seed(42)
            arch = model_name.split('_')[0]
            model = AdvancedGNN(1, 32, 1, arch=arch)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            
            criterion = torch.nn.BCEWithLogitsLoss()
            
            for epoch in range(40):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out.view(-1), batch.y)
                    loss.backward()
                    optimizer.step()
                    
            model.eval()
            with torch.no_grad():
                fold_preds = []
                for batch in test_loader:
                    out = model(batch.x, batch.edge_index, batch.batch)
                    fold_preds.extend(torch.sigmoid(out).view(-1).numpy())
            oof_preds[test_idx] = np.array(fold_preds)
            
            p_test = oof_preds[test_idx]
            p_bin = (p_test > 0.5).astype(int)
            
            fold_metrics.append({
                'Model': model_name,
                'Fold': fold + 1,
                'AUROC': roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else 0.5,
                'F1_Positive': f1_score(y_test, p_bin, pos_label=1, zero_division=0),
                'F1_Weighted': f1_score(y_test, p_bin, average='weighted', zero_division=0)
            })
            
    df_folds = pd.DataFrame(fold_metrics)
    df_folds.to_csv('outputs/gse2034_final/graph_smote_fold_metrics.csv', index=False)
    
    agg = df_folds.groupby('Model').agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    agg.to_csv('outputs/gse2034_final/graph_smote_aggregate_metrics.csv', index=False)
    
    print("Done! Graph-only SMOTE Results saved.")

if __name__ == "__main__":
    run()
