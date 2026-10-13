import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, arch='gcn'):
        super().__init__()
        self.arch = arch
        if arch == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif arch == 'graphsage':
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        elif arch == 'gatv2':
            self.conv1 = GATv2Conv(in_channels, hidden)
            self.conv2 = GATv2Conv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        from torch_geometric.nn import global_mean_pool
        x = global_mean_pool(x, batch)
        return self.lin(x)

def run():
    print("Loading GSE2034 Preprocessed Data...")
    expr = pd.read_csv('data/processed/gse2034/expression.csv', index_col=0)
    labels = pd.read_csv('data/processed/gse2034/labels.csv', index_col=0)
    
    kg = pd.read_csv('data/primekg_cancer_subnetwork.csv')
    kg_genes = kg[(kg['x_type'] == 'gene') & (kg['y_type'] == 'gene')].copy()
    
    # Get IDs as strings
    kg_genes['x_id'] = kg_genes['x_index'].astype(str).str.replace('.0', '', regex=False)
    kg_genes['y_id'] = kg_genes['y_index'].astype(str).str.replace('.0', '', regex=False)
    
    panel = set(kg_genes['x_id']).union(set(kg_genes['y_id']))
    
    expr.index = expr.index.astype(str).str.replace('.0', '', regex=False)
    overlap = list(set(expr.index).intersection(panel))
    
    expr = expr[~expr.index.duplicated(keep='first')]
    expr = expr.loc[overlap]
    
    print(f"Genes overlapping with panel: {len(overlap)}")
    if len(overlap) == 0:
        print("No overlapping genes found. Cannot train graph models.")
        return
        
    node_to_idx = {n: i for i, n in enumerate(overlap)}
    valid_edges = kg_genes[(kg_genes['x_id'].isin(overlap)) & (kg_genes['y_id'].isin(overlap))]
    
    edge_index = torch.tensor([
        valid_edges['x_id'].map(node_to_idx).values,
        valid_edges['y_id'].map(node_to_idx).values
    ], dtype=torch.long)
    
    print(f"Graph constructed with {edge_index.size(1)} edges.")
    
    X = expr.T.values
    y = labels['label'].values
    
    dataset = []
    for i in range(X.shape[0]):
        x_feats = torch.tensor(X[i].reshape(-1, 1), dtype=torch.float)
        dataset.append(Data(x=x_feats, edge_index=edge_index, y=torch.tensor([y[i]], dtype=torch.float)))
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for model_name in ['LogisticRegression', 'RandomForest', 'gcn', 'graphsage']:
        print(f"Training {model_name}...")
        metrics = {'auroc': [], 'auprc': [], 'f1_pos': [], 'macro_f1': []}
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            if model_name in ['LogisticRegression', 'RandomForest']:
                clf = LogisticRegression(max_iter=1000) if model_name == 'LogisticRegression' else RandomForestClassifier(n_estimators=100)
                clf.fit(X[train_idx], y[train_idx])
                preds = clf.predict_proba(X[test_idx])[:, 1]
                preds_bin = clf.predict(X[test_idx])
            else:
                train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=True)
                test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=32)
                
                model = SimpleGNN(1, 16, 1, arch=model_name)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                criterion = torch.nn.BCEWithLogitsLoss()
                
                model.train()
                for _ in range(10):
                    for batch in train_loader:
                        optimizer.zero_grad()
                        loss = criterion(model(batch.x, batch.edge_index, batch.batch).view(-1), batch.y)
                        loss.backward()
                        optimizer.step()
                        
                model.eval()
                preds = []
                with torch.no_grad():
                    for batch in test_loader:
                        preds.extend(torch.sigmoid(model(batch.x, batch.edge_index, batch.batch)).view(-1).numpy())
                preds = np.array(preds)
                preds_bin = (preds > 0.5).astype(int)
            
            y_test = y[test_idx]
            metrics['auroc'].append(roc_auc_score(y_test, preds))
            metrics['auprc'].append(average_precision_score(y_test, preds))
            metrics['f1_pos'].append(f1_score(y_test, preds_bin, pos_label=1))
            metrics['macro_f1'].append(f1_score(y_test, preds_bin, average='macro'))
            
        results.append({
            'Model': model_name,
            'AUROC': f"{np.mean(metrics['auroc']):.3f} ± {np.std(metrics['auroc']):.3f}",
            'AUPRC': f"{np.mean(metrics['auprc']):.3f} ± {np.std(metrics['auprc']):.3f}",
            'Positive F1': f"{np.mean(metrics['f1_pos']):.3f} ± {np.std(metrics['f1_pos']):.3f}",
            'Macro F1': f"{np.mean(metrics['macro_f1']):.3f} ± {np.std(metrics['macro_f1']):.3f}"
        })
        
    os.makedirs('outputs/gse2034_benchmark', exist_ok=True)
    pd.DataFrame(results).to_csv('outputs/gse2034_benchmark/results.csv', index=False)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    run()
