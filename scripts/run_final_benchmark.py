import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F

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
        self.lin = torch.nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        from torch_geometric.nn import global_mean_pool
        x = global_mean_pool(x, batch)
        return self.lin(x)

def run():
    print("Loading Preprocessed Data...")
    expr = pd.read_csv('data/processed/gse2034/expression.csv', index_col=0)
    labels = pd.read_csv('data/processed/gse2034/labels.csv', index_col=0)
    
    # Save a copy as requested
    expr.to_csv('outputs/gse2034_final/processed_expression.csv')
    labels.to_csv('outputs/gse2034_final/labels.csv')
    
    expr.index = expr.index.astype(str)
    
    print("Loading PrimeKG structure...")
    kg = pd.read_csv('data/primekg_cancer_subnetwork.csv')
    kg_genes = kg[(kg['x_type'] == 'gene/protein') & (kg['y_type'] == 'gene/protein')].copy()
    
    kg_genes['x_id'] = kg_genes['x_index'].astype(str)
    kg_genes['y_id'] = kg_genes['y_index'].astype(str)
    
    panel = set(kg_genes['x_id']).union(set(kg_genes['y_id']))
    overlap = list(set(expr.index).intersection(panel))
    
    # Diagnostic save
    diag = pd.DataFrame({
        'Metric': ['Total GSE2034 Genes', 'Total PrimeKG Genes', 'Intersecting Genes'],
        'Count': [len(expr.index), len(panel), len(overlap)]
    })
    diag.to_csv('outputs/gse2034_final/primekg_overlap_diagnostics.csv', index=False)
    
    expr = expr.loc[overlap]
    expr = expr[~expr.index.duplicated(keep='first')]
    
    node_to_idx = {n: i for i, n in enumerate(overlap)}
    valid_edges = kg_genes[(kg_genes['x_id'].isin(overlap)) & (kg_genes['y_id'].isin(overlap))]
    
    edge_index = torch.tensor([
        valid_edges['x_id'].map(node_to_idx).values,
        valid_edges['y_id'].map(node_to_idx).values
    ], dtype=torch.long)
    
    print(f"Graph constructed with {edge_index.size(1)} edges over {len(overlap)} nodes.")
    
    X = expr.T.values # Shape: (Patients, Genes)
    y = labels['label'].values
    
    # Pre-build graphs
    dataset = []
    for i in range(X.shape[0]):
        x_feats = torch.tensor(X[i].reshape(-1, 1), dtype=torch.float)
        dataset.append(Data(x=x_feats, edge_index=edge_index, y=torch.tensor([y[i]], dtype=torch.float)))
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_oof_preds = []
    fold_metrics = []
    
    models = ['MajorityBaseline', 'LogisticRegression', 'RandomForest', 'gcn', 'graphsage']
    
    for model_name in models:
        print(f"Training {model_name}...")
        
        oof_preds = np.zeros(len(y))
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            if model_name in ['MajorityBaseline', 'LogisticRegression', 'RandomForest']:
                if model_name == 'MajorityBaseline':
                    clf = DummyClassifier(strategy='most_frequent')
                elif model_name == 'LogisticRegression':
                    clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
                else:
                    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
                    
                clf.fit(X[train_idx], y[train_idx])
                if model_name == 'MajorityBaseline':
                    preds = clf.predict_proba(X[test_idx])[:, 1]
                else:
                    preds = clf.predict_proba(X[test_idx])[:, 1]
                oof_preds[test_idx] = preds
                
            else:
                train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=True)
                test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=32)
                
                # Use a fixed random seed
                torch.manual_seed(42)
                model = SimpleGNN(1, 16, 1, arch=model_name)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
                
                # Balanced BCE Loss
                pos_weight = torch.tensor([(len(train_idx) - sum(y[train_idx])) / sum(y[train_idx])])
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                model.train()
                for epoch in range(30):
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
                
            # Compute fold metrics
            y_test = y[test_idx]
            p_test = oof_preds[test_idx]
            p_bin = (p_test > 0.5).astype(int)
            
            try: auroc = roc_auc_score(y_test, p_test)
            except: auroc = 0.5
            try: auprc = average_precision_score(y_test, p_test)
            except: auprc = 0.0
            
            fold_metrics.append({
                'Model': model_name,
                'Fold': fold + 1,
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1_Positive': f1_score(y_test, p_bin, pos_label=1),
                'Precision': precision_score(y_test, p_bin, pos_label=1, zero_division=0),
                'Recall': recall_score(y_test, p_bin, pos_label=1),
                'Specificity': recall_score(y_test, p_bin, pos_label=0),
                'Balanced_Accuracy': accuracy_score(y_test, p_bin) # Will refine below
            })
            
        # Save OOF predictions
        for i, val in enumerate(oof_preds):
            all_oof_preds.append({
                'Patient_Idx': i,
                'Model': model_name,
                'True_Label': y[i],
                'Predicted_Prob': val
            })
            
    # Save Fold Metrics
    df_folds = pd.DataFrame(fold_metrics)
    df_folds['Balanced_Accuracy'] = (df_folds['Recall'] + df_folds['Specificity']) / 2
    df_folds.to_csv('outputs/gse2034_final/fold_metrics.csv', index=False)
    
    # Aggregate Metrics
    agg = df_folds.groupby('Model').agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    agg.to_csv('outputs/gse2034_final/aggregate_metrics.csv', index=False)
    
    pd.DataFrame(all_oof_preds).to_csv('outputs/gse2034_final/oof_predictions.csv', index=False)
    print("Done! Results saved to outputs/gse2034_final/")

if __name__ == "__main__":
    run()
