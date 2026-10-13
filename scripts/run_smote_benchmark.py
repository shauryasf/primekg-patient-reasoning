import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv

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
    
    models = ['LogisticRegression_SMOTE', 'RandomForest_SMOTE', 'gcn_SMOTE', 'graphsage_SMOTE']
    
    for model_name in models:
        print(f"Training {model_name}...")
        
        oof_preds = np.zeros(len(y))
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Apply SMOTE to the training fold
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            if 'LogisticRegression' in model_name or 'RandomForest' in model_name:
                if 'LogisticRegression' in model_name:
                    clf = LogisticRegression(max_iter=2000, random_state=42)
                else:
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                clf.fit(X_train_res, y_train_res)
                preds = clf.predict_proba(X_test)[:, 1]
                oof_preds[test_idx] = preds
                
            else:
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
                arch = 'gcn' if 'gcn' in model_name else 'graphsage'
                model = SimpleGNN(1, 16, 1, arch=arch)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
                
                # Balanced classes after SMOTE, no need for pos_weight
                criterion = torch.nn.BCEWithLogitsLoss()
                
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
                
            p_test = oof_preds[test_idx]
            p_bin = (p_test > 0.5).astype(int)
            
            fold_metrics.append({
                'Model': model_name,
                'Fold': fold + 1,
                'AUROC': roc_auc_score(y_test, p_test),
                'F1_Positive': f1_score(y_test, p_bin, pos_label=1),
                'F1_Weighted': f1_score(y_test, p_bin, average='weighted')
            })
            
    df_folds = pd.DataFrame(fold_metrics)
    df_folds.to_csv('outputs/gse2034_final/smote_fold_metrics.csv', index=False)
    
    agg = df_folds.groupby('Model').agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    agg.to_csv('outputs/gse2034_final/smote_aggregate_metrics.csv', index=False)
    
    print("Done! SMOTE Results saved.")

if __name__ == "__main__":
    run()
