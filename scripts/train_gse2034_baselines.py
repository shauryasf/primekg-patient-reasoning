import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def run():
    print("Loading GSE2034 Preprocessed Data...")
    expr = pd.read_csv('data/processed/gse2034/expression.csv', index_col=0)
    labels = pd.read_csv('data/processed/gse2034/labels.csv', index_col=0)
    
    X = expr.T.values
    y = labels['label'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for model_name in ['LogisticRegression', 'RandomForest']:
        print(f"Training {model_name}...")
        metrics = {'auroc': [], 'auprc': [], 'f1_pos': [], 'macro_f1': []}
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            clf = LogisticRegression(max_iter=1000) if model_name == 'LogisticRegression' else RandomForestClassifier(n_estimators=100)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict_proba(X[test_idx])[:, 1]
            preds_bin = clf.predict(X[test_idx])
            
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
    print("Graph Models (GCN, GraphSAGE) blocked due to 0-node overlap between mapped Entrez IDs and current primekg_cancer_subnetwork.csv.")

if __name__ == "__main__":
    run()
