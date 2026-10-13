import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from src.dataset_fusion import CLINICAL_DIM, build_cohort, load_gene_panel, load_primekg_slice
from src.labels import load_recurrence_labels
from src.survival import load_survival_targets, align_survival_targets

def main():
    dataset_dir = Path("brca_tcga_pan_can_atlas_2018")
    out_dir = Path("outputs/external_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading TCGA-BRCA cohort for CoxPH survival baseline...")
    # Load required elements to build the cohort
    panel = load_gene_panel(Path("config/gene_panel_500.json"))
    from src.downloads import ensure_primekg_csv
    prime_csv = ensure_primekg_csv(Path("data"))
    panel_entrez = set(panel)
    prime_frame = load_primekg_slice(prime_csv, panel_entrez)
    labels = load_recurrence_labels(dataset_dir, "pfs_36m") 
    cohort = build_cohort(dataset_dir, labels, prime_frame, panel, fusion=None)
    
    # Load the actual continuous survival targets
    targets = load_survival_targets(dataset_dir, "pfs")
    time, event, usable = align_survival_targets(targets, cohort.patient_ids)
    
    # Clinical variables: the first CLINICAL_DIM features in the tabular matrix
    clinical_features = cohort.mlp_x[:, :CLINICAL_DIM].astype(np.float64)
    
    # Filter down to only usable patients for survival analysis
    X_all = clinical_features[usable]
    time_all = time[usable]
    event_all = event[usable]
    
    print(f"Usable survival patients: {len(X_all)} with {int(event_all.sum())} events.")
    
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    
    train_c_indices = []
    test_c_indices = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        T_train, T_test = time_all[train_idx], time_all[test_idx]
        E_train, E_test = event_all[train_idx], event_all[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
        X_test_scaled = scaler.transform(np.nan_to_num(X_test, nan=0.0))
        
        # Drop zero-variance columns (which become NaNs during scaling)
        valid_cols = ~np.isnan(X_train_scaled).any(axis=0)
        X_train_scaled = X_train_scaled[:, valid_cols]
        X_test_scaled = X_test_scaled[:, valid_cols]
        
        # Apply PCA to prevent complete separation and collinearity in sparse clinical features
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Pack into DataFrames for lifelines
        cols = [f"pc_{i}" for i in range(5)]
        df_train = pd.DataFrame(X_train_pca, columns=cols)
        df_train['T'] = T_train
        df_train['E'] = E_train
        
        df_test = pd.DataFrame(X_test_pca, columns=cols)
        df_test['T'] = T_test
        df_test['E'] = E_test
        
        # Heavy L2 penalizer prevents lifelines from failing on complete separation or colinearity
        cph = CoxPHFitter(penalizer=5.0)
        try:
            cph.fit(df_train, duration_col='T', event_col='E')
            
            # Predict partial hazard
            train_preds = cph.predict_partial_hazard(df_train)
            test_preds = cph.predict_partial_hazard(df_test)
            
            # lifelines concordance_index expects higher prediction score = lower risk. 
            # predict_partial_hazard outputs higher = higher risk. So we invert it.
            train_c = concordance_index(T_train, -train_preds, E_train)
            test_c = concordance_index(T_test, -test_preds, E_test)
            
        except Exception as e:
            print(f"Fold {fold+1} failed convergence: {e}")
            train_c, test_c = np.nan, np.nan
            
        train_c_indices.append(train_c)
        test_c_indices.append(test_c)
        print(f"Fold {fold+1}: Train C-index = {train_c:.3f}, Test C-index = {test_c:.3f}")
        
    print("\n=== CoxPH Tabular Baseline Results ===")
    print(f"Mean Train C-index: {np.nanmean(train_c_indices):.3f} +/- {np.nanstd(train_c_indices):.3f}")
    print(f"Mean Test C-index:  {np.nanmean(test_c_indices):.3f} +/- {np.nanstd(test_c_indices):.3f}")
    
    results = {
        "model": "CoxPH",
        "dataset": "TCGA-BRCA",
        "folds": folds,
        "train_c_index_mean": float(np.nanmean(train_c_indices)),
        "train_c_index_std": float(np.nanstd(train_c_indices)),
        "test_c_index_mean": float(np.nanmean(test_c_indices)),
        "test_c_index_std": float(np.nanstd(test_c_indices))
    }
    with open(out_dir / "coxph_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir}/coxph_results.json")

if __name__ == "__main__":
    main()
