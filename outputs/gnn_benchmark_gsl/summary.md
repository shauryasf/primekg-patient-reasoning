# GNN recurrence benchmark

Primary endpoint pfs_36m. Best model by mean AUPRC: **clinical_lr**.
AUROC 0.703 ± 0.081; AUPRC 0.463 ± 0.133.

Y=1 is progression by 36 months. Patients censored earlier are dropped. PFS months define the window and are not features.

Graph and MLP scores add a clinical linear head to the alteration or graph score. Pooling uses the patient and altered genes.

Explanation paths written: 24.
