# GNN recurrence benchmark

Primary endpoint pfs_36m. Best model by mean cindex: **clinical_lr**.
AUROC 0.703 ± 0.081; AUPRC 0.463 ± 0.133; C-index 0.682 ± 0.072.

Objective: `cindex` (early stop on cindex, auxiliary BCE weight 0).

Y=1 is progression by 36 months. Patients censored earlier are dropped. PFS months define the window and are not features.

C-index is Harrell's concordance over comparable pairs of (PFS_MONTHS, PFS_STATUS). A censored patient is only ever the later member of a pair, so the concordance objectives do not treat `Y = 0` as lifelong disease-free. Time and event are targets, never features.

Graph and MLP scores add a clinical linear head to the alteration or graph score. Pooling uses the patient and altered genes.

Explanation paths written: 24.
