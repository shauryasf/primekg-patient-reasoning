# GNN recurrence benchmark

Primary endpoint pfs_36m. Best model by mean auprc: **rgcn**.
AUROC 0.673 ± 0.066; AUPRC 0.384 ± 0.036; C-index 0.659 ± 0.055.

Objective: `bce` (early stop on auprc, auxiliary BCE weight 0).

Y=1 is progression by 36 months. Patients censored earlier are dropped. PFS months define the window and are not features.

C-index is Harrell's concordance over comparable pairs of (PFS_MONTHS, PFS_STATUS). A censored patient is only ever the later member of a pair, so the concordance objectives do not treat `Y = 0` as lifelong disease-free. Time and event are targets, never features.

Graph and MLP scores add a clinical linear head to the alteration or graph score. Pooling uses the patient and altered genes.

Explanation paths written: 24.
