# GNN recurrence benchmark

Primary endpoint pfs. Best model by mean cindex: **graphsage**.
AUROC 0.611 ± 0.004; AUPRC 0.266 ± 0.051; C-index 0.651 ± 0.027.

Objective: `cindex` (early stop on cindex, auxiliary BCE weight 0).

Y=0 is censored at last follow-up, not lifelong disease-free confirmation. PFS months are not features.

C-index is Harrell's concordance over comparable pairs of (PFS_MONTHS, PFS_STATUS). A censored patient is only ever the later member of a pair, so the concordance objectives do not treat `Y = 0` as lifelong disease-free. Time and event are targets, never features.

Graph and MLP scores add a clinical linear head to the alteration or graph score. Pooling uses the patient and altered genes.

Sensitivity on NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT with graphsage: AUROC 0.564 ± 0.060; AUPRC 0.182 ± 0.054.
Explanation paths written: 24.
