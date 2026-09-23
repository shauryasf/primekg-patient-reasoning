# GNN recurrence benchmark

Primary endpoint PFS. Best model by mean AUPRC: **graphsage**.
AUROC 0.542 ± 0.051; AUPRC 0.216 ± 0.054.

Y=0 means censored at last follow-up. Age was z-scored inside each training fold. Outcome columns were not graph features.

Sensitivity on NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT with graphsage: AUROC 0.537 ± 0.084; AUPRC 0.158 ± 0.048.
Explanation paths written: 24.
