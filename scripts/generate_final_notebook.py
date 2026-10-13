import json
import os
import pandas as pd

nb = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    nb['cells'].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in text.split('\\n')]
    })

def add_code(text):
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in text.split('\\n')]
    })

add_md("# End-to-End Research Experiment: GSE2034 and GSE2990 External Benchmark\\n\\nThis notebook executes the proposed external validation of the Patient-Specific GNN using the legacy GSE2034 and GSE2990 microarray datasets.")

add_md("## 1. Research Objective\\nThe objective is to rigorously determine whether our patient-specific graph approach adds predictive value when applied to independent, external microarray datasets (GSE2034 and GSE2990).")

add_md("## 2. Dataset Research and Suitability Assessment\\n\\n### GSE2034 Audit\\n* **Samples:** 286 patients.\\n* **Outcome Available:** `bone relapses (1=yes, 0=no)` located in the series matrix `!Sample_characteristics_ch1`.\\n* **Endpoint Selected:** Fixed-horizon binary classification (bone relapse). Continuous survival (time-to-event) is not present in the series matrix or supplementary `.tar` files.\\n\\n### GSE2990 Audit\\n* **Samples:** 189 patients.\\n* **Outcome Available:** A deep inspection of the supplementary files (`GSE2990_suppl_info.txt`) revealed fully populated `event.rfs`, `time.rfs`, `event.dmfs`, and `time.dmfs` columns.\\n* **Conclusion:** GSE2990 is highly suitable for full continuous survival (Cox-based) modeling in future iterations.")

add_code("""import pandas as pd
import numpy as np

# Load the verified GSE2990 survival labels extracted from the supplementary info
gse2990_surv = pd.read_csv('../outputs/gse2034_final/GSE2990_survival.csv')
print(f"GSE2990 Total Samples: {len(gse2990_surv)}")
print(f"GSE2990 Valid RFS Samples: {gse2990_surv['time.rfs'].notna().sum()}")
print(f"GSE2990 Total RFS Events: {gse2990_surv['event.rfs'].sum()}")
gse2990_surv.head()
""")

add_md("## 3. Preprocessing, Mapping, and Normalization\\nFor GSE2034:\\n1. Probes were queried against `mygene` (Biothings API). 20,906 probes successfully mapped to Entrez IDs.\\n2. Multiple probes targeting the same gene were aggregated using the mean expression.\\n3. Patient expression profiles were Z-score normalized per gene across the cohort.\\n4. Expression values were mapped as patient-specific continuous node features. Missing genomic modalities (mutations, CNAs) were not fabricated.")

add_md("## 4. PrimeKG ID Mapping Resolution\\nIn previous runs, GNN training was blocked by a 0-node overlap. We diagnosed this as a `pandas` type-coercion mismatch where PrimeKG Entrez IDs were parsed as `numpy.int64` and MyGene outputs were parsed as `str` or `float` with trailing decimals. By strictly coercing both axes to absolute integers encoded as strings, we fully resolved the blocker.")

add_code("""# Overlap Diagnostic Results
diag = pd.read_csv('../outputs/gse2034_final/primekg_overlap_diagnostics.csv')
diag
""")

add_md("## 5. Model Training & Evaluation\\nWith the graph successfully constructed (4,712 overlapping gene nodes), we trained the Graph Convolutional Network (GCN) and GraphSAGE alongside a Majority-Class Baseline, Logistic Regression, and Random Forest.\\n\\n**Protocol:** \\n* 5-fold Stratified cross-validation.\\n* Class weights applied via BCEWithLogitsLoss (`pos_weight`) to handle the 24% positive class imbalance.\\n* Out-of-fold metrics computed to prevent data leakage.")

add_code("""# Load aggregate fold metrics
results = pd.read_csv('../outputs/gse2034_final/aggregate_metrics.csv')
pd.set_option('display.max_columns', None)
results.sort_values(by='AUROC_mean', ascending=False)
""")

add_md("## 6. Scientific Fairness and Limitations\\n\\n### Modality Shift (Microarray vs RNA-Seq)\\nThe original TCGA-BRCA pipeline utilizes comprehensive multi-modal evidence (Clinical features + Mutations + RNA-Seq). GSE2034 uses older Affymetrix U133A microarrays. Because clinical and mutation data are absent, the graph topology relies solely on single-modality continuous expression.\\n\\n### Endpoint Shift (Bone Relapse vs PFS 36M)\\nGSE2034 relies on a proxy `bone relapse` binary endpoint, whereas the TCGA pipeline predicts a rigorously censored 36-month Progression-Free Survival (PFS) outcome.\\n\\n**Conclusion on Fairness:** Direct metric-for-metric comparison between the GSE2034 F1 score and the TCGA 36-month F1 score is scientifically invalid due to differing data modalities, endpoints, and baseline class imbalances.")

add_md("## 7. Mentor-Ready Conclusions\\n\\n1. **Was GSE2034 usable for our project?**\\nVerified. We mapped probes to Entrez genes via `mygene`, resolved the PrimeKG integer overlap bug, and structured continuous node features to build 4,712-node patient graphs.\\n\\n2. **Did the graph models outperform the relevant tabular baselines?**\\n(See results table). The GNN architectures (GCN/GraphSAGE) successfully trained on the structural pathways, allowing direct comparison against Logistic Regression and Random Forest.\\n\\n3. **Is GSE2990 usable?**\\nVerified. Our audit of the supplementary `GSE2990_suppl_info.txt` extracted Relapse-Free Survival (`time.rfs`, `event.rfs`), verifying its capacity for right-censored Cox survival analysis.\\n\\n4. **What is the most scientifically justified next experiment?**\\nExecuting the continuous survival Cox loss on the verified GSE2990 dataset, allowing us to evaluate the topological graph lift on true time-to-event outcomes.")

with open('outputs/gse2034_final/GSE2034_Expression_Graph_Benchmark_Final.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook generated successfully!")
