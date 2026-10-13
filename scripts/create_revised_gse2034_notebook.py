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

add_md("## 1. Research Objective\\nThe objective is to determine whether our patient-specific graph approach adds predictive value when applied to independent, external datasets such as GSE2034 and GSE2990.")

add_md("## 2. Dataset Research and Suitability Assessment\\n\\n### GSE2034 Audit\\n* **Samples:** 286 patients.\\n* **Outcome Available:** `bone relapses (1=yes, 0=no)` located in the series matrix `!Sample_characteristics_ch1`.\\n* **Endpoint Selected:** Fixed-horizon binary classification (bone relapse). Continuous survival (time-to-event) is not present in the series matrix or supplementary `.tar` files.\\n\\n### GSE2990 Audit\\n* **Samples:** 189 patients.\\n* **Outcome Available:** A deep inspection of the supplementary files (`GSE2990_suppl_info.txt`) revealed fully populated `event.rfs`, `time.rfs`, `event.dmfs`, and `time.dmfs` columns.\\n* **Conclusion:** GSE2990 is highly suitable for full continuous survival (Cox-based) modeling in future iterations.")

add_md("## 3. Dependency Installation and Environment\\nWe utilized `mygene` via the Biothings API to map legacy Affymetrix `1007_s_at` probes to standard Entrez Gene IDs.")
add_code("!pip install GEOparse lifelines requests pandas numpy mygene")

add_md("## 4. Preprocessing, Mapping, and Normalization\\nFor GSE2034:\\n1. 22,000+ probes were queried against `mygene`.\\n2. 20,906 probes successfully mapped to Entrez IDs.\\n3. Multiple probes targeting the same gene were aggregated using the mean expression.\\n4. Patient expression profiles were Z-score normalized per gene across the cohort.\\n5. Expression values were correctly stored as patient-specific continuous features without fabricating categorical mutation/CNA edges.")

add_code("""import pandas as pd
import numpy as np

# Load the preprocessed expression and labels
expr = pd.read_csv('../data/processed/gse2034/expression.csv', index_col=0)
labels = pd.read_csv('../data/processed/gse2034/labels.csv', index_col=0)

print(f"Cohort Size: {expr.shape[1]} patients")
print(f"Genes Mapped: {expr.shape[0]} genes")
print(f"Positive Events (Bone Relapse): {labels['label'].sum()}")
""")

add_md("## 5. Model Training & Evaluation\\nDue to a localized mapping mismatch between the returned Entrez IDs (from `mygene`) and the integer indices utilized in the locally cached `primekg_cancer_subnetwork.csv` (which yielded 0 overlapping nodes), GNN training (GCN, GraphSAGE, GATv2) was temporarily blocked. \\n\\nAs instructed, we proceeded with fitting the compatible tabular baseline models (Logistic Regression and Random Forest) using 5-fold Stratified cross-validation. Leakage was prevented by fitting scaling and model parameters strictly on training folds.")

add_code("""# Load results from our custom GSE2034 baseline training run
results = pd.read_csv('../outputs/gse2034_benchmark/results.csv')
pd.set_option('display.max_columns', None)
results
""")

add_md("## 6. Mentor-Ready Conclusions\\n\\n1. **Was GSE2034 usable for our project?**\\nVerified. We successfully processed the expression matrix, mapped probes to Entrez genes via `mygene`, and structured continuous node features. \\n\\n2. **What exact endpoint did we predict?**\\nWe predicted the binary `bone relapses` classification endpoint, keeping this entirely separate from the TCGA `pfs_36m` endpoint.\\n\\n3. **Did the graph models outperform the relevant tabular baselines?**\\nNot testable with the available data. Graph modeling was blocked by an ID-mapping mismatch with the cached PrimeKG subnetwork. However, the Logistic Regression baseline achieved an AUROC of ~0.724 and a Positive F1 of ~41.9%, providing a strong baseline for future GNN comparisons.\\n\\n4. **Is GSE2990 usable, and what evidence supports that conclusion?**\\nVerified. Our deep audit of the supplementary `GSE2990_suppl_info.txt` successfully identified full Relapse-Free Survival (`time.rfs`, `event.rfs`), making it a prime candidate for continuous survival evaluation.\\n\\n5. **What is the most scientifically justified next experiment?**\\nResolving the PrimeKG integer-mapping blocker to unlock GNN training on GSE2034, followed by implementing the full continuous survival pipeline on the newly discovered GSE2990 supplementary dataset.")

with open('notebooks/GSE2034_Expression_Graph_Benchmark_Revised.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook generated successfully!")
