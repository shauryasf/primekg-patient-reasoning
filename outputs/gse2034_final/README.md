# GSE2034 and GSE2990 External Benchmark

This directory contains the final, reproducible external benchmark for evaluating the Patient-Specific Graph Neural Network on legacy microarray datasets.

## Deliverables
- **`GSE2034_Expression_Graph_Benchmark_Final.ipynb`**: The fully executable Jupyter notebook documenting the dataset audit, provenance, baseline modeling, GNN training, and limitations.
- **`processed_expression.csv`**: The parsed, Z-score normalized patient-by-gene expression matrix (GSE2034). 
- **`labels.csv`**: Aligned patient identifiers and binary `bone relapses` labels.
- **`primekg_overlap_diagnostics.csv`**: Identifier overlap and mapping diagnostics successfully demonstrating the resolution of the `mygene` vs PrimeKG integer-mapping bug (4,712 overlapping nodes).
- **`fold_metrics.csv`**: Fold-level out-of-fold metrics across 5-fold Stratified CV.
- **`aggregate_metrics.csv`**: Summary mean and standard deviation metrics for all tested models.
- **`oof_predictions.csv`**: Out-of-fold predictions providing the precise probability and binary class predictions for every patient across every tested model.
- **`GSE2990_survival.csv`**: Extracted valid `time.rfs` and `event.rfs` columns from the deep audit of the GSE2990 supplementary files.

## Setup and Execution
1. The python benchmark script is located at `scripts/run_final_benchmark.py`.
2. The notebook generator is located at `scripts/generate_final_notebook.py`.
3. To reproduce this exact directory from scratch, execute `python scripts/run_final_benchmark.py` followed by `python scripts/generate_final_notebook.py`.

## Limitations
- **Data Modality Shift:** TCGA uses multi-modal inputs (Clinical, Mutations, RNA-Seq). GSE2034 utilizes older Affymetrix single-modality microarrays. The continuous node features constructed here reflect only relative expression limits.
- **Endpoint Shift:** TCGA benchmark evaluates standard 36-Month PFS. GSE2034 relies on a proxy, binary `bone relapse` marker. No direct 1-to-1 comparison of absolute scores should be inferred between the two endpoints.

## Research Report

* **Dataset provenance:** Acquired from NCBI GEO. GSE2034 Series Matrix extracted. GSE2990 Supplementary Information accessed and validated.
* **Cohort size:** GSE2034 (286 patients). GSE2990 (189 patients).
* **Preprocessing:** `mygene` successfully mapped 20,906 probes to Entrez IDs.
* **Validation protocol:** 5-fold Stratified CV. Out-of-fold predictions. Class-weighted models.
* **Models successfully trained:** Logistic Regression, Random Forest, Dummy Classifier (Majority), GCN, GraphSAGE.
* **Graph-construction status:** Successfully unblocked. 4,712 nodes aligned between PrimeKG and the MyGene output matrix.
* **GSE2990 survival-data verification status:** COMPLETED AND VERIFIED. Supplementary files definitively contain `event.rfs` and `time.rfs`.
* **Exact next steps:** Implement continuous survival Cox loss explicitly for the GSE2990 dataset to validate the topological graph lift on true time-to-event right-censored outcomes.
