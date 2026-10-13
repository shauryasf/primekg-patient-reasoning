# Recurrence labels, PrimeKG fusion, and the GNN benchmark

Milestone 1 patient graphs stay as they are: `Patient → Sample → Gene`, 500-gene panel, and CNA limited to homozygous deletion (−2) and high-level amplification (+2). This note covers the supervised layer built on top of that evidence.

## Target

The primary label is `PFS_STATUS` from `data_clinical_patient.txt`. The public datahub release downloaded by this pipeline has 145 `1:PROGRESSION` labels, 938 `0:CENSORED` labels, and 1 missing value. An older local bundle can show a different event count; the mapping rule does not change.

| Source value | Y |
| --- | --- |
| `1:PROGRESSION` | 1 |
| `0:CENSORED` | 0 |
| anything else, including blank | dropped |

`Y = 0` means the patient was censored at the end of follow-up. It is not a demonstration that recurrence never occurred later. `PFS_MONTHS` is not a feature: follow-up time would leak information about who was observed long enough to record an event.

That censoring is why a binary model on raw `PFS_STATUS` sits near AUROC 0.5. The improved endpoint `pfs_36m` asks a different question: did progression occur by 36 months?

| Situation | `pfs_36m` |
| --- | --- |
| Progression at or before 36 months | 1 |
| Progression after 36 months | 0 |
| Censored at or after 36 months | 0 |
| Censored before 36 months, or missing time | dropped |

`PFS_MONTHS` is read only to apply that window. It is not copied into the clinical block, the MLP matrix, or the graph. A patient censored at month 10 is excluded rather than labeled disease-free.

`NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT` (`Yes` → 1, `No` → 0, blank dropped) is only a sensitivity analysis. The original four-model cross-validation uses raw PFS. The 36-month run compares a clinical logistic regression with the same four architectures and skips that sensitivity.

Overall survival, disease-specific survival, and disease-free survival columns are never encoded. `src/labels.py` lists them in `OUTCOME_FIELDS`, and `assert_no_outcome_leakage` rejects those names in the feature vector.

## Fusion

Patient genes and PrimeKG `gene/protein` nodes join on NCBI Entrez id. Panel values such as `5290.0` and PrimeKG ids such as `5290` both become `"5290"`.

The PrimeKG slice keeps four relations:

- `protein_protein`, and only when both proteins are in the 500-gene panel
- `pathway_protein` for panel genes
- `drug_protein` for panel genes
- `disease_protein` whose disease name contains "breast" or "mammary"

A 2-hop walk over all PrimeKG protein interactions would import most of the interactome and dilute the patient signal. Pathway nodes are the mechanistic second hop: an ERBB2-altered patient still receives the unaltered panel genes that share a pathway with ERBB2, so message passing can meet at that pathway.

Each fused graph is directed from the patient toward samples, altered genes, and PrimeKG nodes, and every relation is also stored in reverse so three message-passing layers can carry pathway context back to the patient. Explanation text uses the forward relation names only.

Node features share one width:

- node type
- clinical block on the patient node only: age, sex, subtype, collapsed AJCC stage (I–IV, X, missing), pathologic M stage
- alteration block on gene nodes: mutation, high amplification, homozygous deletion, log1p mutation-record count, known-driver flag
- a trainable id embedding for genes, pathways, diseases, and drugs

Age is z-scored from the training fold only. The embedding is initialized randomly and trained. It is not a TransE or BioBERT vector; those weights are not part of this run, and the patient-specific signal is the alteration block.

The MLP baseline receives the same clinical block and a 500-gene alteration multi-hot. It does not see PrimeKG edges.

## Models

`src/gnn_models.py` implements seven message-passing architectures plus the MLP baseline:

| Name | Convolution | What it adds |
| --- | --- | --- |
| `gcn` | GCNConv | degree-normalised neighbourhood averaging |
| `graphsage` | SAGEConv | separate self and neighbour transforms |
| `gatv2` | GATv2Conv | dynamic attention, four heads |
| `rgcn` | RGCNConv | relation-specific weights, basis-decomposed over the 8 relations |
| `rgat` | RGATConv | relation-specific attention over the same relations |
| `gin` | GINConv | sum aggregation with an MLP, no relation information |
| `transformer` | TransformerConv | a learned 8-dimensional relation embedding as the edge feature |

The patient graph has always carried `rel_code` per edge, reverse edges included, so the relational models read
the relation of every edge from data that already existed. `gin` deliberately ignores relations and sees only
connectivity, which is the control for "does typing the edges matter". The four new architectures are the
standard relational and attention baselines for heterogeneous graphs, not architectures tuned on this cohort.

Only `gcn`, `graphsage`, and `gatv2` can consume a learned graph-structure weight: PyG's `RGCNConv`, `RGATConv`,
`GINConv`, and `TransformerConv` do not accept an `edge_weight` argument. Asking for `--fusion-mode gsl` with one
of the other four is refused rather than trained on unweighted messages.

Graph models use three layers of width 64, dropout 0.3, and a readout that concatenates the patient node with mean and max pooling over the patient and altered genes. Unaltered pathway neighbors still pass messages, and they are left out of the pool. A linear head on the 19 clinical fields is added to the graph score. The graph classifier starts at zero and is multiplied by a gate that starts near 0.12, so optimization begins from the clinical score. The MLP uses the same clinical head, with the 500-gene multi-hot behind the same gate. `clinical_lr` is logistic regression on those 19 fields alone, with the clinical block standardized inside each training fold.

Training is stratified 5-fold cross-validation with an inner validation split for early stopping on AUPRC. The positive-class weight is computed on the inner training split. F1 and balanced accuracy use a threshold chosen on that validation split, then frozen for the test fold. Reported metrics are AUROC, AUPRC, the C-index, F1, and balanced accuracy.

## Objectives

`--loss` selects what training optimises:

| `--loss` | Objective | Reads |
| --- | --- | --- |
| `bce` (default) | weighted binary cross-entropy on `Y` | `PFS_STATUS` |
| `cindex` | `1 -` a differentiable C-index surrogate over comparable pairs | `PFS_STATUS`, `PFS_MONTHS` |
| `cox` | Breslow negative partial log-likelihood | `PFS_STATUS`, `PFS_MONTHS` |

The concordance index asks a different question from AUROC: of two *comparable* patients, does the model score
the one with the earlier event higher? `src/survival.py` builds the pairs. Patient `i` is comparable to patient
`j` when `event_i = 1` and `time_i < time_j`, or when both events happen at the same time. A censored patient is
therefore only ever the *later* member of a pair, which is what stops `Y = 0` from being read as "never recurs".
Tied risks score 0.5, and a fold with no comparable pair reports `nan` rather than an invented 0.5.

The soft C-index is the mean of `sigmoid((risk_i - risk_j) / temperature)` over the comparable pairs inside each
mini-batch, so it is a stochastic but unbiased estimate of the fold-level soft C-index. `--aux-bce` adds an
optional BCE anchor term; the runs below keep it at 0. The Cox objective maximises the partial likelihood, whose
maximiser ranks patients by hazard. `--early-stop` chooses whether model selection uses validation AUPRC or the
validation C-index, and defaults to the C-index for a concordance loss. `--rank-by` chooses which metric picks
the headline and explained model.

`PFS_MONTHS` is read only to place the event window and the pair order. It is not in the clinical block, not in
the MLP matrix, and not on any graph node; `OUTCOME_FIELDS` and `assert_no_outcome_leakage` continue to enforce
that. `pfs` uses the observed follow-up, `pfs_36m` administratively censors everyone at 36 months, and
`new_tumor_event` has no event time at all, so its sensitivity run stays on `bce`.

A ranking objective is not calibrated, so the F1 threshold is now chosen from the validation scores themselves
instead of a fixed 0.05 .. 0.95 grid. AUROC, AUPRC, and the C-index are rank metrics and do not depend on that
choice. Every fold reports `c_index` next to AUROC, AUPRC, F1, and balanced accuracy, and `metrics_folds.csv`
and `oof_predictions.csv` both carry the follow-up target so the C-index can be recomputed from the artifacts.

On 498 patients with a known 36-month window (89 progressions), stratified 5-fold results are:

| Model | AUROC | AUPRC |
| --- | --- | --- |
| Clinical logistic regression | 0.703 ± 0.081 | 0.463 ± 0.133 |
| GCN | 0.680 ± 0.076 | 0.396 ± 0.091 |
| GraphSAGE | 0.650 ± 0.045 | 0.379 ± 0.091 |
| GATv2 | 0.631 ± 0.046 | 0.355 ± 0.060 |
| MLP (clinical head + gene multi-hot) | 0.562 ± 0.098 | 0.262 ± 0.064 |

The prevalence, which is the AUPRC of a random ranking, is 89/498 = 0.179. The clinical model is the best of this run. GCN is close and does not beat age, sex, subtype, stage, and M stage. The gene multi-hot MLP is worse than the clinical linear model: the extra alteration features add noise. The earlier raw-PFS GraphSAGE result was AUROC 0.542 and AUPRC 0.216 on 1,083 patients. Most of the gain here is the cleaner label plus a model that actually uses the clinical block, not a new PrimeKG signal.

The raw-PFS artifacts remain in `outputs/gnn_benchmark/`. The 36-month comparison is written to `outputs/gnn_benchmark_36m/` by:

```bash
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity --output-dir outputs/gnn_benchmark_36m
```

## Explanations

For held-out patients with a positive PFS label, edge importance comes from GATv2 attention when that model wins, and from PyG GNNExplainer otherwise. Paths are enumerated only along stored edges, from the patient through an altered gene to a pathway, drug, or disease node. A resistance or treatment claim is not added unless that edge is in the filtered PrimeKG slice.

## Run

```bash
python scripts/train_and_evaluate_gnns.py
```

The three 5-fold comparison runs behind the tables above and in
[objectives and architectures](objective_and_architecture_runs.md) are:

```bash
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity \
  --loss bce    --rank-by auprc  --output-dir outputs/gnn_benchmark_36m_zoo
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity \
  --loss cindex --rank-by cindex --output-dir outputs/gnn_benchmark_36m_cindex
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity \
  --loss cox    --rank-by cindex --output-dir outputs/gnn_benchmark_36m_cox
python scripts/compare_benchmark_runs.py
```

The default run evaluates only `rgcn` and `graphsage` with 3 folds, 12 maximum epochs, and patience 5.
Use `--models` to request another supported architecture explicitly; `--folds 2 --epochs 3` remains useful
for a fast smoke check.

The script downloads the public cBioPortal clinical and discrete CNA tables when they are missing. The study MAF is not available from the data hub LFS budget, so a missing `data_mutations.txt` is rebuilt from the cBioPortal mutations API for genes in `data_cna.txt`. That API extract is smaller than the raw MAF (for example 24 calls instead of 50 on `TCGA-3C-AAAU-01`), because the portal does not return every raw row. `tests/test_mutation_mapping.py` and `tests/test_mutation_count.py` still expect the original study file, so they fail when only the API extract is present. An existing `data_mutations.txt` larger than 10 KB is left untouched. PrimeKG `kg.csv` is downloaded from Harvard Dataverse and filtered to `data/primekg/primekg_panel_slice.csv`.

Artifacts land in `outputs/gnn_benchmark/`.
