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

`NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT` (`Yes` → 1, `No` → 0, blank dropped) is only a sensitivity analysis. The full four-model cross-validation uses PFS. The sensitivity reruns the winning architecture on the complete cases for the new-tumor column.

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

`src/gnn_models.py` implements:

- MLP
- GCN
- GATv2 (four heads, attention aligned with `edge_index`)
- GraphSAGE

Graph models use three layers of width 64, dropout 0.3, and a readout that concatenates the patient node with global mean and max pooling. Training is stratified 5-fold cross-validation with an inner validation split for early stopping on AUPRC. The positive-class weight is computed on the inner training split. F1 and balanced accuracy use a threshold chosen on that validation split, then frozen for the test fold. Reported metrics are AUROC, AUPRC, F1, and balanced accuracy.

## Explanations

For held-out patients with a positive PFS label, edge importance comes from GATv2 attention when that model wins, and from PyG GNNExplainer otherwise. Paths are enumerated only along stored edges, from the patient through an altered gene to a pathway, drug, or disease node. A resistance or treatment claim is not added unless that edge is in the filtered PrimeKG slice.

## Run

```bash
python scripts/train_and_evaluate_gnns.py
```

The script downloads the public cBioPortal clinical and discrete CNA tables when they are missing. The study MAF is not available from the data hub LFS budget, so a missing `data_mutations.txt` is rebuilt from the cBioPortal mutations API for genes in `data_cna.txt`. That API extract is smaller than the raw MAF (for example 24 calls instead of 50 on `TCGA-3C-AAAU-01`), because the portal does not return every raw row. `tests/test_mutation_mapping.py` and `tests/test_mutation_count.py` still expect the original study file, so they fail when only the API extract is present. An existing `data_mutations.txt` larger than 10 KB is left untouched. PrimeKG `kg.csv` is downloaded from Harvard Dataverse and filtered to `data/primekg/primekg_panel_slice.csv`.

Artifacts land in `outputs/gnn_benchmark/`.
