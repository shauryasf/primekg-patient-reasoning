# Patient-conditioned PrimeKG graph structure learning

PrimeKG hop depth and GNN depth are different things. Hop depth decides which biomedical nodes are retrieved. The GNN still has three message-passing layers over whatever graph was retrieved.

## What changed

The old builder always retrieved a fixed neighborhood: one protein-protein step inside the 500-gene panel, then pathway, drug, and breast/mammary disease links from the altered gene or that protein partner. A second protein's neighbors were not crawled. That behavior is now `fixed_hops: 2`.

`config/features.yaml` selects the candidate bound:

| Setting | What is retrieved |
| --- | --- |
| `mode: fixed`, `fixed_hops: 0` | Patient, sample, and altered genes only |
| `fixed_hops: 1` | Plus direct panel protein partners, pathways, drugs, and breast/mammary diseases of altered genes |
| `fixed_hops: 2` | The previous default, including panel genes that share an altered gene's pathway |
| `fixed_hops: 3` | One further pathway, drug, or disease step from those nodes |
| `mode: gsl` | Retrieve `max_candidate_hops`, then learn a weight on each PrimeKG edge |

Protein-protein expansion still starts only at altered genes and still stops at the 500-gene panel. Disease names still have to contain "breast" or "mammary".

## Learned weights

A global search over K is feasible, but it picks one depth for the whole cohort. The scorer instead starts from the fixed neighborhood and learns a patient-specific exception.

Hops `1..fixed_hops` start at weight 1. That is the same graph as `mode: fixed` with that K. A farther hop, up to `max_candidate_hops`, starts at `sigmoid(gsl_closed_logit)` (about 0.12 with the default -2) and a patient-specific head can open it. The edge and relation residual is `2 * sigmoid(logit)`, which is also 1 at the zero start and can move either way. Observed patient edges stay at 1.

The first GSL run did not do this. A free sigmoid on `[source, destination, relation, patient, hop]` started near 0.5 and then drove every 2-hop edge below 0.5, including the pathway and drug edges the fixed K=2 model was using. GCN AUPRC fell from 0.396 to 0.316. Hop is no longer an input the MLP can use to shut a whole ring off. It is only the prior gate above.

The score multiplies the message in GCN, GraphSAGE, and GATv2. The structure parameters use `gsl_learning_rate` and no weight decay, so the prior does not get pulled back to zero. A threshold is applied only when `gsl_hard_prune` is true and the model is in eval mode.

`gsl_patient_context: false` drops the patient vector. That is the ablation for a score that ignores which patient the edge was attached to. `gsl_node_scores: true` also multiplies by a learned destination-node score.

The scorer never reads `PFS_STATUS`, `PFS_MONTHS`, or any other outcome column.

## How to run a baseline or GSL

```bash
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity --fusion-mode fixed --fixed-hops 1 --output-dir outputs/gnn_benchmark_k1
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity --fusion-mode fixed --fixed-hops 2 --output-dir outputs/gnn_benchmark_k2
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity --fusion-mode fixed --fixed-hops 3 --output-dir outputs/gnn_benchmark_k3
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity --fusion-mode gsl --max-candidate-hops 3 --output-dir outputs/gnn_benchmark_gsl
```

GSL writes `gsl_edges.csv` for the explained patients. Each row has the source, relation, destination, original hop, learned score, and whether the optional threshold would keep it.

## Trained 36-month run

`outputs/gnn_benchmark_gsl` is a 5-fold run of the same models as the fixed K=2 benchmark, with `mode: gsl` and `max_candidate_hops: 3`. Soft weights were used in training. Hard pruning stayed off. `gsl_edges.csv` is the GraphSAGE scorer on the explained progression cases (the best graph model by mean AUPRC).

| Model | AUROC | AUPRC |
| --- | --- | --- |
| clinical_lr | 0.703 ± 0.081 | 0.463 ± 0.133 |
| graphsage | 0.660 ± 0.116 | 0.383 ± 0.103 |
| gatv2 | 0.627 ± 0.062 | 0.366 ± 0.091 |
| gcn | 0.609 ± 0.080 | 0.316 ± 0.111 |
| mlp | 0.562 ± 0.098 | 0.262 ± 0.064 |

On the explained patients, observed patient edges stay at score 1. PrimeKG edges at hop 1 have mean score 0.44 and 52% would pass a 0.5 threshold. Hop 2 has mean score 0.24 and none pass 0.5. Hop 3 has mean score 0.23 and 19% would pass 0.5. That threshold is a report cutoff, not a training prune.

## What this does not claim

Retrieving three hops does not mean the model uses three hops. The clinical logistic regression still has the highest mean AUPRC on this endpoint. GraphSAGE and GATv2 are close to their fixed K=2 numbers; GCN is lower than the fixed K=2 GCN (AUPRC 0.396). The unit tests check shapes, the panel bound, relation labels, patient conditioning, and that the loss reaches the scorer.
