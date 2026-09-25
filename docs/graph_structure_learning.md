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

For a PrimeKG edge the score is

`sigmoid(MLP([source, destination, relation embedding, patient clinical vector, hop]) / temperature)`.

Edges that were observed in the patient (sample, mutation, amplification, deletion) stay at weight 1. They are not PrimeKG hypotheses.

The score multiplies the message in GCN, GraphSAGE, and GATv2. Training uses that soft weight so the scorer receives gradient from the same loss as the GNN. A threshold is applied only when `gsl_hard_prune` is true and the model is in eval mode.

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

## What this does not claim

Retrieving three hops does not mean the model uses three hops. A high score on a 1-hop edge and a low score on a 3-hop edge is an effective neighborhood, and that distribution has to be counted from a trained `gsl_edges.csv`. This repository does not yet report those percentages because the GSL model has not been trained on the cohort. The unit tests check shapes, the panel bound, relation labels, patient conditioning, and that the loss reaches the scorer.
