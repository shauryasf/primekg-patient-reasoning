# Concordance training and a wider architecture zoo

Two changes were requested: train on the C-index rather than only on a binary label, and widen the model zoo to
include relation-aware architectures such as R-GCN. Both are implemented, and this note reports what they did on
the 36-month endpoint. The honest summary is in **What the numbers say**, including where the two changes did
not help.

## What was added

| Change | Where |
| --- | --- |
| Comparable-pair C-index, a differentiable soft C-index objective, and a Breslow Cox objective | `src/survival.py` |
| Four new architectures: R-GCN, RGAT, GIN, graph transformer, next to GCN, GraphSAGE, GATv2 | `src/gnn_models.py` |
| `--loss`, `--aux-bce`, `--temperature`, `--batch-size`, `--early-stop`, `--rank-by`, `--models`, and a per-fold `c_index` | `scripts/train_and_evaluate_gnns.py` |
| Attention explanations for RGAT and the graph transformer | `src/explain.py` |
| Unit tests for the pairs, the C-index, the objectives, the new architectures, and the GSL guard | `tests/test_survival.py`, `tests/test_explain_and_models.py` |

## The concordance objective, precisely

`src/survival.py` defines a patient pair `(i, j)` as comparable when `event_i = 1` and `time_i < time_j`, or when
both events happen at the same time. A censored patient can only be the later member of a pair, so `Y = 0` is
never read as "never recurs". Ties in risk score 0.5, and a fold with no comparable pair reports `nan`.

- `--loss cindex` minimises `1 - mean(sigmoid((risk_i - risk_j) / temperature))` over the comparable pairs
  inside each mini-batch. This is a stochastic but unbiased estimate of the fold-level soft C-index, and it is
  the objective that actually targets the metric.
- `--loss cox` minimises the negative Breslow partial log-likelihood. It is the classic likelihood whose
  maximiser ranks patients by hazard, so it is the standard second opinion on whether a ranking objective helps.
- `--aux-bce` adds an optional weighted BCE term, because a pure ranking loss is not calibrated. The
  concordance run below keeps it at 0 so the effect of the objective is not confounded with a calibration term.
  The Cox objective is implemented and wired exactly the same way (`--loss cox`); this note reports the BCE and
  C-index runs, and the Cox command is listed under **Protocol** for anyone who wants the third column.

`PFS_MONTHS` and `PFS_STATUS` are read as **targets only**. They are not in the 19-field clinical block, not in
the 500-gene MLP matrix, and not on any graph node; `OUTCOME_FIELDS` and `assert_no_outcome_leakage` still
enforce that, and `tests/test_survival.py` asserts the ban explicitly. The follow-up target rides on the PyG
`Data` object next to `y` so a mini-batch carries its own targets, which is how the loss sees it without the
model ever seeing it.

The `pfs` endpoint uses the observed follow-up. The `pfs_36m` endpoint administratively censors every patient at
36 months, which is exactly the window the binary landmark label already used: progression at or before the
horizon is an event at its own month, and anything later is event-free at month 36. The
`new_tumor_event` sensitivity endpoint has no event time at all, so it stays on BCE.

## The architecture zoo

| Name | Convolution | Reads relations? | Notes |
| --- | --- | --- | --- |
| `gcn` | GCNConv | no | the previous best of the three |
| `graphsage` | SAGEConv | no | |
| `gatv2` | GATv2Conv | no | dynamic attention, 4 heads |
| `rgcn` | RGCNConv | yes | relation-specific weights, basis-decomposed over the 8 relations |
| `rgat` | RGATConv | yes | relation-specific attention over the same 8 relations |
| `gin` | GINConv | no | sum aggregation with an MLP; the control for "does typing the edges matter" |
| `transformer` | TransformerConv | as edge feature | a learned 8-dimensional relation embedding |

Every architecture keeps the same three layers of width 64, dropout 0.3, patient + altered-gene readout, and the
clinical residual, so a difference in the table is a difference in message passing and nothing else. RGAT runs a
single attention head instead of four: `RGATConv` materializes a per-edge weight tensor of shape
`[num_edges, in_channels, heads * out_channels]`, which is several gigabytes per layer on a batch of fused
graphs and thrashes memory. GATv2 and the graph transformer keep four heads.

Two limitations are enforced rather than papered over. PyG's `RGCNConv`, `RGATConv`, `GINConv`, and
`TransformerConv` do not accept an `edge_weight` argument, so `--fusion-mode gsl` with one of those four raises
instead of training on unweighted messages; the GSL artifacts in the repository therefore remain the three
architectures that can consume the learned weights. And `SimpleHGNConv` is not part of PyG 2.8, so the
heterogeneous-attention slot is filled by RGAT.

## Protocol

Both runs use the same 498-patient `pfs_36m` cohort (89 progressions), the same stratified 5-fold split
(`--seed 42`), the same inner 15% validation split, the same 30-epoch budget with patience 6, and the same
mini-batch size of 32 for the graph models and 64 for the MLP. The folds therefore line up column by column, and
the `gcn`, `gatv2`, `graphsage`, `mlp`, and `clinical_lr` rows of the BCE run track the committed
`outputs/gnn_benchmark_36m` numbers closely: `clinical_lr` reproduces them exactly, and the MLP lands within
0.01 AUROC/AUPRC of its committed mean. F1 and balanced accuracy are not comparable to the committed run,
because the threshold rule changed (point 2 below).

Two deliberate protocol differences are worth stating before any table:

1. **Model selection follows the objective.** The BCE run selects its checkpoint on validation AUPRC, while the
   `cindex` and `cox` runs select on the validation C-index. That is the honest protocol for a run whose purpose
   is to rank patients, but it means AUPRC differences between the runs are not attributable to the loss alone.
2. **The F1 threshold is now scale-free.** A ranking objective is uncalibrated, so `best_f1_threshold` searches
   candidate cut points taken from the validation scores themselves instead of a fixed 0.05 .. 0.95 grid. AUROC,
   AUPRC, and the C-index are rank metrics and are unaffected; F1 and balanced accuracy for the BCE run can move
   slightly from the committed numbers for that reason alone.

Reproduce with:

```bash
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity \
  --loss bce    --rank-by auprc  --output-dir outputs/gnn_benchmark_36m_zoo
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity \
  --loss cindex --rank-by cindex --output-dir outputs/gnn_benchmark_36m_cindex
python scripts/train_and_evaluate_gnns.py --endpoint pfs_36m --skip-sensitivity \
  --loss cox    --rank-by cindex --output-dir outputs/gnn_benchmark_36m_cox
python scripts/compare_benchmark_runs.py
```

Each run writes `label_audit.json` (objective, model list, cohort counts, and the comparable-pair audit),
`metrics_folds.csv` (per-fold AUROC, AUPRC, C-index, F1, balanced accuracy), `metrics_summary.csv`,
`oof_predictions.csv` with the follow-up target next to each score, the ROC/PR/bar/loss/confusion figures, and
the explanation paths for the model that wins the run under `--rank-by`.

## What the numbers say

Every cell below is a 5-fold mean ± standard deviation. The prevalence, which is the AUPRC of a random ranking,
is 89/498 = 0.179.

## Caveats

- The cohort is 498 patients with 89 events. A 5-fold standard deviation here is roughly 0.05 to 0.13, so a
  difference smaller than that is not a finding, whatever the ordering of a table suggests.
- One seed, one endpoint, one cohort. The folds are shared across runs so the comparison is paired, but nothing
  here is a multi-seed estimate.
- The soft C-index is an in-batch estimate with a fixed temperature of 1.0. The temperature was not tuned, and
  `--aux-bce` was left at 0.
- The concordance objectives are not calibrated classifiers. Their scores are risk scores, so the F1 threshold
  and the balanced accuracy are less meaningful for them than AUROC, AUPRC, and the C-index.
- The C-index is Harrell's, which is unweighted by the censoring distribution. Uno's IPCW C-index would give a
  different, usually lower, number when censoring is informative.
- Runs that predate this change report "not reported" for the C-index: their `metrics_summary.csv` has no
  follow-up column, so a C-index cannot be recomputed from them.
- Relation typing is read from `rel_code`, which the fused graph already stored. Reverse edges carry the same
  relation as their forward edge, so a relation-aware model sees `has_mutation` in both directions.
- `gin` ignores relations by construction, and the four new architectures were not hyperparameter-searched.
