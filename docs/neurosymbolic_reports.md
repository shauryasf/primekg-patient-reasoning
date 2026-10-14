# Neuro-symbolic progression reasoning and patient reports

This iteration stops chasing the GNN score, which had saturated, and builds the deliverable: a neuro-symbolic
model whose predictions come with knowledge-graph paths and survival rules, and a hospital-style PDF report for
every patient, binned by progression timeline.

The mentor walkthrough with pre-run outputs is
[notebooks/neurosymbolic_patient_reports.ipynb](../notebooks/neurosymbolic_patient_reports.ipynb).

## Cohort and groups

Every patient with a `PFS_STATUS` is used: 1,083 patients, 145 recorded progressions, full follow-up rather
than the 36-month landmark subset. Reports are grouped by the **recorded** progression month:

| Group | Recorded PFS | Patients |
| --- | --- | --- |
| A | progression ≤ 12 months | 32 |
| B | progression 12–24 months | 30 |
| C | progression 24–36 months | 27 |
| D | progression > 36 months | 56 |
| N | no progression recorded (censored) | 938 |

Only 12 progressions happen within 6 months, so finer early bins would be nearly empty.

## Pipeline

`scripts/run_neurosymbolic_pipeline.py` runs four cached stages over one set of stratified 5-fold splits, so
every prediction is out-of-fold. RuleKit runs on the JVM, so the symbolic stage needs a Java runtime (11 or
later). The `rulekit` pip package ships its own jar. The full run took about 45 minutes on an 8-core laptop
CPU.

1. **Neural.** GCN, GraphSAGE, and R-GCN on the fused patient + PrimeKG graph, trained with the Breslow Cox
   likelihood or the differentiable C-index. RGAT was dropped (over five minutes per epoch on a laptop CPU).
2. **Symbolic.** RuleKit LR-Rules (log-rank rule induction) over 187 readable predicates: extended clinical
   context (T/N/M, AJCC stage, PAM50, histology, radiation, aneuploidy, TMB), driver mutations and CNAs,
   amplicons (co-amplified genes with Jaccard ≥ 0.8 merged), and PrimeKG facts ("an altered gene is annotated to
   pathway P / breast disease D"). Binary predicates enter as "yes" or missing so no rule is built from the absence
   of a rare passenger mutation. A log-rank test of every PrimeKG pathway, drug, and disease node is also computed
   per training fold.
3. **Fusion.** Cox models on the clinical block with and without the out-of-fold GNN and rule risks, the rule and
   GNN risks alone, and a random survival forest. Survival curves for the report model; progression-time
   estimators scored on held-out progressors.
4. **Explain.** GNNExplainer edge importances from the held-out fold model of the best GNN, and every
   patient → sample → altered gene → (partner) → pathway / drug / disease path over stored edges.

## Results

Fold-mean Harrell C-index (± sd over folds), all out-of-fold:

| Model | Harrell C | Uno C (120 mo) | IBS (120 mo) |
| --- | --- | --- | --- |
| Clinical Cox, 19-field block | 0.675 ± 0.035 | 0.616 | 0.134 |
| Clinical Cox, extended | 0.667 ± 0.042 | 0.657 | 0.135 |
| **Neuro-symbolic Cox** (clinical + GNN + rules) | 0.665 ± 0.024 | 0.616 | 0.134 |
| Neuro-symbolic Cox, extended clinical | 0.665 ± 0.043 | 0.656 | 0.135 |
| GNN + rules, no clinical | 0.642 ± 0.045 | 0.594 | 0.139 |
| GraphSAGE, C-index objective | 0.622 ± 0.067 | 0.573 | — |
| RuleKit rules | 0.619 ± 0.048 | 0.623 | — |
| GCN, Cox | 0.612 ± 0.044 | 0.549 | — |
| GraphSAGE, Cox | 0.603 ± 0.062 | 0.556 | — |
| R-GCN, Cox | 0.586 ± 0.041 | 0.590 | — |
| Random survival forest | 0.554 ± 0.023 | 0.559 | 0.145 |

**Model choice.** The top four models are within 0.01 of each other with a fold sd of about 0.04. Under the
one-standard-error rule (best minus sd/√5 = 0.659) they are tied, and the report model is the most explainable
of them, the neuro-symbolic Cox model. In every fold its GNN and rule terms get hazard ratios of 0.95–1.05 per
sd with p > 0.5: once the clinical block is present they add no independent prognostic information. The
neuro-symbolic layer adds explanation, not accuracy. `--report-model clinical_cox_19` reports the clinical risk
instead.

**Progression month.** Estimators of "when, given progression", on the 145 held-out progressors:

| Estimator | MAE (mo) | Median |error| | Within 12 mo | Window accuracy |
| --- | --- | --- | --- | --- |
| Constant (median of training progressors) | 22.9 | 16.0 | 37% | 19% |
| Conditional median of the fused curve, τ = 120 | 28.0 | 24.6 | 23% | 39% |
| Log-normal AFT on training progressors | 21.1 | 12.6 | 50% | 34% |
| **Log-logistic AFT on training progressors** | 20.4 | 11.6 | 51% | 34% |

Chance window accuracy with four windows is about 25%. The variants were compared on the same held-out
progressors, so the chosen row is mildly optimistic.

## Neuro-symbolic path score

Each candidate path gets `NS = 0.6 × neural + 0.4 × symbolic`, where *neural* is the cohort percentile of its
mean GNNExplainer edge importance and *symbolic* is `min(|z|, 3) / 3` for the log-rank z of the PrimeKG fact the
path ends in, taken from the patient's training folds. A path is prominent when NS is in the cohort's top 20%.
A report shows `min(5, prominent)` paths with at most two per gene and each end node once, topped up to three
unmarked paths when fewer are prominent.

## Reports

```bash
python scripts/build_patient_reports.py --workers 4
```

writes `outputs/patient_reports/` with one PDF per patient in its group folder, a group summary with a patient
index, a combined PDF for groups A–D, and `00_cohort_overview.pdf`. Each report has the clinical record, the
progression assessment (risk percentile, probabilities at 12/24/36/60 months, estimated month and window, the
recorded outcome with ✓/✗, the survival curve), the top-K paths and fired rules, the molecular profile with GNN
gene importance and PrimeKG drug links, and the five most similar patients.

## Limitations

- Discrimination is saturated at a C-index of about 0.67 for every model family tried.
- The month estimate is within a year for about half of progressors; late progressions are hard to place.
- Paths explain the GNN, the weaker component, and are paired with cohort evidence; they are not causal claims.
- PrimeKG drug links include enzymes, carriers, and transporters and are not treatment recommendations.
- One cohort, one seed, no external validation; the fusion uses cross-validated stacking.
