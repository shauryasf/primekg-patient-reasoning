# Neuro-symbolic progression reasoning and patient reports

This iteration stops chasing the GNN score, which had saturated, and builds the deliverable: a neuro-symbolic
model whose predictions come with knowledge-graph paths and survival rules, and a hospital-style PDF report for
every patient, binned by progression timeline.

The mentor walkthrough with pre-run outputs is
[notebooks/neurosymbolic_patient_reports.ipynb](../notebooks/neurosymbolic_patient_reports.ipynb). The latest
change, which replaced IF-THEN rules with Markov logic, soft logic, and reinforcement-learned rules, is explained with
its novelty points and the best survival model in
[notebooks/symbolic_reasoning_novelty.ipynb](../notebooks/symbolic_reasoning_novelty.ipynb). The 25-page project
report is `report/thesis.tex` (compiled to `report/thesis.pdf`).

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

`scripts/run_neurosymbolic_pipeline.py` runs five cached stages over one set of stratified 5-fold splits, so
every prediction is out-of-fold. RuleKit runs on the JVM, so the symbolic stage needs a Java runtime (11 or
later). The `rulekit` pip package ships its own jar. The full run took about 50 minutes on an 8-core laptop
CPU.

1. **Neural.** GCN, GraphSAGE, and R-GCN on the fused patient + PrimeKG graph, trained with the Breslow Cox
   likelihood or the differentiable C-index. RGAT was dropped (over five minutes per epoch on a laptop CPU).
2. **Symbolic baseline.** RuleKit LR-Rules (log-rank rule induction) over 188 readable predicates: extended clinical
   context (T/N/M, AJCC stage, PAM50, histology, radiation, aneuploidy, TMB), driver mutations and CNAs,
   amplicons (co-amplified genes with Jaccard ≥ 0.8 merged), and PrimeKG facts ("an altered gene is annotated to
   pathway P / breast disease D"). Binary predicates enter as "yes" or missing so no rule is built from the absence
   of a rare passenger mutation. A log-rank test of every PrimeKG pathway, drug, and disease node is also computed
   per training fold.
3. **Logic.** Four survival-aware reasoners over 422 first-order ground atoms (see below); the best by the
   one-standard-error rule becomes the rule engine.
4. **Fusion.** Cox models on the clinical block with and without the out-of-fold GNN and rule-engine risks, those
   risks alone, and a random survival forest. Survival curves for the report model; progression-time
   estimators scored on held-out progressors.
5. **Explain.** GNNExplainer edge importances from the held-out fold model of the best GNN, and every
   patient → sample → altered gene → (partner) → pathway / drug / disease path over stored edges.

## Logic reasoners: beyond IF-THEN rules

RuleKit's crisp IF-THEN rules were replaced, at the mentor's request, by reasoners from three families (Markov
logic, probabilistic soft logic, reinforcement learning), all built in `src/neurosymbolic/logic/` over the same
ground atoms and folds.

**Ground atoms** (`atoms.py`). 26 clinical atoms (`NStage(p, N3)`, `Age(p) ≥ 65`, …), 57 genomic atoms
(`Mutated(p, TP53)`, `Amplified(p, MYC)` with co-amplified genes merged), and 339 relational PrimeKG atoms
`Altered(p, g) ∧ InPathway(g, w)`, `… ∧ TargetedBy(g, d)`, `… ∧ LinkedTo(g, d)`. The value of a PrimeKG atom is its
number of true groundings, that is, how many of the patient's altered genes the relation holds for. Atoms are kept at
3–90% prevalence (never by outcome) and identical carrier sets are merged. Atom order is fixed (sorted), so every
index-based learner is reproducible.

| Reasoner | Module | Method |
| --- | --- | --- |
| Survival MLN | `mln.py` | Formulas `w : Body(p) ⇒ Progress(p)`; bodies are every atom plus clinical/genomic × PrimeKG conjunctions mined on the training fold. With one query atom per patient, MLN inference factorises into `Σ w·log(1 + n(p))`. Weights are learned with the **elastic-net Cox partial likelihood** (censoring-correct) instead of the logistic pseudo-likelihood, and the L1 penalty, chosen by inner 3-fold C-index, is the structure learning. |
| Survival PSL | `psl.py` | Hinge-loss MRF on soft truth values: risk and protective rules per fact, a **collective rule** `Similar(p,q) ∧ HighRisk(q) ⇒ HighRisk(p)` over each patient's 10 most similar training patients, and a prior. Training neighbours carry censoring-aware soft targets (Kaplan–Meier conditional probability of progression by 36 months). MAP inference is unrolled projected gradient, so the rule weights are learned end to end with soft-label cross-entropy plus Cox loss. |
| RL rules | `rl.py` | A GRU policy writes a rule of up to three atoms (REINFORCE, 40,960 episodes per fold). **Stability reward**: the rule's log-rank z must agree in sign on two disjoint halves of 70% of the fold, scored by the smaller \|z\|; the remaining 30% keeps only rules that replicate (same sign, \|z\| ≥ 1). A ridge Cox model on the replicated rules gives the risk. |
| **RL-structured MLN** (engine) | `runner.py` | The RL search proposes up to 150 replicated multi-atom formulas; the survival MLN adds them to its candidates and learns all weights. RL does the MLN's structure learning, and the Cox likelihood does its weight learning. |

Five-fold out-of-fold Harrell C (mean ± sd), formulas per fold, and the share of formulas that use a PrimeKG relation:

| Reasoner | Harrell C | Uno C | Formulas / fold | PrimeKG formulas |
| --- | --- | --- | --- | --- |
| Survival MLN | 0.630 ± 0.055 | 0.617 | 32.8 | 63% |
| **RL-structured survival MLN** | 0.628 ± 0.055 | 0.615 | 35.2 | 69% |
| RuleKit LR-Rules (baseline) | 0.619 ± 0.048 | 0.623 | 7.8 | 10% |
| Survival PSL (collective) | 0.604 ± 0.059 | 0.571 | 40.0 | 63% |
| RL rules (policy gradient) | 0.563 ± 0.046 | 0.570 | 25.6 | 91% |

The two Markov logic variants lead, but their margin over RuleKit is within one fold standard deviation. **The
gain is in the reasoning, not the accuracy**: weighted, combinable, multi-hop PrimeKG formulas, seven times as
knowledge-grounded as RuleKit's rules, for example
`Altered(p, g) ∧ InPathway(g, 'VEGFA-VEGFR2 Pathway') ∧ Altered(p, g) ∧ InPathway(g, 'Ion homeostasis') ⇒ Progress(p)`
(weight −0.14 per SD, an RL proposal) or `NStage(p, N3) ∧ Altered(p, g) ∧ LinkedTo(g, 'breast ductal adenocarcinoma')
⇒ Progress(p)` (+0.15). PSL learned almost no collective weight (s⁺ ≈ 0.01), so similar patients carried no extra
outcome signal. RL alone sampled 20,000–27,000 distinct rules per fold and overfit its search even with the
replication filter; it works best as the MLN's structure learner. The engine is chosen by the one-standard-error
rule (bar 0.605) with the preference RL-structured MLN > MLN > PSL > RL > RuleKit.

## Results

Fold-mean Harrell C-index (± sd over folds), all out-of-fold:

| Model | Harrell C | Uno C (120 mo) | IBS (120 mo) |
| --- | --- | --- | --- |
| Clinical Cox, 19-field block | 0.675 ± 0.035 | 0.616 | 0.134 |
| Clinical Cox, extended | 0.667 ± 0.042 | 0.657 | 0.135 |
| **Neuro-symbolic Cox** (clinical + GNN + RL-structured MLN) | 0.664 ± 0.021 | 0.616 | 0.134 |
| Neuro-symbolic Cox, extended clinical | 0.662 ± 0.042 | 0.655 | 0.135 |
| GNN + logic, no clinical | 0.642 ± 0.056 | 0.606 | 0.139 |
| Survival MLN (logic engine alone) | 0.630 ± 0.055 | 0.617 | — |
| RL-structured MLN (logic engine alone) | 0.628 ± 0.055 | 0.615 | — |
| GraphSAGE, C-index objective | 0.622 ± 0.067 | 0.573 | — |
| RuleKit rules | 0.619 ± 0.048 | 0.623 | — |
| Survival PSL | 0.604 ± 0.059 | 0.571 | — |
| RL rules | 0.563 ± 0.046 | 0.570 | — |
| GCN, Cox | 0.612 ± 0.044 | 0.549 | — |
| GraphSAGE, Cox | 0.603 ± 0.062 | 0.556 | — |
| R-GCN, Cox | 0.586 ± 0.041 | 0.590 | — |
| Random survival forest | 0.554 ± 0.023 | 0.559 | 0.145 |

**Model choice.** The top four models are within 0.01 of each other with a fold sd of about 0.04. Under the
one-standard-error rule (best minus sd/√5 = 0.659) they are tied, and the report model is the most explainable
of them, the neuro-symbolic Cox model. In every fold its GNN and logic terms get hazard ratios of 0.94–1.09 per
sd with p > 0.28: once the clinical block is present they add no independent prognostic information. The
neuro-symbolic layer adds explanation, not accuracy. `--report-model clinical_cox_19` reports the clinical risk
instead.

**Progression month.** Estimators of "when, given progression", on the 145 held-out progressors:

| Estimator | MAE (mo) | Median |error| | Within 12 mo | Window accuracy |
| --- | --- | --- | --- | --- |
| Constant (median of training progressors) | 22.9 | 16.0 | 37% | 19% |
| Conditional median of the fused curve, τ = 120 | 28.0 | 24.6 | 23% | 39% |
| Log-normal AFT on training progressors | 20.6 | 12.9 | 48% | 34% |
| **Log-logistic AFT on training progressors** | 20.1 | 12.9 | 48% | 33% |

Chance window accuracy with four windows is about 25%. The variants were compared on the same held-out
progressors, so the chosen row is mildly optimistic.

## Neuro-symbolic path score

Each candidate path gets `NS = 0.6 × neural + 0.4 × rule`. *Neural* is the cohort percentile of its mean
GNNExplainer edge importance. *Rule* is the summed weight of the logic-engine formulas the path grounds: formulas
that fire for the patient (from the fold that held them out) and whose atoms mention the path's end node, its
altered gene, or, for a relational atom such as `Altered(p, g) ∧ InPathway(g, w)`, a gene on the path. Its
magnitude is divided by the 95th percentile and capped at 1, and its sign gives the direction. 27.5% of the 59,598
candidate paths ground at least one formula, as do 35% of the paths the reports show. The log-rank evidence of the
end fact is still printed as cohort context. A path is prominent when NS is in the cohort's top 20%.
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
- The logic engine adds reasoning, not accuracy: Markov logic matches RuleKit within noise, and RL rules alone overfit
  their search.
- Paths explain the GNN, the weaker component, and are paired with cohort evidence; they are not causal claims.
- PrimeKG drug links include enzymes, carriers, and transporters and are not treatment recommendations.
- One cohort, one seed, no external validation; the fusion uses cross-validated stacking.
