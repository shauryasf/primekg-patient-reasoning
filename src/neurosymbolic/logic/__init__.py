"""Survival-aware symbolic reasoners over patient and PrimeKG facts.

Three reasoners replace plain IF-THEN rule induction, each over the same ground
atoms and the same folds:

- :mod:`.mln`: a Markov logic network whose first-order formulas are grounded in
  the fused graph and whose weights are learned with the Cox partial likelihood;
- :mod:`.psl`: probabilistic soft logic, a hinge-loss Markov random field with a
  collective rule that propagates risk between similar patients;
- :mod:`.rl`: a policy-gradient rule generator that composes conjunctive rules
  and is rewarded by their log-rank separation of progression-free survival.
"""
