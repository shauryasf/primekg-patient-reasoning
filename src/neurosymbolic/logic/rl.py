"""Reinforcement-learned survival rules (RL-Rules).

A policy network writes a rule one atom at a time: a GRU reads the atoms chosen
so far and outputs a distribution over the next atom (clinical, genomic, or a
relational PrimeKG atom) or STOP, up to three atoms. The rule body is the
conjunction of the chosen atoms, and the episode's reward is how well that body
separates progression-free survival:

    reward = min(|z_A|, |z_B|) / Z_CAP     if the two signs agree, else 0

where ``z_A`` and ``z_B`` are log-rank statistics of the covered patients against
the rest, computed on two disjoint halves of 70% of the training fold. Requiring the
same direction on both halves is a *stability reward*: a rule earns credit only
for separation that replicates, which is what greedy covering on the full fold
(LR-Rules) cannot check. Rules covering fewer than 4% or more than 90% of the fold
earn nothing; each extra atom costs ``LENGTH_COST`` (``KG_BONUS`` can favour
PrimeKG atoms and is off by default). The policy is trained with REINFORCE, a
moving-average baseline, and an entropy bonus. Sampling tens of thousands of rules
makes the best training reward a multiple-testing artefact, so the remaining 30%
of the fold, which no reward ever saw, keeps only rules whose separation has the
same sign there with ``|z| >= 1``. The surviving distinct rules form the rule set,
and a ridge Cox model on their activations turns them into a risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from torch import nn

from ..kg_evidence import logrank_table
from .atoms import AtomTable

MAX_ATOMS = 3
BATCH = 256
ITERATIONS = 160
Z_CAP = 5.0
LENGTH_COST = 0.03
KG_BONUS = 0.0
ENTROPY = 0.01
MIN_SHARE, MAX_SHARE = 0.04, 0.90
RULE_SET = 30
OVERLAP = 0.8
VALIDATION_SHARE = 0.3
CANDIDATES = 3000
MIN_VALIDATION_Z = 1.0


class RulePolicy(nn.Module):
    def __init__(self, n_atoms: int, dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_atoms + 1, dim)
        self.cell = nn.GRUCell(dim, hidden)
        self.head = nn.Linear(hidden, n_atoms + 1)
        self.start = n_atoms

    def sample(self, batch: int, exclusive: torch.Tensor, greedy: bool = False):
        """Sample ``batch`` rules. Returns atom indices (``-1`` = unused), log-probs, entropies."""
        n_atoms = self.start
        state = torch.zeros(batch, self.cell.hidden_size)
        token = torch.full((batch,), self.start, dtype=torch.long)
        chosen = torch.full((batch, MAX_ATOMS), -1, dtype=torch.long)
        mask = torch.zeros(batch, n_atoms + 1, dtype=torch.bool)
        alive = torch.ones(batch, dtype=torch.bool)
        log_prob = torch.zeros(batch)
        entropy = torch.zeros(batch)
        for step in range(MAX_ATOMS):
            state = self.cell(self.embedding(token), state)
            logits = self.head(state)
            blocked = mask.clone()
            if step == 0:
                blocked[:, n_atoms] = True
            logits = logits.masked_fill(blocked, -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(1) if greedy else distribution.sample()
            log_prob = log_prob + torch.where(alive, distribution.log_prob(action), torch.zeros(batch))
            entropy = entropy + torch.where(alive, distribution.entropy(), torch.zeros(batch))
            stop = action == n_atoms
            take = alive & ~stop
            chosen[take, step] = action[take]
            mask[take] = mask[take] | exclusive[action[take]]
            alive = alive & ~stop
            token = torch.where(alive, action, token)
            if not alive.any():
                break
        return chosen, log_prob, entropy


def _coverage(binary: np.ndarray, rules: np.ndarray) -> np.ndarray:
    covered = np.ones((binary.shape[0], len(rules)), dtype=bool)
    for slot in range(rules.shape[1]):
        atom = rules[:, slot]
        use = atom >= 0
        covered[:, use] &= binary[:, atom[use]]
    return covered


def _rewards(rules: np.ndarray, binary: np.ndarray, time: np.ndarray, event: np.ndarray, halves, is_kg: np.ndarray):
    covered = _coverage(binary, rules)
    share = covered.mean(axis=0)
    zs = []
    for half in halves:
        zs.append(np.nan_to_num(logrank_table(time[half], event[half], covered[half].astype(float))["z"].to_numpy()))
    z_a, z_b = zs
    agree = np.sign(z_a) == np.sign(z_b)
    stable = np.minimum(np.abs(z_a), np.abs(z_b))
    length = (rules >= 0).sum(axis=1)
    reward = np.where(agree, np.minimum(stable, Z_CAP) / Z_CAP, 0.0)
    reward = reward - LENGTH_COST * np.maximum(length - 1, 0)
    reward = reward + KG_BONUS * np.array([is_kg[r[r >= 0]].any() for r in rules])
    reward = np.where((share >= MIN_SHARE) & (share <= MAX_SHARE), reward, 0.0)
    return reward, covered


def propose_rules(atoms: AtomTable, time: np.ndarray, event: np.ndarray, train: np.ndarray, seed: int = 0, limit: int = RULE_SET):
    """Train the policy on ``train`` and return up to ``limit`` replicated rules (atom tuples) with their search stats.

    The search runs on one CPU thread: multi-threaded reductions change the last
    bits of the GRU's outputs from run to run, which changes the sampled rules.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _propose_rules(atoms, time, event, train, seed, limit)
    finally:
        torch.set_num_threads(threads)


def _propose_rules(atoms: AtomTable, time: np.ndarray, event: np.ndarray, train: np.ndarray, seed: int, limit: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    binary = atoms.binary
    variables = np.array(atoms.variables)
    is_kg = np.array([kind.startswith("kg_") for kind in atoms.kinds])
    n_atoms = binary.shape[1]
    exclusive = torch.zeros(n_atoms, n_atoms + 1, dtype=torch.bool)
    for index in range(n_atoms):
        exclusive[index, :n_atoms] = torch.tensor(variables == variables[index])
    # 70% of the fold drives the reward (split into two halves); the other 30% is
    # never rewarded and is used only to keep rules whose separation replicates.
    order = rng.permutation(len(train))
    cut = int(len(order) * (1 - VALIDATION_SHARE))
    search, validation = order[:cut], order[cut:]
    halves = [search[: len(search) // 2], search[len(search) // 2 :]]
    train_binary, train_time, train_event = binary[train], time[train], event[train]
    policy = RulePolicy(n_atoms)
    optimiser = torch.optim.Adam(policy.parameters(), lr=3e-3)
    baseline = 0.0
    archive: dict[tuple[int, ...], float] = {}
    for _ in range(ITERATIONS):
        rules, log_prob, entropy = policy.sample(BATCH, exclusive)
        rule_array = rules.numpy()
        reward, _ = _rewards(rule_array, train_binary, train_time, train_event, halves, is_kg)
        for rule, value in zip(rule_array, reward):
            key = tuple(sorted(int(a) for a in rule if a >= 0))
            archive[key] = max(archive.get(key, 0.0), float(value))
        reward_t = torch.tensor(reward, dtype=torch.float32)
        baseline = 0.9 * baseline + 0.1 * float(reward_t.mean())
        loss = -((reward_t - baseline) * log_prob).mean() - ENTROPY * entropy.mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    candidates = [key for key, value in sorted(archive.items(), key=lambda item: -item[1])[:CANDIDATES] if value > 0]
    candidate_cover = np.vstack([np.all(train_binary[:, list(key)], axis=1) for key in candidates]).T
    search_z = np.nan_to_num(logrank_table(train_time[search], train_event[search], candidate_cover[search].astype(float))["z"].to_numpy())
    validation_z = np.nan_to_num(logrank_table(train_time[validation], train_event[validation], candidate_cover[validation].astype(float))["z"].to_numpy())
    replicated = (np.sign(search_z) == np.sign(validation_z)) & (np.abs(validation_z) >= MIN_VALIDATION_Z)
    score = np.where(replicated, np.minimum([archive[key] for key in candidates], np.minimum(np.abs(validation_z), Z_CAP) / Z_CAP), 0.0)
    ranked = [(candidates[i], float(score[i])) for i in np.argsort(-score)]
    selected: list[tuple[int, ...]] = []
    covers: list[np.ndarray] = []
    for key, value in ranked:
        if value <= 0 or len(selected) >= limit:
            break
        cover = np.all(train_binary[:, list(key)], axis=1)
        if any((cover & other).sum() / max((cover | other).sum(), 1) >= OVERLAP for other in covers):
            continue
        selected.append(key)
        covers.append(cover)
    return selected, archive, {"episodes": ITERATIONS * BATCH, "distinct_rules": len(archive), "rules": len(selected)}


def fit_fold(atoms: AtomTable, time: np.ndarray, event: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int = 0):
    """RL rules alone: a ridge Cox model on the activations of the replicated rules."""
    train = train[np.isfinite(time[train]) & (time[train] > 0)]
    selected, archive, info = propose_rules(atoms, time, event, train, seed)
    binary = atoms.binary
    is_kg = np.array([kind.startswith("kg_") for kind in atoms.kinds])
    train_time, train_event = time[train], event[train]
    activation = np.vstack([np.all(binary[:, list(key)], axis=1) for key in selected]).T.astype(float)
    frame = pd.DataFrame(activation[train], columns=[f"r{i}" for i in range(len(selected))])
    keep = [column for column in frame.columns if frame[column].std() > 0]
    fitter = CoxPHFitter(penalizer=0.5).fit(frame[keep].assign(T=train_time, E=train_event), duration_col="T", event_col="E")
    test_frame = pd.DataFrame(activation[test], columns=frame.columns)[keep]
    risk_test = fitter.predict_log_partial_hazard(test_frame).to_numpy()
    full_z = np.nan_to_num(logrank_table(train_time, train_event, activation[train])["z"].to_numpy())
    rows = []
    for index, key in enumerate(selected):
        column = f"r{index}"
        weight = float(fitter.params_.get(column, 0.0))
        rows.append(
            {
                "rule": " ∧ ".join(atoms.formulas[a] for a in key) + " ⇒ Progress(p)",
                "atoms": "|".join(atoms.names[a] for a in key),
                "reward": float(archive[key]),
                "weight": weight,
                "logrank_z": float(full_z[index]),
                "support": int(activation[train][:, index].sum()),
                "uses_kg": bool(is_kg[list(key)].any()),
                "length": len(key),
            }
        )
    weights = pd.DataFrame(rows).sort_values("weight", key=np.abs, ascending=False)
    return risk_test, weights, info
