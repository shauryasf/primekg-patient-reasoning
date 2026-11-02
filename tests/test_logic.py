import json
import unittest

import numpy as np
import pandas as pd
import torch

from src.neurosymbolic.logic import mln, psl, rl
from src.neurosymbolic.logic.atoms import AtomTable, soft_targets
from src.neurosymbolic.logic.runner import choose_engine
from src.neurosymbolic.paths import rule_scores


def synthetic_atoms(n: int = 400, seed: int = 0):
    """Atom 0 doubles the hazard, atom 1 is noise, atom 2 is a PrimeKG count atom."""
    rng = np.random.default_rng(seed)
    risky = rng.random(n) < 0.4
    noise = rng.random(n) < 0.5
    pathway = rng.integers(0, 3, n).astype(np.float32)
    hazard = np.exp(1.2 * risky)
    true_time = rng.exponential(60 / hazard)
    censor = rng.uniform(10, 120, n)
    time = np.minimum(true_time, censor)
    event = (true_time <= censor).astype(int)
    atoms = AtomTable(
        patient_ids=[f"P{i}" for i in range(n)],
        values=np.column_stack([risky, noise, pathway]).astype(np.float32),
        names=["risky", "noise", "kg_pathway:R-1"],
        labels=["risky", "noise", "altered gene in pathway 'R-1'"],
        kinds=["clinical", "genomic", "kg_pathway"],
        formulas=["Risky(p)", "Noise(p)", "Altered(p, g) ∧ InPathway(g, 'R-1')"],
        variables=["risky", "noise", "kg_pathway:R-1"],
        members=[set(), {"NOISE"}, {"pathway:R-1"}],
    )
    folds = [(np.arange(n)[np.arange(n) % 4 != k], np.arange(n)[np.arange(n) % 4 == k]) for k in range(4)]
    return atoms, time, event, folds


class SurvivalMLNTests(unittest.TestCase):
    def test_learns_positive_weight_for_planted_risk_atom(self):
        atoms, time, event, folds = synthetic_atoms()
        train, test = folds[0]
        risk, weights, info = mln.fit_fold(atoms, time, event, train, test)
        self.assertEqual(len(risk), len(test))
        unary = weights[weights["name"] == "risky"]
        self.assertFalse(unary.empty)
        self.assertGreater(float(unary["weight_per_sd"].iloc[0]), 0)
        self.assertGreater(info["candidates"], 3)

    def test_grounding_counts_multiply_atoms_of_a_conjunction(self):
        atoms, *_ = synthetic_atoms(n=10)
        counts = mln.grounding_counts(atoms, [mln.Formula("pair", "", (0, 2))])
        expected = np.log1p(atoms.values[:, 0] * atoms.values[:, 2])
        np.testing.assert_allclose(counts[:, 0], expected)


class SoftLogicTests(unittest.TestCase):
    def test_inferred_truth_values_stay_in_unit_interval(self):
        program = psl.SoftLogicProgram(3)
        facts = torch.rand(20, 3)
        similarity = torch.rand(20, 4)
        neighbours = torch.rand(20, 4)
        risk = program.infer(facts, similarity, neighbours)
        self.assertTrue(bool(((risk >= 0) & (risk <= 1)).all()))

    def test_soft_targets_handle_censoring(self):
        time = np.array([10.0, 50.0, 20.0, 40.0, 5.0])
        event = np.array([1, 0, 0, 1, 1])
        targets = soft_targets(time, event, np.arange(5))
        self.assertEqual(targets[0], 1.0)
        self.assertEqual(targets[1], 0.0)
        self.assertTrue(0.0 <= targets[2] <= 1.0)
        self.assertEqual(targets[3], 0.0)


class RLRuleTests(unittest.TestCase):
    def test_coverage_is_the_conjunction_of_chosen_atoms(self):
        binary = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=bool)
        rules = np.array([[0, 1, -1], [2, -1, -1]])
        covered = rl._coverage(binary, rules)
        np.testing.assert_array_equal(covered[:, 0], [True, False, True])
        np.testing.assert_array_equal(covered[:, 1], [False, True, True])


class EngineChoiceTests(unittest.TestCase):
    def test_one_standard_error_rule_prefers_richer_reasoner(self):
        summary = pd.DataFrame({"method": ["mln", "rl_mln", "rl"], "c_mean": [0.64, 0.63, 0.55], "c_std": [0.05, 0.02, 0.05]})
        chosen, info = choose_engine(summary, folds=5)
        self.assertEqual(chosen, "rl_mln")
        self.assertNotIn("rl", info["eligible"])


class PathGroundingTests(unittest.TestCase):
    def test_path_matches_formula_through_a_gene_on_the_path(self):
        atom_frame = pd.DataFrame({"atom": ["kg_pathway:R-1", "risky"], "members": ["pathway:R-1", ""], "kind": ["kg_pathway", "clinical"]})
        atom_binary = np.array([[True, True]])
        weights = pd.DataFrame({"fold": [0, 0], "atoms": ["kg_pathway:R-1", "risky"], "weight_per_sd": [0.3, 0.5], "formula": ["kg", "risky"]})
        nodes = [{"id": "patient:P0", "type": "patient", "name": "P0"}, {"id": "gene:TP53:7157", "type": "gene", "name": "TP53"}, {"id": "pathway:R-9", "type": "pathway", "name": "other"}]
        candidates = pd.DataFrame({"patient_id": ["P0"], "fold": [0], "terminal_id": ["pathway:R-9"], "altered_gene": ["TP53"], "nodes": [json.dumps(nodes)]})
        without = rule_scores(candidates, weights, atom_frame, atom_binary, {"P0": 0})
        with_genes = rule_scores(candidates, weights, atom_frame, atom_binary, {"P0": 0}, [{"7157"}, set()])
        self.assertEqual(int(without["rule_count"].iloc[0]), 0)
        self.assertEqual(int(with_genes["rule_count"].iloc[0]), 1)
        self.assertAlmostEqual(float(with_genes["rule_score"].iloc[0]), 0.3)


if __name__ == "__main__":
    unittest.main()
