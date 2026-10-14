import json
import unittest

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test

from src.dataset_fusion import PrimeSlice
from src.neurosymbolic.features import PredicateTable, predicate_table
from src.neurosymbolic.kg_evidence import logrank_table
from src.neurosymbolic.paths import K_MAX, K_MIN, MAX_PER_GENE, select_top_paths
from src.neurosymbolic.rules import parse_rule, readable_condition, rulekit_frame
from src.neurosymbolic.timing import conditional_quantile, timeline_group, window_code


class TimelineTests(unittest.TestCase):
    def test_groups_follow_recorded_month_and_censoring(self):
        self.assertEqual(timeline_group(1, 3.0), "A")
        self.assertEqual(timeline_group(1, 12.0), "A")
        self.assertEqual(timeline_group(1, 12.1), "B")
        self.assertEqual(timeline_group(1, 36.0), "C")
        self.assertEqual(timeline_group(1, 80.0), "D")
        self.assertEqual(timeline_group(0, 5.0), "N")
        self.assertEqual(timeline_group(1, float("nan")), "N")
        self.assertEqual(window_code(0.0), "A")

    def test_conditional_median_given_progression_before_tau(self):
        grid = np.arange(0.0, 121.0)
        # Failure rises linearly to 0.2 at 100 months, then stays flat.
        curve = 1.0 - np.clip(grid / 100.0, 0, 1) * 0.2
        self.assertAlmostEqual(conditional_quantile(curve, grid, 100.0, 0.5), 50.0)
        self.assertAlmostEqual(conditional_quantile(curve, grid, 120.0, 0.5), 50.0)
        self.assertTrue(np.isnan(conditional_quantile(np.ones_like(grid), grid, 120.0, 0.5)))


def _paths(rows):
    frame = pd.DataFrame(rows)
    frame["patient_id"] = "P1"
    frame["nodes"] = json.dumps([])
    return frame


class TopPathTests(unittest.TestCase):
    def test_k_is_capped_when_many_paths_are_prominent(self):
        rows = [{"ns_score": 0.9 - i * 0.01, "altered_gene": f"G{i}", "terminal_id": f"pathway:{i}"} for i in range(9)]
        chosen = select_top_paths(_paths(rows), cutoff=0.5)
        self.assertEqual(len(chosen), K_MAX)
        self.assertTrue(chosen["prominent"].all())
        self.assertEqual(chosen["rank"].tolist(), list(range(1, K_MAX + 1)))

    def test_top_up_to_k_min_when_few_paths_are_prominent(self):
        rows = [{"ns_score": score, "altered_gene": f"G{i}", "terminal_id": f"pathway:{i}"} for i, score in enumerate([0.8, 0.3, 0.2, 0.1])]
        chosen = select_top_paths(_paths(rows), cutoff=0.5)
        self.assertEqual(len(chosen), K_MIN)
        self.assertEqual(int(chosen["prominent"].sum()), 1)

    def test_gene_cap_and_unique_end_nodes(self):
        rows = [{"ns_score": 0.9 - i * 0.01, "altered_gene": "ERBB2", "terminal_id": f"pathway:{i}"} for i in range(4)]
        rows.append({"ns_score": 0.95, "altered_gene": "TP53", "terminal_id": "pathway:0"})
        chosen = select_top_paths(_paths(rows), cutoff=0.0)
        self.assertLessEqual(int((chosen["altered_gene"] == "ERBB2").sum()), MAX_PER_GENE)
        self.assertEqual(chosen["terminal_id"].nunique(), len(chosen))


class EvidenceTests(unittest.TestCase):
    def test_vectorised_logrank_matches_lifelines(self):
        rng = np.random.default_rng(3)
        n = 200
        group = rng.random(n) < 0.4
        time = rng.exponential(np.where(group, 30, 60))
        event = (rng.random(n) < 0.7).astype(int)
        table = logrank_table(time, event, group[:, None])
        reference = logrank_test(time[group], time[~group], event[group], event[~group])
        self.assertAlmostEqual(float(table["z"].iloc[0]) ** 2, reference.test_statistic, places=6)
        self.assertGreater(float(table["hazard_ratio"].iloc[0]), 1.0)


def _table() -> PredicateTable:
    frame = pd.DataFrame(
        {"age": [50.0, 70.0], "n_stage": ["N0", "missing"], "mut_TP53": ["yes", "no"], "kg_pathway_R-1": ["no", "yes"]},
        index=["P1", "P2"],
    )
    return PredicateTable(
        frame=frame,
        labels={"age": "Age", "n_stage": "N stage", "mut_TP53": "TP53 mutated", "kg_pathway_R-1": "altered gene in pathway 'X'"},
        kinds={"age": "clinical", "n_stage": "clinical", "mut_TP53": "genomic", "kg_pathway_R-1": "kg_pathway"},
        members={"mut_TP53": {"TP53"}, "kg_pathway_R-1": {"pathway:R-1"}},
    )


class RuleTextTests(unittest.TestCase):
    def test_binary_predicates_enter_rulekit_as_yes_or_missing(self):
        frame = rulekit_frame(_table())
        self.assertEqual(frame.loc["P1", "mut_TP53"], "yes")
        self.assertTrue(pd.isna(frame.loc["P2", "mut_TP53"]))
        self.assertTrue(pd.isna(frame.loc["P2", "n_stage"]))

    def test_rule_conditions_read_as_clinical_language(self):
        table = _table()
        self.assertEqual(readable_condition("age", "<41.50, inf)", table), "Age ≥ 41.5")
        self.assertEqual(readable_condition("age", "(-inf, 53.37)", table), "Age < 53.4")
        self.assertEqual(readable_condition("kg_pathway_R-1", "{yes}", table), "altered gene in pathway 'X'")
        conditions = parse_rule("IF n_stage = {N3} AND mut_TP53 = {yes} THEN ", table)
        self.assertEqual([c.text for c in conditions], ["N stage = N3", "TP53 mutated"])


class PredicateLeakageTests(unittest.TestCase):
    def test_predicates_exclude_outcomes_and_passenger_mutations(self):
        patients = [f"P{i}" for i in range(40)]
        clinical = pd.DataFrame(
            {
                "age": np.linspace(30, 80, 40),
                "aneuploidy_score": 5.0,
                "tmb_nonsynonymous": 1.0,
                "t_stage": "T2",
                "n_stage": "N0",
                "m_stage": "M0",
                "ajcc_stage": "II",
                "subtype": "LumA",
                "histology": "ductal",
                "radiation_therapy": "yes",
            },
            index=patients,
        )
        rows = []
        for i, patient in enumerate(patients):
            if i % 2 == 0:
                rows.append({"patient_id": patient, "gene": "TP53", "entrez": "7157", "kind": "mut", "driver": True})
            if i % 3 == 0:
                rows.append({"patient_id": patient, "gene": "TTN", "entrez": "7273", "kind": "mut", "driver": False})
        alterations = pd.DataFrame(rows)
        prime = PrimeSlice(pathway_genes={"R-1": {"7157"}}, pathway_names={"R-1": "TP53 pathway"}, gene_pathways={"7157": {"R-1"}})
        table = predicate_table(clinical, alterations, prime)
        self.assertIn("mut_TP53", table.frame.columns)
        self.assertNotIn("mut_TTN", table.frame.columns)
        self.assertFalse({"PFS_STATUS", "PFS_MONTHS", "OS_STATUS"} & set(table.frame.columns))


if __name__ == "__main__":
    unittest.main()
