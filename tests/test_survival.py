import unittest

import numpy as np
import pandas as pd
import torch

from src.labels import LANDMARK_MONTHS, OUTCOME_FIELDS
from src.survival import (
    build_objective,
    concordance_audit,
    harrell_c_index,
    pair_indices,
    survival_targets_from_patients,
)


def patient_table() -> pd.DataFrame:
    """Five patients: two events, two censored, one censored without a follow-up time."""
    return pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2", "P3", "P4", "P5"],
            "PFS_STATUS": ["1:PROGRESSION", "1:PROGRESSION", "0:CENSORED", "0:CENSORED", "0:CENSORED"],
            "PFS_MONTHS": ["10", "50", "40", "20", ""],
        }
    )


class SurvivalTargetTests(unittest.TestCase):
    def test_landmark_window_censors_at_the_horizon(self):
        targets = survival_targets_from_patients(patient_table(), "pfs_36m")
        rows = dict(zip(targets.patient_ids, zip(targets.time.tolist(), targets.event.tolist())))
        self.assertEqual(rows["P1"], (10.0, 1))
        self.assertEqual(rows["P2"], (float(LANDMARK_MONTHS), 0))
        self.assertEqual(rows["P3"], (float(LANDMARK_MONTHS), 0))
        self.assertNotIn("P4", rows)
        self.assertNotIn("P5", rows)
        self.assertEqual(targets.horizon, float(LANDMARK_MONTHS))
        self.assertEqual(targets.audit()["events"], 1)

    def test_raw_endpoint_keeps_censored_follow_up_and_flags_missing_time(self):
        targets = survival_targets_from_patients(patient_table(), "pfs")
        rows = dict(zip(targets.patient_ids, zip(targets.time.tolist(), targets.event.tolist())))
        self.assertEqual(rows["P4"], (20.0, 0))
        self.assertTrue(np.isnan(rows["P5"][0]))
        self.assertEqual(targets.audit()["with_follow_up_time"], 4)
        self.assertEqual(targets.audit()["without_follow_up_time"], 1)

    def test_endpoint_without_an_event_time_is_refused(self):
        with self.assertRaises(ValueError):
            survival_targets_from_patients(patient_table(), "new_tumor_event")
        with self.assertRaises(ValueError):
            survival_targets_from_patients(patient_table(), "os")

    def test_outcome_columns_stay_banned_as_features(self):
        self.assertIn("PFS_STATUS", OUTCOME_FIELDS)
        self.assertIn("PFS_MONTHS", OUTCOME_FIELDS)


class PairAndCIndexTests(unittest.TestCase):
    def test_pairs_use_earlier_events_and_skip_censored_leads(self):
        earlier, later = pair_indices(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 1]))
        self.assertEqual(sorted(zip(earlier.tolist(), later.tolist())), [(0, 1), (0, 2)])
        earlier, later = pair_indices(np.array([2.0, 5.0]), np.array([1, 0]))
        self.assertEqual((earlier.tolist(), later.tolist()), ([0], [1]))
        earlier, later = pair_indices(np.array([5.0, 2.0]), np.array([0, 1]))
        self.assertEqual((earlier.tolist(), later.tolist()), ([1], [0]))

    def test_untimed_patients_are_not_paired(self):
        earlier, later = pair_indices(np.array([1.0, np.nan, 3.0]), np.array([1, 1, 0]))
        self.assertEqual((earlier.tolist(), later.tolist()), ([0], [2]))

    def test_tied_events_are_comparable_and_a_tied_risk_scores_half(self):
        time = np.array([4.0, 4.0])
        event = np.array([1, 1])
        self.assertEqual(pair_indices(time, event)[0].size, 1)
        self.assertEqual(harrell_c_index(time, event, np.array([1.0, 1.0])), 0.5)
        self.assertEqual(harrell_c_index(time, event, np.array([2.0, 1.0])), 1.0)
        self.assertEqual(harrell_c_index(time, event, np.array([1.0, 2.0])), 0.0)

    def test_perfect_and_reversed_rankings(self):
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([1, 1, 1, 0])
        self.assertEqual(harrell_c_index(time, event, np.array([4.0, 3.0, 2.0, 1.0])), 1.0)
        self.assertEqual(harrell_c_index(time, event, np.array([1.0, 2.0, 3.0, 4.0])), 0.0)
        self.assertAlmostEqual(harrell_c_index(time, event, np.array([3.0, 4.0, 1.0, 2.0])), 2 / 3)

    def test_a_fold_without_pairs_reports_nan_instead_of_a_number(self):
        time = np.array([1.0, 2.0])
        event = np.array([0, 0])
        self.assertTrue(np.isnan(harrell_c_index(time, event, np.array([1.0, 2.0]))))

    def test_audit_counts_pairs_per_event(self):
        audit = concordance_audit(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 0, 1, 0]))
        self.assertEqual(audit["comparable_pairs"], 4)
        self.assertEqual(audit["events"], 2)
        self.assertEqual(audit["censored"], 2)


class ObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time = torch.tensor([10.0, 20.0, 30.0, 40.0])
        self.event = torch.tensor([1.0, 1.0, 1.0, 0.0])
        self.target = torch.tensor([1.0, 1.0, 1.0, 0.0])

    def test_concordance_objective_prefers_the_correct_order_and_still_learns(self):
        objective = build_objective("cindex", np.array([1, 1, 1, 0]))
        correct = torch.tensor([3.0, 2.0, 1.0, 0.0], requires_grad=True)
        wrong = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
        self.assertLess(
            float(objective(correct, self.target, self.time, self.event)),
            float(objective(wrong, self.target, self.time, self.event)),
        )
        objective(correct, self.target, self.time, self.event).backward()
        self.assertGreater(float(correct.grad.abs().sum()), 0.0)

    def test_cox_objective_prefers_the_correct_order(self):
        objective = build_objective("cox", np.array([1, 1, 1, 0]))
        correct = torch.tensor([3.0, 2.0, 1.0, 0.0])
        wrong = torch.tensor([0.0, 1.0, 2.0, 3.0])
        self.assertLess(
            float(objective(correct, self.target, self.time, self.event)),
            float(objective(wrong, self.target, self.time, self.event)),
        )

    def test_survival_objectives_refuse_to_run_without_targets(self):
        for name in ("cindex", "cox"):
            objective = build_objective(name, np.array([1, 0]))
            with self.assertRaises(ValueError):
                objective(torch.tensor([0.5, -0.5]), torch.tensor([1.0, 0.0]))

    def test_bce_stays_available_and_ignores_time_and_event(self):
        objective = build_objective("bce", np.array([1, 1, 1, 0]))
        self.assertTrue(torch.isfinite(objective(torch.tensor([0.5, -0.5]), torch.tensor([1.0, 0.0]))))

    def test_batch_without_comparable_pairs_returns_a_connected_zero(self):
        objective = build_objective("cindex", np.array([1, 0]))
        risk = torch.tensor([0.4, -0.4], requires_grad=True)
        loss = objective(risk, torch.tensor([1.0, 0.0]), torch.tensor([5.0, 5.0]), torch.tensor([0.0, 0.0]))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(risk.grad)

    def test_bce_anchor_keeps_patients_without_a_pair_in_the_gradient(self):
        objective = build_objective("cindex", np.array([1, 0]), aux_bce=0.5)
        risk = torch.tensor([0.4, -0.4], requires_grad=True)
        loss = objective(risk, torch.tensor([1.0, 0.0]), torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 0.0]))
        loss.backward()
        self.assertGreater(abs(float(risk.grad[1])), 0.0)

    def test_unknown_objective_is_refused(self):
        with self.assertRaises(ValueError):
            build_objective("auroc", np.array([1, 0]))


if __name__ == "__main__":
    unittest.main()
