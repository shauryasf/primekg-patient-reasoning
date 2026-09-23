import unittest

import pandas as pd

from src.labels import (
    CLINICAL_FEATURE_FIELDS,
    OUTCOME_FIELDS,
    assert_no_outcome_leakage,
    labels_from_patients,
    landmark_pfs_label,
    parse_new_tumor_event,
    parse_pfs_status,
)


class LabelParsingTests(unittest.TestCase):
    def test_pfs_status_mapping(self):
        self.assertEqual(parse_pfs_status("1:PROGRESSION"), 1)
        self.assertEqual(parse_pfs_status("0:CENSORED"), 0)
        self.assertIsNone(parse_pfs_status(""))
        self.assertIsNone(parse_pfs_status("NA"))
        self.assertIsNone(parse_pfs_status("[Not Available]"))

    def test_new_tumor_event_mapping_drops_incomplete_rows(self):
        self.assertEqual(parse_new_tumor_event("Yes"), 1)
        self.assertEqual(parse_new_tumor_event("No"), 0)
        self.assertIsNone(parse_new_tumor_event(""))
        self.assertIsNone(parse_new_tumor_event("Unknown"))

    def test_incomplete_patients_are_excluded(self):
        patients = pd.DataFrame(
            [
                {"PATIENT_ID": "P1", "PFS_STATUS": "1:PROGRESSION", "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT": "Yes"},
                {"PATIENT_ID": "P2", "PFS_STATUS": "0:CENSORED", "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT": ""},
                {"PATIENT_ID": "P3", "PFS_STATUS": "", "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT": "No"},
            ]
        )
        pfs = labels_from_patients(patients, "pfs")
        event = labels_from_patients(patients, "new_tumor_event")
        self.assertEqual(pfs.set_index("PATIENT_ID")["y"].to_dict(), {"P1": 1, "P2": 0})
        self.assertEqual(event.set_index("PATIENT_ID")["y"].to_dict(), {"P1": 1, "P3": 0})

    def test_landmark_uses_months_only_to_define_the_window(self):
        self.assertEqual(landmark_pfs_label("1:PROGRESSION", 12), 1)
        self.assertEqual(landmark_pfs_label("1:PROGRESSION", 36), 1)
        self.assertEqual(landmark_pfs_label("1:PROGRESSION", 40), 0)
        self.assertEqual(landmark_pfs_label("0:CENSORED", 36), 0)
        self.assertEqual(landmark_pfs_label("0:CENSORED", 80), 0)
        self.assertIsNone(landmark_pfs_label("0:CENSORED", 10))
        self.assertIsNone(landmark_pfs_label("1:PROGRESSION", None))
        patients = pd.DataFrame(
            [
                {"PATIENT_ID": "P1", "PFS_STATUS": "1:PROGRESSION", "PFS_MONTHS": 10},
                {"PATIENT_ID": "P2", "PFS_STATUS": "0:CENSORED", "PFS_MONTHS": 4},
                {"PATIENT_ID": "P3", "PFS_STATUS": "0:CENSORED", "PFS_MONTHS": 50},
                {"PATIENT_ID": "P4", "PFS_STATUS": "1:PROGRESSION", "PFS_MONTHS": 48},
            ]
        )
        labels = labels_from_patients(patients, "pfs_36m")
        self.assertEqual(labels.set_index("PATIENT_ID")["y"].to_dict(), {"P1": 1, "P3": 0, "P4": 0})
        self.assertNotIn("PFS_MONTHS", CLINICAL_FEATURE_FIELDS)

    def test_clinical_feature_fields_exclude_outcomes(self):
        self.assertTrue(set(CLINICAL_FEATURE_FIELDS).isdisjoint(OUTCOME_FIELDS))
        assert_no_outcome_leakage(CLINICAL_FEATURE_FIELDS)
        with self.assertRaises(ValueError):
            assert_no_outcome_leakage([*CLINICAL_FEATURE_FIELDS, "PFS_STATUS"])
