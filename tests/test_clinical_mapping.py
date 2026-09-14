from pathlib import Path
import unittest

from src.clinical import link_patients_to_samples, load_clinical_data


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "brca_tcga_pan_can_atlas_2018"


class ClinicalMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clinical = load_clinical_data(DATASET)
        cls.linked, cls.unmapped = link_patients_to_samples(cls.clinical)

    def test_unique_and_complete_clinical_mapping(self):
        self.assertEqual(self.clinical.patients["PATIENT_ID"].nunique(), 1084)
        self.assertEqual(self.clinical.samples["SAMPLE_ID"].nunique(), 1084)
        self.assertTrue(self.unmapped.empty)
        self.assertTrue((self.linked.groupby("PATIENT_ID")["SAMPLE_ID"].size() == 1).all())
