from pathlib import Path
import unittest

from src.cna import load_cna_for_samples


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "brca_tcga_pan_can_atlas_2018"


class CnaMappingTests(unittest.TestCase):
    def test_only_non_neutral_discrete_calls_are_emitted(self):
        sample = "TCGA-3C-AAAU-01"
        calls, issues = load_cna_for_samples(DATASET, [sample])
        self.assertFalse(calls.empty)
        self.assertEqual(set(calls["SAMPLE_ID"]), {sample})
        self.assertTrue(set(calls["cna_value"]).issubset({-2, -1, 1, 2}))
        self.assertFalse(calls.duplicated(["Hugo_Symbol", "Entrez_Gene_Id", "SAMPLE_ID"]).any())
        self.assertTrue(any("duplicate" in issue["reason"] or "conflicting" in issue["reason"] for issue in issues))
