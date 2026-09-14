from pathlib import Path
import unittest

from src.mutations import load_mutations


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "brca_tcga_pan_can_atlas_2018"


class MutationMappingTests(unittest.TestCase):
    def test_known_sample_records_retain_source_sample_and_gene(self):
        sample = "TCGA-3C-AAAU-01"
        mutations, malformed = load_mutations(DATASET, {sample})
        self.assertFalse(malformed)
        self.assertEqual(len(mutations), 50)
        self.assertEqual(set(mutations["Tumor_Sample_Barcode"]), {sample})
        self.assertFalse(mutations["Hugo_Symbol"].isna().any())
        self.assertTrue(mutations["mutation_record_id"].is_unique)
