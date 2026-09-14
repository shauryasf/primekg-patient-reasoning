from pathlib import Path
import unittest

from src.mutations import load_mutations, mutation_record_counts


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "brca_tcga_pan_can_atlas_2018"


class MutationCountTests(unittest.TestCase):
    def test_maf_row_count_is_deterministic_and_not_a_claimed_clinical_field(self):
        samples = ["TCGA-3C-AAAU-01", "TCGA-A8-A08G-01"]
        mutations, _ = load_mutations(DATASET, set(samples))
        counts = mutation_record_counts(mutations, samples)
        self.assertEqual(counts.to_dict(), {"TCGA-3C-AAAU-01": 50, "TCGA-A8-A08G-01": 62})
