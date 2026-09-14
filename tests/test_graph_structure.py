from pathlib import Path
import unittest

from src.clinical import link_patients_to_samples, load_clinical_data
from src.cna import load_cna_for_samples
from src.mutations import load_mutations
from src.patient_graph import build_patient_graph
from src.validation import validate_patient_graph


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "brca_tcga_pan_can_atlas_2018"


class GraphStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clinical = load_clinical_data(DATASET)
        linked, _ = link_patients_to_samples(clinical)
        cls.patient = clinical.patients.iloc[0]
        cls.samples = linked[linked["PATIENT_ID"] == cls.patient["PATIENT_ID"]]
        sample_id = cls.samples.iloc[0]["SAMPLE_ID"]
        mutations, _ = load_mutations(DATASET, {sample_id})
        calls, _ = load_cna_for_samples(DATASET, [sample_id])
        cls.graph = build_patient_graph(cls.patient, cls.samples, mutations, calls, mutations.groupby("Tumor_Sample_Barcode").size())

    def test_graph_is_valid_and_typed(self):
        result = validate_patient_graph(self.graph, self.patient, self.samples)
        self.assertTrue(result.valid, result.errors)
        node_types = {data["node_type"] for _, data in self.graph.nodes(data=True)}
        self.assertEqual(node_types, {"Patient", "Sample", "Gene"})
