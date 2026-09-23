import unittest

import pandas as pd

from src.dataset_fusion import (
    FEATURE_DIM,
    PrimeSlice,
    assemble_fused_graph,
    build_catalog,
    clinical_block,
    collapse_m_stage,
    collapse_stage,
    feature_names,
    filter_primekg_frame,
    normalize_entrez,
)
from src.labels import OUTCOME_FIELDS


def _panel():
    return {
        "TP53": {"entrez": "7157", "is_known_driver": True},
        "PIK3CA": {"entrez": "5290", "is_known_driver": True},
        "ERBB2": {"entrez": "2064", "is_known_driver": True},
    }


def _prime_frame():
    rows = [
        ["protein_protein", "ppi", "7157", "gene/protein", "TP53", "5290", "gene/protein", "PIK3CA"],
        ["protein_protein", "ppi", "7157", "gene/protein", "TP53", "1956", "gene/protein", "EGFR"],
        ["pathway_protein", "interacts with", "5290", "gene/protein", "PIK3CA", "R-HSA-1", "pathway", "PI3K/Akt signaling"],
        ["pathway_protein", "interacts with", "2064", "gene/protein", "ERBB2", "R-HSA-1", "pathway", "PI3K/Akt signaling"],
        ["drug_protein", "target", "DB00072", "drug", "Trastuzumab", "2064", "gene/protein", "ERBB2"],
        ["disease_protein", "associated with", "MONDO:1", "disease", "breast carcinoma", "7157", "gene/protein", "TP53"],
        ["disease_protein", "associated with", "MONDO:2", "disease", "lung cancer", "7157", "gene/protein", "TP53"],
    ]
    columns = ["relation", "display_relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]
    return filter_primekg_frame(pd.DataFrame(rows, columns=columns), {"7157", "5290", "2064"})


class FusionTests(unittest.TestCase):
    def test_entrez_normalization_joins_panel_float_to_primekg_id(self):
        self.assertEqual(normalize_entrez(5290.0), "5290")
        self.assertEqual(normalize_entrez("5290"), "5290")
        self.assertIsNone(normalize_entrez("inf"))
        self.assertIsNone(normalize_entrez("Infinity"))
        frame = _prime_frame()
        self.assertTrue((frame["x_entrez"] == "5290").any() or (frame["y_entrez"] == "5290").any())
        self.assertFalse(frame["y_id"].eq("1956").any() | frame["x_id"].eq("1956").any())
        self.assertFalse(frame["x_name"].str.contains("lung", case=False).any())
        self.assertFalse(frame["y_name"].str.contains("lung", case=False).any())

    def test_stage_and_m_stage_collapse(self):
        self.assertEqual(collapse_stage("STAGE IIB"), "II")
        self.assertEqual(collapse_stage("STAGE IA"), "I")
        self.assertEqual(collapse_stage("STAGE IIIA"), "III")
        self.assertEqual(collapse_stage("STAGE IV"), "IV")
        self.assertEqual(collapse_stage("STAGE X"), "X")
        self.assertEqual(collapse_m_stage("cM0 (i+)"), "CM0")
        self.assertEqual(collapse_m_stage("M0"), "M0")

    def test_outcome_columns_do_not_change_features(self):
        base = {"AGE": "61", "SEX": "Female", "SUBTYPE": "BRCA_LumA", "AJCC_PATHOLOGIC_TUMOR_STAGE": "STAGE IIA", "PATH_M_STAGE": "M0"}
        leaked = {**base, "PFS_STATUS": "1:PROGRESSION", "PFS_MONTHS": "12.4", "OS_STATUS": "1:DECEASED"}
        self.assertTrue((clinical_block(base) == clinical_block(leaked)).all())
        self.assertTrue(set(feature_names()).isdisjoint(OUTCOME_FIELDS))
        self.assertEqual(len(feature_names()), FEATURE_DIM)

    def test_fused_graph_links_erbb2_to_pik3ca_through_the_pathway(self):
        panel = _panel()
        prime = PrimeSlice.from_frame(_prime_frame())
        catalog = build_catalog(panel, prime)
        sample_genes = {"TCGA-TEST-01": {"2064": {"symbol": "ERBB2", "mut_count": 0, "amp": True, "homdel": False}}}
        row = {"AGE": "55", "SEX": "Female", "SUBTYPE": "BRCA_Her2", "AJCC_PATHOLOGIC_TUMOR_STAGE": "STAGE IIB", "PATH_M_STAGE": "M0", "PFS_STATUS": "1:PROGRESSION"}
        data, meta = assemble_fused_graph("TCGA-TEST", row, sample_genes, prime, panel, catalog, 1)
        self.assertIn("gene:ERBB2:2064", meta["node_ids"])
        self.assertIn("gene:PIK3CA:5290", meta["node_ids"])
        self.assertNotIn("gene:TP53:7157", meta["node_ids"])
        self.assertTrue(any(node_id.startswith("pathway:R-HSA-1") for node_id in meta["node_ids"]))
        self.assertTrue(any("Trastuzumab" in name for name in meta["node_names"]))
        self.assertFalse(any("lung" in name.lower() for name in meta["node_names"]))
        self.assertEqual(meta["stored_fields"], ["AGE", "SEX", "SUBTYPE", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_M_STAGE"])
        self.assertIn("5290", meta["grafted_entrez"])
        self.assertEqual(int(data.is_patient.sum()), 1)
        self.assertEqual(data.x.shape[1], FEATURE_DIM)
        self.assertEqual(int(data.is_reverse.sum()), int((~data.is_reverse).sum()))
