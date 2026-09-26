import unittest

import pandas as pd
import torch
from torch_geometric.data import Batch

from src.dataset_fusion import (
    FEATURE_DIM,
    FusionConfig,
    PrimeSlice,
    assemble_fused_graph,
    build_catalog,
    expand_primekg_neighborhood,
    filter_primekg_frame,
)
from src.gnn_models import build_model
from src.gsl import EdgeStructureLearner, edge_report


def _panel():
    return {
        "TP53": {"entrez": "7157", "is_known_driver": True},
        "PIK3CA": {"entrez": "5290", "is_known_driver": True},
        "ERBB2": {"entrez": "2064", "is_known_driver": True},
        "MDM2": {"entrez": "4193", "is_known_driver": False},
        "EGFR": {"entrez": "1956", "is_known_driver": False},
    }


def _frame():
    rows = [
        ["protein_protein", "ppi", "7157", "gene/protein", "TP53", "4193", "gene/protein", "MDM2"],
        ["protein_protein", "ppi", "4193", "gene/protein", "MDM2", "1956", "gene/protein", "EGFR"],
        ["pathway_protein", "interacts with", "2064", "gene/protein", "ERBB2", "R-HSA-1", "pathway", "PI3K/Akt signaling"],
        ["pathway_protein", "interacts with", "5290", "gene/protein", "PIK3CA", "R-HSA-1", "pathway", "PI3K/Akt signaling"],
        ["drug_protein", "target", "DB00072", "drug", "Trastuzumab", "2064", "gene/protein", "ERBB2"],
        ["drug_protein", "target", "DB99999", "drug", "BridgeDrug", "5290", "gene/protein", "PIK3CA"],
        ["disease_protein", "associated with", "MONDO:1", "disease", "breast carcinoma", "7157", "gene/protein", "TP53"],
        ["disease_protein", "associated with", "MONDO:2", "disease", "lung cancer", "7157", "gene/protein", "TP53"],
    ]
    columns = ["relation", "display_relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]
    return filter_primekg_frame(pd.DataFrame(rows, columns=columns), set(_panel()[g]["entrez"] for g in _panel()))


def _graph(altered, hops):
    panel = _panel()
    prime = PrimeSlice.from_frame(_frame())
    catalog = build_catalog(panel, prime)
    row = {"AGE": "55", "SEX": "Female", "SUBTYPE": "BRCA_Her2", "AJCC_PATHOLOGIC_TUMOR_STAGE": "STAGE IIB", "PATH_M_STAGE": "M0"}
    sample = {entrez: {"symbol": symbol, "mut_count": 0, "amp": True, "homdel": False} for symbol, entrez in altered}
    return assemble_fused_graph("P", row, {"S": sample}, prime, panel, catalog, 0, max_hops=hops)


class CandidateHopTests(unittest.TestCase):
    def test_protein_partner_of_a_partner_is_not_crawled(self):
        prime = PrimeSlice.from_frame(_frame())
        panel = {info["entrez"] for info in _panel().values()}
        neighborhood = expand_primekg_neighborhood({"7157"}, prime, panel, max_hops=3)
        self.assertIn("4193", neighborhood.gene_hops)
        self.assertNotIn("1956", neighborhood.gene_hops)
        self.assertEqual(neighborhood.gene_hops["4193"], 1)

    def test_hop_limits_keep_relations_and_panel_boundary(self):
        _, meta1 = _graph([("ERBB2", "2064")], 1)
        _, meta2 = _graph([("ERBB2", "2064")], 2)
        _, meta3 = _graph([("ERBB2", "2064")], 3)
        self.assertNotIn("gene:PIK3CA:5290", meta1["node_ids"])
        self.assertIn("gene:PIK3CA:5290", meta2["node_ids"])
        self.assertTrue(any("Trastuzumab" in name for name in meta1["node_names"]))
        self.assertFalse(any("BridgeDrug" in name for name in meta2["node_names"]))
        self.assertTrue(any("BridgeDrug" in name for name in meta3["node_names"]))
        self.assertFalse(any("lung" in name.lower() for name in meta3["node_names"]))
        self.assertTrue(set(meta2["relations"]).issubset({
            "has_sample", "has_high_level_amplification", "pathway_protein", "drug_protein", "protein_protein", "disease_protein"
        }))
        self.assertTrue(all(hop <= 2 for hop in meta2["edge_hops"]))
        self.assertTrue(all(hop <= 3 for hop in meta3["edge_hops"]))

    def test_outcome_is_not_an_input_to_structure_learning(self):
        config = FusionConfig(mode="gsl", gsl_enabled=True, gsl_patient_context=True, gsl_hidden=8, gsl_relation_dim=4)
        learner = EdgeStructureLearner(config)
        self.assertNotIn("y", learner.forward.__code__.co_varnames)


class StructureLearningTests(unittest.TestCase):
    def _data(self, age="40"):
        data, meta = _graph([("TP53", "7157")], 2)
        data.x[0, 6] = float(age)  # age sits in the clinical block; exact index is not required for a difference
        return data, meta

    def test_scores_match_edges_and_observed_edges_stay_one(self):
        data, meta = self._data()
        config = FusionConfig(mode="gsl", gsl_enabled=True, gsl_patient_context=True, gsl_hidden=8, gsl_relation_dim=4)
        learner = EdgeStructureLearner(config)
        weight = learner(data.x, data.edge_index, data.rel_code, data.is_patient, data.hop)
        self.assertEqual(tuple(weight.shape), (data.edge_index.size(1),))
        self.assertTrue(torch.all((weight >= 0) & (weight <= 1)))
        observed = data.rel_code < 4
        self.assertTrue(torch.allclose(weight[observed], torch.ones(int(observed.sum()))))
        report = edge_report(data, meta, weight, threshold=0.5)
        self.assertTrue(all(row["relation"] in meta["relations"] for row in report))
        self.assertTrue(any(row["knowledge"] and row["hop"] >= 1 for row in report))

    def test_kept_hops_start_open_and_the_extra_hop_starts_closed(self):
        data, _ = _graph([("ERBB2", "2064")], 3)
        config = FusionConfig(mode="gsl", gsl_enabled=True, fixed_hops=2, gsl_patient_context=True, gsl_hidden=8, gsl_relation_dim=4)
        learner = EdgeStructureLearner(config)
        weight = learner(data.x, data.edge_index, data.rel_code, data.is_patient, data.hop)
        knowledge = data.rel_code >= 4
        near = knowledge & (data.hop <= 2)
        far = knowledge & (data.hop >= 3)
        self.assertGreater(int(near.sum()), 0)
        self.assertGreater(int(far.sum()), 0)
        self.assertTrue(torch.allclose(weight[near], torch.ones(int(near.sum()))))
        self.assertLess(float(weight[far].detach().mean()), 0.2)
        self.assertTrue(torch.allclose(weight[far], learner.last_hop_gate[far]))
        weight[far].sum().backward()
        self.assertIsNotNone(learner.hop_bias.grad)
        self.assertGreater(float(learner.hop_bias.grad[3].abs()), 0)

    def test_patient_context_changes_primekg_scores(self):
        config = FusionConfig(mode="gsl", gsl_enabled=True, gsl_patient_context=True, gsl_hidden=8, gsl_relation_dim=4)
        learner = EdgeStructureLearner(config)
        hidden = int(config.gsl_hidden)
        # Open a single path from the first clinical feature to the edge logit.
        torch.nn.init.zeros_(learner.patient_encoder.weight)
        torch.nn.init.zeros_(learner.patient_encoder.bias)
        torch.nn.init.zeros_(learner.edge_mlp[0].weight)
        torch.nn.init.zeros_(learner.edge_mlp[0].bias)
        torch.nn.init.zeros_(learner.edge_mlp[-1].weight)
        torch.nn.init.zeros_(learner.edge_mlp[-1].bias)
        with torch.no_grad():
            learner.patient_encoder.weight[0, 0] = 1
            learner.edge_mlp[0].bias.fill_(1)
            learner.edge_mlp[0].weight[:, -hidden] = 1
            learner.edge_mlp[-1].weight[0, 0] = 1
        first, _ = self._data("30")
        second, _ = self._data("90")
        # Put the age difference on every feature of the patient node so the context encoder sees it.
        first.x[0, 6:25] = 0
        second.x[0, 6:25] = 1
        left = learner(first.x, first.edge_index, first.rel_code, first.is_patient, first.hop)
        right = learner(second.x, second.edge_index, second.rel_code, second.is_patient, second.hop)
        knowledge = first.rel_code >= 4
        self.assertFalse(torch.allclose(left[knowledge], right[knowledge]))
        muted = FusionConfig(mode="gsl", gsl_enabled=True, gsl_patient_context=False, gsl_hidden=8, gsl_relation_dim=4)
        muted_learner = EdgeStructureLearner(muted)
        muted_learner.load_state_dict(learner.state_dict(), strict=False)
        same_left = muted_learner(first.x, first.edge_index, first.rel_code, first.is_patient, first.hop)
        same_right = muted_learner(second.x, second.edge_index, second.rel_code, second.is_patient, second.hop)
        self.assertTrue(torch.allclose(same_left, same_right))

    def test_batch_and_backward_keep_graphs_separate(self):
        first, _ = self._data("30")
        second, _ = self._data("90")
        batch = Batch.from_data_list([first, second])
        model = build_model(
            "gcn",
            FEATURE_DIM,
            num_embeddings=12,
            mlp_in_dim=8,
            fusion=FusionConfig(mode="gsl", gsl_enabled=True, gsl_hidden=8, gsl_relation_dim=4),
        )
        weight = model.edge_weights(batch.x, batch.edge_index, batch.rel_code, batch.is_patient, batch.hop, batch.batch)
        self.assertEqual(tuple(weight.shape), (batch.edge_index.size(1),))
        torch.nn.init.constant_(model.classifier.weight, 0.01)
        logit = model(batch.x, batch.edge_index, batch.emb_id, batch.is_patient, batch.batch, rel_code=batch.rel_code, hop=batch.hop, focus=batch.focus)
        self.assertEqual(tuple(logit.shape), (2,))
        logit.sum().backward()
        self.assertIsNotNone(model.gsl.edge_mlp[-1].bias.grad)
        self.assertGreater(float(model.gsl.edge_mlp[-1].bias.grad.abs()), 0)
        self.assertIsNotNone(model.gsl.relation_logit.grad)
        self.assertGreater(float(model.gsl.relation_logit.grad.abs().sum()), 0)
