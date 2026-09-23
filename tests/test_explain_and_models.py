import unittest

import torch

from src.dataset_fusion import FEATURE_DIM, RELATION_NAMES
from src.explain import enumerate_explanation_paths
from src.gnn_models import MODEL_NAMES, build_model


class ExplainAndModelTests(unittest.TestCase):
    def test_path_uses_only_existing_edges(self):
        node_ids = ["patient:P", "sample:S", "gene:PIK3CA:5290", "pathway:R-HSA-1"]
        node_types = ["patient", "sample", "gene", "pathway"]
        node_names = ["P", "S", "PIK3CA", "PI3K/Akt signaling"]
        altered = [False, False, True, False]
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        is_reverse = torch.tensor([False, True, False, True, False, True])
        relations = ["has_sample", "has_sample", "has_mutation", "has_mutation", "pathway_protein", "pathway_protein"]
        scores = torch.tensor([0.1, 0.1, 0.8, 0.8, 0.9, 0.9])
        paths = enumerate_explanation_paths(node_ids, node_types, node_names, altered, edge_index, is_reverse, relations, scores)
        self.assertEqual(len(paths), 1)
        self.assertIn("has_mutation", paths[0]["path"])
        self.assertIn("pathway_protein", paths[0]["path"])
        self.assertIn("PI3K/Akt signaling", paths[0]["path"])
        self.assertNotIn("confers_resistance", paths[0]["path"])

    def test_each_architecture_returns_one_logit_per_graph(self):
        for name in MODEL_NAMES:
            model = build_model(name, FEATURE_DIM, num_embeddings=6, mlp_in_dim=8)
            model.eval()
            if name == "mlp":
                logits = model(torch.randn(4, 8))
                self.assertEqual(tuple(logits.shape), (4,))
                continue
            x = torch.randn(3, FEATURE_DIM)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            emb = torch.tensor([0, 1, 2])
            is_patient = torch.tensor([True, False, False])
            logits = model(x, edge_index, emb, is_patient)
            self.assertEqual(tuple(logits.shape), (1,))
            if name == "gatv2":
                _, attentions = model(x, edge_index, emb, is_patient, return_attention=True)
                self.assertEqual(attentions[0].shape[0], edge_index.size(1))
        self.assertEqual(len(RELATION_NAMES), 8)
