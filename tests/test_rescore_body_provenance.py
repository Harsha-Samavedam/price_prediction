import unittest

from scripts.rescore_body_provenance import rescore_experiment


class RescoreBodyProvenanceTests(unittest.TestCase):
    def test_rescores_stored_probabilities_and_preserves_other_provenance(self):
        data = {
            "rows": 1,
            "body_feature_provenance": [{
                "status": "fetched",
                "claim_id": "c1",
                "body_sha256": "abc",
                "selected_sentences": [
                    {"sentence": "yes evidence", "signed_score": -0.7,
                     "probabilities": {
                         "yes": {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},
                         "no": {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1},
                     }},
                    {"sentence": "ambiguous evidence", "signed_score": 0.7,
                     "probabilities": {
                         "yes": {"entailment": 0.2, "neutral": 0.5, "contradiction": 0.3},
                         "no": {"entailment": 0.3, "neutral": 0.4, "contradiction": 0.3},
                     }},
                ],
                "signed_detail_scores": [-0.7, 0.7],
                "aggregate_signed_score": 0.0,
            }, {"status": "failed", "reason": "timeout"}],
        }

        result = rescore_experiment(data)

        fetched = result["body_feature_provenance"][0]
        self.assertEqual([s["signed_score"] for s in fetched["selected_sentences"]], [0.8, 0.0])
        self.assertEqual(fetched["signed_detail_scores"], [0.8, 0.0])
        self.assertEqual(fetched["aggregate_signed_score"], 0.4)
        self.assertEqual(fetched["scorer_version"], "entailment_v2")
        self.assertTrue(fetched["retrospective"])
        self.assertEqual(fetched["claim_id"], "c1")
        self.assertEqual(fetched["body_sha256"], "abc")
        self.assertEqual(result["body_feature_provenance"][1], {"status": "failed", "reason": "timeout"})
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["scorer_version"], "entailment_v2")
        self.assertTrue(result["retrospective"])
        # The source mapping and nested sentence values remain unchanged.
        self.assertEqual(data["body_feature_provenance"][0]["selected_sentences"][0]["signed_score"], -0.7)


if __name__ == "__main__":
    unittest.main()
