import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_body_sensitivity_links import make_body_sensitivity_links


class BodySensitivityLinkTests(unittest.TestCase):
    def test_all_snapshots_uses_only_own_event_article_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = {
                "claims": [{"claim_id": "later", "is_first_report": False,
                            "event_article_url": "https://example.test/later",
                            "members": [{"url": "https://example.test/earlier",
                                         "market_ids": ["m1"]},
                                        {"url": "https://example.test/later",
                                         "market_ids": ["m1"]}]}],
                "link_candidates": [{"claim_id": "later", "market_id": "m1"}],
            }
            body = {"body_feature_provenance": [
                {"status": "fetched", "article_url": "https://example.test/earlier",
                 "market_slug": "m1", "aggregate_signed_score": -0.8},
                {"status": "fetched", "article_url": "https://example.test/later",
                 "market_slug": "m1", "aggregate_signed_score": 0.65},
                {"status": "fetched", "article_url": "https://example.test/later",
                 "market_slug": "m2", "aggregate_signed_score": -0.5},
            ]}
            paths = [root / name for name in ("graph.json", "scores.jsonl", "body.json",
                                               "out.jsonl", "counts.json")]
            paths[0].write_text(json.dumps(graph), encoding="utf-8")
            paths[1].write_text(json.dumps({"claim_id": "later", "market_slug": "m1",
                                           "signed_score": 0.1}) + "\n", encoding="utf-8")
            paths[2].write_text(json.dumps(body), encoding="utf-8")

            counts = make_body_sensitivity_links(*paths, all_snapshots=True)

            result = json.loads(paths[3].read_text(encoding="utf-8"))
            self.assertEqual(result["signed_score"], 0.65)
            self.assertEqual(result["body_article_url"], "https://example.test/later")
            self.assertEqual(counts["substituted_links"], 1)
            self.assertEqual(counts["candidate_links"], 1)

    def test_all_snapshots_does_not_inherit_market_from_earlier_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = {
                "claims": [{"claim_id": "later", "event_article_url": "https://example.test/later",
                            "members": [{"url": "https://example.test/earlier",
                                         "market_ids": ["m1"]},
                                        {"url": "https://example.test/later",
                                         "market_ids": []}]}],
                "link_candidates": [{"claim_id": "later", "market_id": "m1"}],
            }
            body = {"body_feature_provenance": [{
                "status": "fetched", "article_url": "https://example.test/later",
                "market_slug": "m1", "aggregate_signed_score": 0.65}]}
            paths = [root / name for name in ("graph.json", "scores.jsonl", "body.json",
                                               "out.jsonl", "counts.json")]
            paths[0].write_text(json.dumps(graph), encoding="utf-8")
            paths[1].write_text(json.dumps({"claim_id": "later", "market_slug": "m1",
                                           "signed_score": 0.1}) + "\n", encoding="utf-8")
            paths[2].write_text(json.dumps(body), encoding="utf-8")

            counts = make_body_sensitivity_links(*paths, all_snapshots=True)

            result = json.loads(paths[3].read_text(encoding="utf-8"))
            self.assertEqual(result["signed_score"], 0.1)
            self.assertNotIn("body_score_substituted", result)
            self.assertEqual(counts["candidate_links"], 1)
            self.assertEqual(counts["without_matching_first_report_body"], 1)

    def test_all_snapshots_requires_matching_market_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = {
                "claims": [{"claim_id": "later", "event_article_url": "https://example.test/later",
                            "members": [{"market_ids": ["m1"]}]}],
                "link_candidates": [{"claim_id": "later", "market_id": "m1"}],
            }
            body = {"body_feature_provenance": [{
                "status": "fetched", "article_url": "https://example.test/later",
                "market_slug": "other", "aggregate_signed_score": 0.65}]}
            paths = [root / name for name in ("graph.json", "scores.jsonl", "body.json",
                                               "out.jsonl", "counts.json")]
            paths[0].write_text(json.dumps(graph), encoding="utf-8")
            paths[1].write_text(json.dumps({"claim_id": "later", "market_slug": "m1",
                                           "signed_score": 0.1}) + "\n", encoding="utf-8")
            paths[2].write_text(json.dumps(body), encoding="utf-8")

            counts = make_body_sensitivity_links(*paths, all_snapshots=True)

            result = json.loads(paths[3].read_text(encoding="utf-8"))
            self.assertEqual(result["signed_score"], 0.1)
            self.assertNotIn("body_score_substituted", result)
            self.assertEqual(counts["without_matching_first_report_body"], 1)

    def test_substitutes_only_matching_first_report_market_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = {
                "claims": [{"claim_id": "c1", "is_first_report": True,
                            "members": [{"url": "https://example.test/a",
                                         "market_ids": ["m1"]}]},
                           {"claim_id": "c2", "is_first_report": False,
                            "members": [{"url": "https://example.test/b",
                                         "market_ids": ["m2"]}]}],
                "link_candidates": [{"claim_id": "c1", "market_id": "m1"},
                                    {"claim_id": "c2", "market_id": "m2"}],
            }
            rows = [{"claim_id": "c1", "market_slug": "m1", "first_seen_at": "2026-01-01",
                     "signed_score": 0.2, "headline": "A"},
                    {"claim_id": "c1", "market_slug": "m2", "signed_score": 0.4},
                    {"claim_id": "c2", "market_slug": "m2", "signed_score": -0.2}]
            body = {"body_feature_provenance": [
                {"status": "fetched", "article_url": "https://example.test/a",
                 "market_slug": "m1", "aggregate_signed_score": -0.75,
                 "body_fetched_at": "2026-10-20T00:00:00Z"},
                {"status": "fetched", "article_url": "https://example.test/a",
                 "market_slug": "m2", "aggregate_signed_score": 0.9,
                 "body_fetched_at": "2026-10-20T00:00:00Z"},
                {"status": "fetched", "article_url": "https://example.test/b",
                 "market_slug": "m2", "aggregate_signed_score": float("nan"),
                 "body_fetched_at": "2026-10-20T00:00:00Z"},
            ]}
            paths = [root / name for name in ("graph.json", "scores.jsonl", "body.json",
                                               "out.jsonl", "counts.json")]
            paths[0].write_text(json.dumps(graph), encoding="utf-8")
            paths[1].write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            paths[2].write_text(json.dumps(body), encoding="utf-8")

            counts = make_body_sensitivity_links(*paths)

            result = [json.loads(line) for line in paths[3].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]["signed_score"], -0.75)
            self.assertEqual(result[0]["headline_signed_score"], 0.2)
            self.assertTrue(result[0]["body_score_substituted"])
            self.assertTrue(result[0]["body_retrospective"])
            self.assertEqual(result[0]["body_fetched_at"], "2026-10-20T00:00:00Z")
            self.assertEqual(result[0]["first_seen_at"], "2026-01-01")
            self.assertEqual(result[1], rows[1])
            self.assertEqual(result[2], rows[2])
            self.assertEqual(counts["headline_links"], 3)
            self.assertEqual(counts["substituted_links"], 1)
            self.assertEqual(counts["without_matching_first_report_body"], 0)
            stored_counts = json.loads(paths[4].read_text(encoding="utf-8"))
            self.assertTrue(stored_counts["retrospective"])
            self.assertEqual(stored_counts["candidate_links"], 1)


if __name__ == "__main__":
    unittest.main()
