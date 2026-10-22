import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from claim_reaction.core import Market, NewsItem, parse_time
from claim_reaction.retrieval import (extract_details, hybrid_candidates,
                                      refresh_article_index, retrieve_details,
                                      _require_public_url,
                                      semantic_clusters, score_detail)
from claim_reaction.store import Store


class FakeEmbedder:
    model_id = "fake-encoder"

    def encode(self, texts, role):
        vectors = []
        for text in texts:
            if "delays" in text.lower():
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


class FakeGenerator:
    def body_chunks(self, body, prefix):
        return [body]

    def generate(self, prompt):
        if prompt.startswith("Extract"):
            if "passed committee" in prompt:
                return '{"details":["The bill passed committee"]}'
            return '{"details":["Invented fact"]}'
        return '{"strength":0.7}'


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.temp.name) / "data.sqlite"))
        self.store.put_market(Market("market", "1", "Will the law pass?", "", "yes", "v1", {}))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def add_news(self, slug, url, title, seen):
        self.store.put_news([NewsItem("gdelt-doc", url, slug, "query", url, title,
                                      None, parse_time(seen),
                                      datetime(2026, 10, 21, tzinfo=timezone.utc), {})])

    def test_global_url_dedup_and_asof_filter_applies_to_both_searches(self):
        self.add_news("market", "https://example.org/a", "Law advances", "2026-01-01T09:00:00Z")
        self.add_news("another", "https://example.org/a", "Law advances", "2026-01-01T09:05:00Z")
        self.add_news("another", "https://example.org/future", "Law passes", "2026-01-02T09:00:00Z")
        self.assertEqual(refresh_article_index(self.store), 2)
        candidates = hybrid_candidates(self.store, "Will the law pass?",
                                       parse_time("2026-01-01T12:00:00Z"),
                                       FakeEmbedder(), 10, 10, 60)
        self.assertEqual([row[0].url for row in candidates], ["https://example.org/a"])
        self.assertEqual(candidates[0][0].first_seen_at, "2026-01-01T09:00:00Z")

    def test_semantic_cluster_groups_related_items(self):
        clusters = semantic_clusters([("a", [1.0, 0.0]), ("b", [0.99, 0.1]),
                                      ("c", [0.0, 1.0])], 0.9)
        self.assertEqual(clusters, [["a", "b"], ["c"]])

    def test_extraction_rejects_unanchored_model_claim(self):
        self.assertEqual(extract_details(FakeGenerator(), "Unrelated body", 10), [])

    def test_extraction_rejects_contradictory_paraphrase_and_null_details(self):
        class BadGenerator(FakeGenerator):
            def generate(self, prompt):
                return '{"details":["The bill passed committee"]}'

        self.assertEqual(extract_details(BadGenerator(),
                                         "The bill did not pass committee.", 10), [])

        class NullGenerator(FakeGenerator):
            def generate(self, prompt):
                return '{"details":null}'

        self.assertEqual(extract_details(NullGenerator(), "Some article body", 10), [])

    def test_question_headline_and_boolean_strength_are_not_evidence(self):
        class QuestionGenerator(FakeGenerator):
            def generate(self, prompt):
                return '{"details":["Did the bill pass committee?"]}'

        self.assertEqual(extract_details(QuestionGenerator(),
                                         "Did the bill pass committee?", 10), [])

        class BooleanGenerator(FakeGenerator):
            def generate(self, prompt):
                return '{"strength":true}'

        self.assertEqual(score_detail(BooleanGenerator(), "Will the law pass?",
                                      "The bill passed committee"), 0.0)

    def test_fenced_json_array_from_local_model_is_accepted(self):
        class ArrayGenerator(FakeGenerator):
            def generate(self, prompt):
                return '```json\n["The bill passed committee"]\n```'

        self.assertEqual(extract_details(ArrayGenerator(),
                                         "The bill passed committee today.", 10),
                         [{"text": "The bill passed committee",
                           "support": "The bill passed committee"}])

    def test_local_article_urls_are_rejected(self):
        with self.assertRaises(ValueError):
            _require_public_url("http://127.0.0.1/private")

    def test_pipeline_ranks_atomic_detail_and_marks_retrospective_body(self):
        self.add_news("market", "https://example.org/a", "Law advances",
                      "2026-01-01T09:00:00Z")
        self.add_news("market", "https://example.org/b", "Law advances in Senate",
                      "2026-01-01T09:10:00Z")
        with patch("claim_reaction.retrieval.fetch_article_body",
                   return_value="The bill passed committee today.") as fetch:
            result = retrieve_details(
                self.store, "market", parse_time("2026-01-01T12:00:00Z"),
                FakeEmbedder(), FakeGenerator(), bm25_limit=10, dense_limit=10,
                fusion_k=60, max_clusters=1, bodies_per_cluster=1,
                article_similarity_threshold=0.9, detail_similarity_threshold=0.9,
                min_detail_characters=10)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(result["candidate_articles"], 2)
        self.assertEqual(result["article_clusters"], 1)
        self.assertEqual(len(result["details"]), 1)
        detail = result["details"][0]
        self.assertEqual(detail["detail"], "The bill passed committee")
        self.assertEqual(detail["strength"], 0.7)
        self.assertTrue(detail["any_retrospective_body"])
        self.assertEqual(detail["sources"][0]["support"], "The bill passed committee")
        self.assertTrue(detail["sources"][0]["url"].startswith("https://example.org/"))
        self.assertFalse(detail["point_in_time_eligible"])
        self.assertEqual(len(result["selected_cluster_members"][0]), 2)

    def test_failed_body_fetch_falls_back_to_headline_and_next_source(self):
        self.add_news("market", "https://example.org/a", "Law advances in Senate",
                      "2026-01-01T09:00:00Z")
        self.add_news("market", "https://example.org/b", "Law advances in House",
                      "2026-01-01T09:10:00Z")
        calls = []

        def fetch(url):
            calls.append(url)
            if len(calls) == 1:
                raise OSError("offline")
            return "The bill passed committee today."

        with patch("claim_reaction.retrieval.fetch_article_body", side_effect=fetch):
            result = retrieve_details(
                self.store, "market", parse_time("2026-01-01T12:00:00Z"),
                FakeEmbedder(), FakeGenerator(), bm25_limit=10, dense_limit=10,
                fusion_k=60, max_clusters=1, bodies_per_cluster=2,
                article_similarity_threshold=0.9, detail_similarity_threshold=0.9,
                min_detail_characters=10)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["selected_articles"][0]["text_scope"], "headline")
        self.assertEqual(result["selected_articles"][1]["text_scope"], "body")
        self.assertEqual(len(result["details"]), 1)

    def test_opposing_details_are_not_merged_by_semantic_clustering(self):
        self.add_news("market", "https://example.org/a", "Law committee update one",
                      "2026-01-01T09:00:00Z")
        self.add_news("market", "https://example.org/b", "Law committee update two",
                      "2026-01-01T09:10:00Z")

        class OpposingGenerator(FakeGenerator):
            def generate(self, prompt):
                negative = "did not pass" in prompt
                if prompt.startswith("Extract"):
                    detail = ("The bill did not pass committee" if negative
                              else "The bill passed committee")
                    return '{"details":[' + repr(detail).replace("'", '"') + ']}'
                return '{"strength":-0.8}' if negative else '{"strength":0.7}'

        def fetch(url):
            return ("The bill did not pass committee" if url.endswith("/b")
                    else "The bill passed committee")

        with patch("claim_reaction.retrieval.fetch_article_body", side_effect=fetch):
            result = retrieve_details(
                self.store, "market", parse_time("2026-01-01T12:00:00Z"),
                FakeEmbedder(), OpposingGenerator(), bm25_limit=10, dense_limit=10,
                fusion_k=60, max_clusters=1, bodies_per_cluster=2,
                article_similarity_threshold=0.9, detail_similarity_threshold=0.9,
                min_detail_characters=10)
        self.assertEqual({detail["direction"] for detail in result["details"]}, {"yes", "no"})

    def test_fetch_attempts_are_bounded_even_when_all_bodies_fail(self):
        for suffix in ("a", "b", "c"):
            self.add_news("market", "https://example.org/" + suffix,
                          "Law committee update " + suffix, "2026-01-01T09:00:00Z")
        with patch("claim_reaction.retrieval.fetch_article_body",
                   side_effect=OSError("offline")) as fetch:
            result = retrieve_details(
                self.store, "market", parse_time("2026-01-01T12:00:00Z"),
                FakeEmbedder(), FakeGenerator(), bm25_limit=10, dense_limit=10,
                fusion_k=60, max_clusters=1, bodies_per_cluster=2,
                article_similarity_threshold=0.9, detail_similarity_threshold=0.9,
                min_detail_characters=10)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(len(result["selected_cluster_members"][0]), 3)
        self.assertEqual(sum(member["retrieved_text_scope"] is None for member in
                             result["selected_cluster_members"][0]), 1)


if __name__ == "__main__":
    unittest.main()
