import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from claim_reaction.body_forecast import (add_body_features, compare_body_and_headline,
                                         extract_verbatim_sentences, _score_pair,
                                         select_usable_body_rows, signed_entailment_score)
from claim_reaction.forecast import ForecastRow


class FakeEmbedder:
    model_id = "fixture-e5"
    def encode(self, texts, role):
        return [[1., 0.] if role == "query" or "bill" in t.lower() else [0., 1.] for t in texts]


class FakeClassifier:
    model_id = "fixture-nli"
    def probabilities(self, premise, hypothesis):
        text = premise.lower()
        if "considered" in text:
            return {"entailment": .01, "neutral": .49, "contradiction": .50}
        if "passed" in text or "failed" in text:
            entailed = ("passed" in text and "passed" in hypothesis.lower()) or (
                "failed" in text and "failed" in hypothesis.lower())
            return {"entailment": .9 if entailed else .01, "neutral": .05,
                    "contradiction": .05 if entailed else .94}
        return {"entailment": .01, "neutral": .97, "contradiction": .02}


class FakeGenerator:
    def __init__(self, model_id="fixture-model", max_input_tokens=100, max_new_tokens=20):
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.calls = 0

    def body_chunks(self, body, prefix):
        return [body]

    def generate(self, prompt):
        self.calls += 1
        if "Extract distinct" in prompt:
            return json.dumps({"details": ["The bill passed the Senate.",
                                           "The bill failed its final vote."]})
        return json.dumps({"strength": 0.6 if "passed" in prompt else -0.2})


def row(claim, slug, url, price=.4, delta=.1):
    return ForecastRow(claim, slug, "Will the bill pass?", "Bill headline", url,
                       "gdelt-doc", "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z",
                       "2026-02-01T00:00:00Z", 60, price, delta, "t0", "t1")


def run_features(rows, docs, cache, hypotheses, **kwargs):
    return add_body_features(rows, docs, FakeEmbedder(), FakeClassifier(), hypotheses,
                             cache, **kwargs)


def write_hypotheses(path, slugs=("m", "m1", "m2", "a", "b", "c", "d")):
    Path(path).write_text(json.dumps({slug: {"yes": "The bill passed.",
                                             "no": "The bill failed."} for slug in slugs}),
                          encoding="utf-8")


class BodyForecastTests(unittest.TestCase):
    def test_signed_entailment_score_from_saved_probabilities(self):
        self.assertEqual(signed_entailment_score({
            "yes": {"entailment": .91, "neutral": .06, "contradiction": .03},
            "no": {"entailment": .02, "neutral": .95, "contradiction": .03},
        }), .91)
        self.assertEqual(signed_entailment_score({
            "yes": {"entailment": .02, "neutral": .03, "contradiction": .95},
            "no": {"entailment": .88, "neutral": .09, "contradiction": .03},
        }), -.88)
        self.assertEqual(signed_entailment_score({
            "yes": {"entailment": .01, "neutral": .02, "contradiction": .97},
            "no": {"entailment": .01, "neutral": .97, "contradiction": .02},
        }), 0)
        tied = {"yes": {"entailment": .8, "neutral": .1, "contradiction": .1},
                "no": {"entailment": .8, "neutral": .1, "contradiction": .1}}
        self.assertEqual(signed_entailment_score(tied), 0)

    def test_exact_sentence_extraction_and_directional_gating(self):
        body = "The bill passed the Senate. The bill passed the Senate. The bill was considered."
        sentences = extract_verbatim_sentences(body, 10)
        self.assertEqual(sentences, ["The bill passed the Senate.", "The bill was considered."])
        self.assertTrue(all(s in body for s in sentences))
        classifier = FakeClassifier()
        yes, _ = _score_pair(classifier, "The bill passed the Senate.", "The bill passed.", "The bill failed.")
        no, _ = _score_pair(classifier, "The bill failed its vote.", "The bill passed.", "The bill failed.")
        ambiguous, _ = _score_pair(classifier, "The bill was considered.", "The bill passed.", "The bill failed.")
        irrelevant, _ = _score_pair(classifier, "The Fed met today.", "The bill passed.", "The bill failed.")
        self.assertGreater(yes, 0)
        self.assertLess(no, 0)
        self.assertEqual(ambiguous, 0)
        self.assertEqual(irrelevant, 0)

        class ContradictionOnlyNo:
            def probabilities(self, premise, hypothesis):
                if "passed" in hypothesis.lower():
                    return {"entailment": .002, "neutral": .003, "contradiction": .995}
                return {"entailment": .002, "neutral": .994, "contradiction": .004}

        contradiction_only, raw = _score_pair(
            ContradictionOnlyNo(), "The Fed met today.",
            "The bill passed.", "The bill did not pass.")
        self.assertEqual(contradiction_only, 0)
        self.assertEqual(raw["yes"]["contradiction"], .995)
        self.assertEqual(raw["no"]["contradiction"], .004)
        ambiguous_both_contradicted, _ = _score_pair(
            ContradictionOnlyNo(), "The bill was considered.",
            "The bill passed.", "The bill did not pass.")
        self.assertEqual(ambiguous_both_contradicted, 0)

    def test_sentence_extraction_preserves_decimal_and_common_abbreviations(self):
        body = ("The U.S. report showed 3.75 inches of rain. "
                "Dr. Lee said e.g. the total was accurate. Markets rose.")
        sentences = extract_verbatim_sentences(body, min_characters=1)
        self.assertEqual(sentences, [
            "The U.S. report showed 3.75 inches of rain.",
            "Dr. Lee said e.g. the total was accurate.",
            "Markets rose.",
        ])
        self.assertTrue(all(sentence in body for sentence in sentences))

    def test_rank_cache_key_changes_with_sentence_selection_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp) / "documents.jsonl"
            docs.write_text(json.dumps({"market_slug": "m", "article_url": "u",
                                        "body": "The bill passed the Senate. The bill failed its final vote.",
                                        "body_fetch_status": "fetched"}) + "\n", encoding="utf-8")
            hypotheses = Path(temp) / "hyp.json"
            write_hypotheses(hypotheses)
            cache = Path(temp) / "cache.json"
            run_features([row("c", "m", "u")], docs, cache, hypotheses,
                         min_detail_characters=10, max_sentences_per_article=1)
            run_features([row("c", "m", "u")], docs, cache, hypotheses,
                         min_detail_characters=20, max_sentences_per_article=2)
            saved = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["rankings"]), 2)

    def test_join_aggregation_provenance_and_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp) / "documents.jsonl"
            docs.write_text(json.dumps({"market_slug": "m", "article_url": "u",
                                        "body": "The bill passed the Senate. The bill failed its final vote.",
                                        "body_fetched_at": "2026-01-01T00:02:00Z",
                                        "status": "success"}) + "\n", encoding="utf-8")
            cache = Path(temp) / "cache.json"
            hypotheses = Path(temp) / "hyp.json"
            write_hypotheses(hypotheses)
            features, names, info = run_features([row("c", "m", "u")], docs,
                                                  cache, hypotheses, min_detail_characters=10)
            self.assertEqual(names, ["pre_price", "body_mean_signed_nli_score"])
            self.assertEqual(features.shape, (1, 2))
            self.assertAlmostEqual(features[0, 0], .4)
            self.assertAlmostEqual(features[0, 1], 0.0)
            self.assertEqual(info[0]["detail_count"], 2)
            self.assertEqual(len(info[0]["signed_detail_scores"]), 2)
            self.assertEqual(info[0]["body_fetched_at"], "2026-01-01T00:02:00Z")
            scores = json.loads(cache.read_text())["scores"]
            run_features([row("c", "m", "u")], docs, cache, hypotheses, min_detail_characters=10)
            self.assertEqual(json.loads(cache.read_text())["scores"], scores)

    def test_missing_body_gets_zero_score(self):
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp) / "documents.jsonl"
            docs.write_text("", encoding="utf-8")
            hypotheses = Path(temp) / "hyp.json"
            write_hypotheses(hypotheses)
            features, _, info = run_features([row("c", "m", "u")], docs,
                                             Path(temp) / "cache.json", hypotheses)
        self.assertTrue(np.array_equal(features, np.array([[.4, 0.0]])))
        self.assertEqual(info[0]["status"], "missing")

    def test_cache_isolated_by_classifier_model(self):
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp) / "documents.jsonl"
            docs.write_text(json.dumps({"market_slug": "m", "article_url": "u",
                                        "body": "The bill passed the Senate. The bill failed its final vote.",
                                        "body_fetch_status": "fetched"}) + "\n", encoding="utf-8")
            cache = Path(temp) / "cache.json"
            hypotheses = Path(temp) / "hyp.json"
            write_hypotheses(hypotheses)
            _, _, provenance = run_features([row("c", "m", "u")], docs, cache, hypotheses)
            self.assertEqual(provenance[0]["model"], "fixture-nli")

    def test_completed_body_cache_is_checkpointed_before_later_failure(self):
        class InterruptingClassifier(FakeClassifier):
            def probabilities(self, premise, hypothesis):
                if "second distinct" in premise:
                    raise RuntimeError("simulated interruption")
                return super().probabilities(premise, hypothesis)

        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp) / "documents.jsonl"
            documents = [
                {"market_slug": "m1", "article_url": "u1",
                 "body": "The bill passed the Senate.", "body_fetch_status": "fetched"},
                {"market_slug": "m2", "article_url": "u2",
                 "body": "The second distinct text.", "body_fetch_status": "fetched"},
            ]
            docs.write_text("".join(json.dumps(item) + "\n" for item in documents),
                            encoding="utf-8")
            cache_path = Path(temp) / "cache.json"
            hypotheses = Path(temp) / "hyp.json"
            write_hypotheses(hypotheses)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                add_body_features([row("c1", "m1", "u1"), row("c2", "m2", "u2")],
                                  docs, FakeEmbedder(), InterruptingClassifier(), hypotheses,
                                  cache_path, 10)
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["sentences"]), 2)
            self.assertEqual(len(saved["scores"]), 1)
            self.assertEqual(list(Path(temp).glob(".cache.json.*.tmp")), [])

    def test_same_market_rows_evaluated_for_both_models(self):
        rows = [row(f"{market}{i}", market, f"{market}-{i}", .2 + i * .2,
                    .01 * (i - 1)) for market in ("a", "b", "c") for i in range(3)]
        features = np.array([[r.pre_price, float(i % 3) / 10] for i, r in enumerate(rows)])
        report = compare_body_and_headline(rows, features, features, alphas=(.1, 1.0))
        self.assertTrue(report["same_rows"])
        self.assertEqual(report["body_based"]["aggregate_mae_by_horizon"],
                         report["headline_only"]["aggregate_mae_by_horizon"])
        self.assertEqual(len(report["body_based"]["folds"]), 3)
        self.assertIn("zero_change_mae", report["body_based"]["folds"][0])

    def test_selection_uses_only_successful_sample_bodies_and_parity(self):
        rows = [row(f"{market}{i}", market, f"{market}-{i}", .2 + i * .2,
                    # Deliberately distinct labels: they do not affect inclusion.
                    (-.9 if market == "a" else .8))
                for market in ("a", "b", "c", "d") for i in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp) / "documents.jsonl"
            samples = [
                {"market_slug": market, "article_url": f"{market}-0", "body": "a body",
                 "body_fetch_status": "fetched"}
                for market in ("a", "b", "c")
            ]
            samples += [{"market_slug": "d", "article_url": "d-0", "body": None,
                         "body_fetch_status": "failed"}]
            docs.write_text("".join(json.dumps(item) + "\n" for item in samples), encoding="utf-8")
            chosen, coverage = select_usable_body_rows(rows, docs)
            self.assertEqual([(r.market_slug, r.article_url) for r in chosen],
                             [(m, f"{m}-0") for m in ("a", "b", "c")])
            self.assertEqual(coverage["selected_forecast_rows"], 3)
            self.assertEqual(coverage["failed_or_empty_pairs_with_forecast_rows"], 1)
            self.assertEqual(coverage["forecast_pairs_absent_from_sample"], 4)
            # Both feature matrices are made from this same selected row object list.
            features = np.array([[r.pre_price, .1] for r in chosen])
            report = compare_body_and_headline(chosen, features, features, alphas=(.1,))
            body_ids = {p["claim_id"] for f in report["body_based"]["folds"] for p in f["predictions"]}
            headline_ids = {p["claim_id"] for f in report["headline_only"]["folds"] for p in f["predictions"]}
            self.assertEqual(body_ids, headline_ids)

    def test_preflight_fails_with_fewer_than_three_markets_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp) / "documents.jsonl"
            docs.write_text(json.dumps({"market_slug": "a", "article_url": "a-0",
                                        "body": "body", "body_fetch_status": "fetched"}) + "\n",
                            encoding="utf-8")
            hypotheses = Path(temp) / "hyp.json"
            write_hypotheses(hypotheses)
            with self.assertRaisesRegex(ValueError, "Fewer than three markets"):
                selected, _ = select_usable_body_rows(
                    [row("a0", "a", "a-0"), row("b0", "b", "b-0")], docs)
                run_features(selected, docs, Path(temp) / "cache.json", hypotheses)


if __name__ == "__main__":
    unittest.main()
