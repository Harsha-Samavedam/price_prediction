import json

from scripts.score_graph_links import score_links
from scripts.score_historical_headlines import score_headlines


class FakeClassifier:
    model_id = "fake-nli"

    def __init__(self):
        self.calls = 0

    def probabilities(self, premise, hypothesis):
        self.calls += 1
        return {"entailment": 0.8 if "new" in hypothesis else 0.1,
                "neutral": 0.1, "contradiction": 0.1}


def test_changed_hypothesis_invalidates_prior_and_output_cache(tmp_path):
    graph = {"claims": [{"claim_id": "c", "cluster_id": "c", "is_first_report": True,
                         "representative_title": "headline", "event_time": "2026-01-01T00:00:00Z"}],
             "link_candidates": [{"claim_id": "c", "market_id": "m",
                                  "cosine_similarity": 0.8, "retrieved_member": True}]}
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))
    hypotheses_path = tmp_path / "hypotheses.json"
    hypotheses_path.write_text(json.dumps({"m": {"yes": "new yes", "no": "no"}}))
    stale_hyp = {"yes": "new yes", "no": "no"}
    stale_row = {"source": "gdelt-doc", "source_id": "1", "market_slug": "m",
                 "url": "https://example.test/a", "title": "headline",
                 "provider_seen_at": "2026-01-01T00:00:00Z", "model_id": "fake-nli",
                 "signed_score": -0.9, "probabilities": {
                     "yes": {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},
                     "no": {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}}}
    import hashlib
    stale_row["score_key"] = hashlib.sha256(json.dumps([
        "fake-nli", stale_hyp["yes"], stale_hyp["no"], stale_row["source"],
        stale_row["source_id"], stale_row["market_slug"], stale_row["url"],
        stale_row["title"], stale_row["provider_seen_at"]],
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    prior_path = tmp_path / "prior.jsonl"
    prior_path.write_text(json.dumps(stale_row) + "\n")
    output_path = tmp_path / "links.jsonl"
    classifier = FakeClassifier()
    first = score_links(str(graph_path), str(hypotheses_path), str(output_path),
                        classifier, str(prior_path))
    assert first["new_nli_scores"] == 0
    assert first["reused_headline_scores"] == 1
    assert classifier.calls == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[-1]["signed_score"] == 0.8
    second = score_links(str(graph_path), str(hypotheses_path), str(output_path),
                         classifier, str(prior_path))
    assert second["cached_links"] == 1
    assert classifier.calls == 0
    hypotheses_path.write_text(json.dumps({"m": {"yes": "another new yes", "no": "no"}}))
    third = score_links(str(graph_path), str(hypotheses_path), str(output_path),
                        classifier, str(prior_path))
    assert third["new_nli_scores"] == 1
    assert classifier.calls == 2
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[-1]["input_key"] != rows[0]["input_key"]
    assert rows[-1]["scorer_version"] == "entailment_v2"


def test_all_snapshots_scores_later_reports_without_future_member_leakage(tmp_path):
    class RecordingClassifier(FakeClassifier):
        def __init__(self):
            super().__init__()
            self.premises = []

        def probabilities(self, premise, hypothesis):
            self.premises.append(premise)
            return super().probabilities(premise, hypothesis)

    graph = {
        "claims": [
            {"claim_id": "c0", "cluster_id": "cluster", "is_first_report": True,
             "representative_title": "first", "event_time": "2026-01-01T10:00:00Z",
             "earliest_seen_at": "2026-01-01T10:00:00Z", "utc_day": "2026-01-01",
             "members": [{"url": "u0", "title": "first",
                          "provider_seen_at": "2026-01-01T10:00:00Z"}]},
            {"claim_id": "c1", "cluster_id": "cluster", "is_first_report": False,
             "representative_title": "first", "event_title": "new event headline",
             "event_time": "2026-01-01T11:00:00Z",
             "earliest_seen_at": "2026-01-01T11:00:00Z", "utc_day": "2026-01-01",
             "members": [{"url": "u0", "title": "first",
                          "provider_seen_at": "2026-01-01T10:00:00Z"},
                         {"url": "u1", "title": "second",
                          "provider_seen_at": "2026-01-01T11:00:00Z"}]},
        ],
        "link_candidates": [
            {"claim_id": "c0", "market_id": "m", "cosine_similarity": 0.8,
             "retrieved_member": True},
            {"claim_id": "c1", "market_id": "m", "cosine_similarity": 0.7,
             "retrieved_member": False},
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))
    hypotheses_path = tmp_path / "hypotheses.json"
    hypotheses_path.write_text(json.dumps({"m": {"yes": "yes", "no": "no"}}))

    default_path = tmp_path / "default.jsonl"
    default_result = score_links(str(graph_path), str(hypotheses_path), str(default_path),
                                 FakeClassifier())
    default_rows = [json.loads(line) for line in default_path.read_text().splitlines()]
    assert default_result["candidate_links"] == 1
    assert [row["claim_id"] for row in default_rows] == ["c0"]

    all_path = tmp_path / "all.jsonl"
    classifier = RecordingClassifier()
    all_result = score_links(str(graph_path), str(hypotheses_path), str(all_path),
                             classifier, all_snapshots=True)
    all_rows = [json.loads(line) for line in all_path.read_text().splitlines()]
    assert all_result["candidate_links"] == 2
    assert [row["claim_id"] for row in all_rows] == ["c0", "c1"]
    later = all_rows[1]
    assert classifier.premises == ["first", "first", "new event headline",
                                   "new event headline"]
    assert later["headline"] == "new event headline"
    assert later["event_time"] == later["first_seen_at"] == "2026-01-01T11:00:00Z"
    assert [member["url"] for member in later["members"]] == ["u0", "u1"]
    assert all(member["provider_seen_at"] <= later["event_time"]
               for member in later["members"])


def test_headline_scorer_reuses_legacy_raw_probabilities_to_fresh_v2_output(tmp_path):
    import hashlib
    import sqlite3

    db_path = tmp_path / "news.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE news (source TEXT, source_id TEXT, market_slug TEXT, "
                       "url TEXT, title TEXT, provider_seen_at TEXT, published_at TEXT, "
                       "first_collected_at TEXT)")
    connection.execute("INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       ("gdelt-doc", "1", "m", "https://example.test/a", "headline",
                        "2026-01-01T00:00:00Z", None, "2026-01-01T00:00:00Z"))
    connection.commit()
    connection.close()
    hypotheses = {"m": {"yes": "yes hypothesis", "no": "no hypothesis"}}
    hypotheses_path = tmp_path / "hypotheses.json"
    hypotheses_path.write_text(json.dumps(hypotheses))
    row = {"source": "gdelt-doc", "source_id": "1", "market_slug": "m",
           "url": "https://example.test/a", "title": "headline",
           "provider_seen_at": "2026-01-01T00:00:00Z"}
    raw = {"yes": {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},
           "no": {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}}
    old_key = hashlib.sha256(json.dumps([
        "fake-nli", hypotheses["m"]["yes"], hypotheses["m"]["no"], row["source"],
        row["source_id"], row["market_slug"], row["url"], row["title"],
        row["provider_seen_at"]], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    prior_path = tmp_path / "prior.jsonl"
    prior_path.write_text(json.dumps({**row, "model_id": "fake-nli", "score_key": old_key,
                                      "signed_score": -0.9, "probabilities": raw}) + "\n")
    output_path = tmp_path / "v2.jsonl"
    classifier = FakeClassifier()
    result = score_headlines(str(db_path), str(hypotheses_path), str(output_path),
                             classifier, [str(prior_path), str(prior_path)])
    assert result["new_scores"] == 1
    assert classifier.calls == 0
    output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(output_rows) == 1
    assert output_rows[0]["scorer_version"] == "entailment_v2"
    assert output_rows[0]["signed_score"] == 0.8
