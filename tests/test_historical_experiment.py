import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from types import SimpleNamespace

from claim_reaction.hawkes import history_features
import claim_reaction.historical_experiment as historical_experiment
from claim_reaction.historical_experiment import run_historical_experiment

UTC = timezone.utc
T = datetime(2025, 1, 1, tzinfo=UTC)


def _fixture(tmp_path):
    db_path = tmp_path / "prices.sqlite"
    db = sqlite3.connect(db_path)
    db.executescript("CREATE TABLE markets(slug TEXT, raw_json TEXT);"
                     "CREATE TABLE prices(market_slug TEXT, timestamp TEXT, price REAL, resolution_seconds INTEGER);")
    markets = {"a": "ga", "b": "gb", "c": "gc"}
    for slug in markets:
        db.execute("INSERT INTO markets VALUES (?, ?)", (slug, json.dumps({})))
        # Both resolutions occur in each market, proving that runs remain
        # independently indexed by resolution.
        for resolution in (300, 1800):
            n = 8
            for i in range(n):
                start = T + timedelta(seconds=resolution * i)
                price = .4 + (.02 if i in (3, 6) else .001 * i)
                db.execute("INSERT INTO prices VALUES (?, ?, ?, ?)",
                           (slug, start.isoformat(), price, resolution))
    db.commit()
    db.close()
    graph = {"claims": [{"claim_id": "claim-a", "earliest_seen_at": (T + timedelta(minutes=10)).isoformat(),
                         "event_time": (T + timedelta(minutes=10)).isoformat(), "is_first_report": True}],
             "market_edges": [], "link_candidates": [{"claim_id": "claim-a", "market_id": "a"}]}
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(json.dumps({"claim_id": "claim-a", "market_slug": "a",
                                       "first_seen_at": (T + timedelta(minutes=10)).isoformat(),
                                       "signed_score": -.7}) + "\n")
    topics_path = tmp_path / "topics.json"
    topics_path.write_text(json.dumps({"groups": {g: {"markets": [m]} for m, g in markets.items()}}))
    return db_path, graph_path, scores_path, topics_path


def test_history_features_do_not_read_current_bucket_label_or_same_time_news():
    starts = np.array([0., 10., 20.])
    labels = np.array([0, 1, 2])
    x, names = history_features(starts, np.array([10.]), np.array([1.]), labels,
                                10, 30, "signed_news")
    assert x[1, names.index("news_up")] == 0  # event at feature time is unavailable
    assert x[1, names.index("price_up")] == 0  # label for bucket starting at 10 is future
    assert x[2, names.index("price_up")] > 0  # prior bucket ended at 20


def test_topic_group_holdout_and_resolutions_are_separate(tmp_path):
    args = _fixture(tmp_path)
    result = run_historical_experiment(*args, thresholds=[.01], decay_seconds=[600],
                                       resolutions=[300, 1800])
    folds = result["folds"]
    assert {f["resolution_seconds"] for f in folds} == {300, 1800}
    assert {f["held_out_topic_group"] for f in folds} == {"ga", "gb", "gc"}
    for fold in folds:
        assert fold["held_out_bin_count"] == 7
        # Each market has its own group, so the test market never enters train.
        assert fold["train_bin_count"] == 14
    assert all(row["resolution_seconds"] in (300, 1800)
               for row in result["aggregate_oof_metrics"])
    assert result["event_study"]["price_window_coverage"]["1800"]["before_first_price"] == 1
    assert not [row for row in result["event_study"]["outcomes"]
                if row["resolution_seconds"] == 1800]
    assert all(f["decay_seconds"] == 600 for f in folds)
    assert all([s["decay_seconds"] for s in f["inner_decay_scores"]] == [600]
               for f in folds)


def test_inner_decay_selection_uses_inner_losses_and_smaller_decay_on_tie(tmp_path, monkeypatch):
    monkeypatch.setattr(historical_experiment, "fit_from_features", lambda *args, **kwargs: object())
    monkeypatch.setattr(historical_experiment, "score_hawkes",
                        lambda *args, **kwargs: SimpleNamespace(log_loss=1.0, brier=.5))
    result = run_historical_experiment(*_fixture(tmp_path), thresholds=[.01],
                                       decay_seconds=[3600, 900], resolutions=[300])
    assert result["folds"]
    for fold in result["folds"]:
        assert fold["decay_seconds"] == 900
        assert [s["decay_seconds"] for s in fold["inner_decay_scores"]] == [3600, 900]
        assert all(s["inner_jump_log_loss"] == 1.0 for s in fold["inner_decay_scores"])


def test_missing_link_score_coverage_fails_loudly(tmp_path):
    db, graph, scores, topics = _fixture(tmp_path)
    scores.write_text("")
    with pytest.raises(ValueError, match="missing link-score coverage"):
        run_historical_experiment(db, graph, scores, topics, thresholds=[.01],
                                  decay_seconds=[60], resolutions=[300])


def test_evidence_updates_are_chronological_and_deduplicated(tmp_path):
    claims = [
        {"claim_id": "r3", "cluster_id": "cluster", "event_time": (T + timedelta(minutes=4)).isoformat(), "is_first_report": False},
        {"claim_id": "r1", "cluster_id": "cluster", "event_time": T.isoformat(), "is_first_report": True},
        {"claim_id": "r5", "cluster_id": "cluster", "event_time": (T + timedelta(minutes=6)).isoformat(), "is_first_report": False},
        {"claim_id": "r2", "cluster_id": "cluster", "event_time": (T + timedelta(minutes=2)).isoformat(), "is_first_report": False},
        {"claim_id": "r4", "cluster_id": "cluster", "event_time": (T + timedelta(minutes=5)).isoformat(), "is_first_report": False},
    ]
    graph = {"claims": claims, "link_candidates": [
        {"claim_id": row["claim_id"], "market_id": "a"} for row in claims]}
    graph_path = tmp_path / "updates-graph.json"
    graph_path.write_text(json.dumps(graph))
    marks = {"r1": 0, "r2": .7, "r3": .6, "r4": -.8, "r5": -.5}
    scores_path = tmp_path / "updates-scores.jsonl"
    scores_path.write_text("".join(json.dumps({"claim_id": claim_id, "market_slug": "a",
        "first_seen_at": (T + timedelta(minutes=i)).isoformat(), "signed_score": mark}) + "\n"
        for i, (claim_id, mark) in enumerate(marks.items())))

    events, rows, counts = historical_experiment._load_graph_events(
        graph_path, scores_path, {}, evidence_updates=True)

    assert [row["claim_id"] for row in rows] == ["r1", "r2", "r4"]
    assert [row["event_kind"] for row in rows] == ["first_report", "positive_update", "negative_update"]
    assert [mark for _, mark in events["a"]] == [0, .7, -.8]
    assert counts["repeated_reports_suppressed"] == 2


def test_evidence_update_mode_requires_scores_for_every_candidate(tmp_path):
    graph_path = tmp_path / "updates-graph.json"
    graph_path.write_text(json.dumps({"claims": [
        {"claim_id": "first", "cluster_id": "c", "event_time": T.isoformat(), "is_first_report": True},
        {"claim_id": "later", "cluster_id": "c", "event_time": (T + timedelta(minutes=1)).isoformat(), "is_first_report": False},
    ], "link_candidates": [{"claim_id": "first", "market_id": "a"},
                           {"claim_id": "later", "market_id": "a"}]}))
    scores_path = tmp_path / "updates-scores.jsonl"
    scores_path.write_text(json.dumps({"claim_id": "first", "market_slug": "a",
        "first_seen_at": T.isoformat(), "signed_score": 0}) + "\n")
    with pytest.raises(ValueError, match="missing link-score coverage"):
        historical_experiment._load_graph_events(graph_path, scores_path, {}, evidence_updates=True)


def test_evidence_updates_emit_market_first_seen_after_cluster_report(tmp_path):
    claims = [
        {"claim_id": "first", "cluster_id": "c", "event_time": T.isoformat(), "is_first_report": True},
        {"claim_id": "later-yes", "cluster_id": "c", "event_time": (T + timedelta(minutes=1)).isoformat(), "is_first_report": False},
        {"claim_id": "later-yes-repeat", "cluster_id": "c", "event_time": (T + timedelta(minutes=2)).isoformat(), "is_first_report": False},
        {"claim_id": "later-no", "cluster_id": "c", "event_time": (T + timedelta(minutes=3)).isoformat(), "is_first_report": False},
    ]
    graph_path = tmp_path / "late-market-graph.json"
    graph_path.write_text(json.dumps({"claims": claims, "link_candidates": [
        {"claim_id": row["claim_id"], "market_id": "a"}
        for row in claims if row["claim_id"] != "first"
    ]}))
    scores_path = tmp_path / "late-market-scores.jsonl"
    scores_path.write_text("".join(json.dumps({"claim_id": claim_id, "market_slug": "a",
        "first_seen_at": (T + timedelta(minutes=i)).isoformat(), "signed_score": mark}) + "\n"
        for i, (claim_id, mark) in enumerate((("later-yes", .7),
                                               ("later-yes-repeat", .4),
                                               ("later-no", -.6)))))

    events, rows, counts = historical_experiment._load_graph_events(
        graph_path, scores_path, {}, evidence_updates=True)

    assert [row["claim_id"] for row in rows] == ["later-yes", "later-no"]
    assert [row["event_kind"] for row in rows] == ["positive_update", "negative_update"]
    assert [row["time"] for row in rows] == [T + timedelta(minutes=1), T + timedelta(minutes=3)]
    assert [mark for _, mark in events["a"]] == [.7, -.6]
    assert counts["repeated_reports_suppressed"] == 1


def test_event_study_reports_censoring_and_deadline_exclusions(tmp_path):
    db, graph, scores, topics = _fixture(tmp_path)
    deadline = tmp_path / "deadlines.json"
    deadline.write_text(json.dumps({"a": (T + timedelta(minutes=20)).isoformat()}))
    result = run_historical_experiment(db, graph, scores, topics, thresholds=[.5],
                                       decay_seconds=[600], deadline_overrides_path=deadline,
                                       resolutions=[300])
    assert result["deadline_exclusions"]["a"]["deadline_excluded_price_bins"] > 0
    assert result["deadline_exclusions"]["a"]["removed_price_bin_count"] >= 0
    assert result["event_study"]["censoring_counts"]
    assert {row["horizon"] for row in result["event_study"]["outcomes"]} == {"5m", "1h", "1d"}
    first = result["event_study"]["outcomes"][0]
    assert first["claim_time"]
    assert "baseline_price" in first and "crossing_delta" in first
    assert "latency_lower_bound_seconds" in first and "latency_upper_bound_seconds" in first
    assert result["event_study"]["summary"]


def test_evidence_event_summary_reports_signed_alignment_and_censoring(tmp_path):
    db, graph, scores, topics = _fixture(tmp_path)
    # A positive mark defines upward crossings as aligned for this summary.
    scores.write_text(json.dumps({"claim_id": "claim-a", "market_slug": "a",
        "first_seen_at": (T + timedelta(minutes=10)).isoformat(), "signed_score": .7}) + "\n")
    result = run_historical_experiment(db, graph, scores, topics, thresholds=[.01],
                                       decay_seconds=[600], resolutions=[300],
                                       evidence_updates=True)
    summaries = result["event_study"]["summary"]
    five_minute = next(row for row in summaries if row["horizon"] == "5m")
    assert five_minute["aligned_denominator"] == 1
    assert five_minute["opposite_denominator"] == 1
    assert five_minute["aligned_crossed"] + five_minute["opposite_crossed"] <= 1
    assert five_minute["no_threshold_crossing_censored"] in (0, 1)
    assert (five_minute["median_aligned_crossing_upper_bound_lag_seconds"] is None
            or five_minute["aligned_crossed"] > 0)
    assert (five_minute["median_opposite_crossing_upper_bound_lag_seconds"] is None
            or five_minute["opposite_crossed"] > 0)


def test_later_market_end_override_reports_added_bins_without_negative_exclusions(tmp_path):
    db_path, graph, scores, topics = _fixture(tmp_path)
    db = sqlite3.connect(db_path)
    raw_end = (T + timedelta(minutes=20)).isoformat()
    db.execute("UPDATE markets SET raw_json = ? WHERE slug = 'a'", (json.dumps({"endDate": raw_end}),))
    db.commit()
    db.close()
    later = tmp_path / "later.json"
    later.write_text(json.dumps({"a": (T + timedelta(hours=1)).isoformat()}))
    result = run_historical_experiment(db_path, graph, scores, topics, thresholds=[.5],
                                       decay_seconds=[600], deadline_overrides_path=later,
                                       resolutions=[300])
    report = result["deadline_exclusions"]["a"]
    assert report["override_timestamp"] == (T + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    assert report["added_price_bin_count"] > 0
    assert report["removed_price_bin_count"] == 0
    assert report["deadline_excluded_price_bins"] == 0
