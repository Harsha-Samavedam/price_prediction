import json
import sqlite3
from datetime import datetime, timedelta, timezone

from claim_reaction.historical_prices import (
    PriceBin, classify_change, event_study, load_market_bins,
    split_contiguous_segments,
)

UTC = timezone.utc
T = datetime(2025, 1, 1, tzinfo=UTC)


def bin_(start_min, price, resolution=300):
    start = T + timedelta(minutes=start_min)
    return PriceBin("m", start, start + timedelta(seconds=resolution), price, resolution)


def test_load_uses_bucket_end_market_end_and_explicit_deadline():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE markets(slug, raw_json)")
    db.execute("CREATE TABLE prices(market_slug, timestamp, price, resolution_seconds, source)")
    db.execute("INSERT INTO markets VALUES (?, ?)", ("m", json.dumps({"endDate": (T + timedelta(minutes=8)).isoformat()})))
    db.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?)", [
        ("m", T.isoformat(), .4, 300, "h"),
        ("m", (T + timedelta(minutes=5)).isoformat(), .5, 300, "h"),
        ("m", (T + timedelta(minutes=9)).isoformat(), .6, 300, "h"),
        ("m", (T + timedelta(minutes=10)).isoformat(), 1, 0, "h"),
    ])
    bins = load_market_bins(db, "m")
    assert len(bins) == 1
    assert bins[0].bucket_end == T + timedelta(minutes=5)
    deadline_bins = load_market_bins(db, "m", T + timedelta(minutes=4))
    assert deadline_bins == []
    later_deadline_bins = load_market_bins(db, "m", T + timedelta(minutes=15))
    assert [item.bucket_start for item in later_deadline_bins] == [
        T, T + timedelta(minutes=5), T + timedelta(minutes=9)
    ]


def test_segments_and_threshold_classification():
    a, b, c = bin_(0, .4), bin_(5, .41), bin_(15, .38)
    assert [len(s) for s in split_contiguous_segments([a, b, c])] == [2, 1]
    assert classify_change(a, b, .01).direction == "up"
    assert classify_change(b, c, .02).direction == "down"
    assert classify_change(a, b, .02).direction == "none"


def test_event_uses_bucket_end_availability_and_caps_baseline_age_at_30_minutes():
    baseline, rise = bin_(0, .4), bin_(30, .415)
    claim = T + timedelta(minutes=30)
    outcome = event_study([baseline, rise], claim, threshold=.01,
                          horizon=timedelta(minutes=10), max_pre_event_age=timedelta(minutes=30))
    assert outcome.status == "crossed"
    assert outcome.interval_start == rise.bucket_start
    assert outcome.interval_end == rise.bucket_end
    assert outcome.direction == "up"
    stale = event_study([baseline], T + timedelta(minutes=36), threshold=.01,
                        horizon=timedelta(minutes=10), max_pre_event_age=timedelta(minutes=30))
    assert stale.reason == "stale_pre_event_baseline"
    assert stale.direction is None
    assert stale.baseline == baseline
    assert stale.crossing is None
    assert stale.interval_start is None
    assert stale.interval_end is None
    assert stale.delta is None


def test_deadline_overlap_and_no_crossing_are_censored():
    baseline = bin_(-5, .4)
    after_deadline = bin_(16, .42)
    late = event_study([baseline, after_deadline], T + timedelta(minutes=5), threshold=.01,
                       horizon=timedelta(minutes=10), max_pre_event_age=timedelta(minutes=30))
    assert late.reason == "no_bucket_within_horizon"
    overlapping = bin_(0, .42)
    overlap_outcome = event_study([baseline, overlapping], T + timedelta(minutes=2), threshold=.01,
                                  horizon=timedelta(minutes=10), max_pre_event_age=timedelta(minutes=30))
    assert overlap_outcome.reason == "no_post_claim_coverage"
    assert overlap_outcome.direction is None
    assert overlap_outcome.baseline == baseline
    assert overlap_outcome.crossing is None
    assert overlap_outcome.interval_start is None
    assert overlap_outcome.interval_end is None
    assert overlap_outcome.delta is None
    flat = bin_(5, .405)
    none = event_study([baseline, flat], T + timedelta(minutes=5), threshold=.01,
                       horizon=timedelta(minutes=5), max_pre_event_age=timedelta(minutes=30))
    assert none.reason == "no_threshold_crossing"
    assert none.direction is None
    assert none.baseline == baseline
    assert none.crossing is None
    assert none.interval_start is None
    assert none.interval_end is None
    assert none.delta is None


def test_no_crossing_with_short_followup_reports_right_censoring():
    baseline, flat = bin_(-5, .4), bin_(0, .401)
    outcome = event_study([baseline, flat], T, threshold=.01,
                          horizon=timedelta(hours=1), max_pre_event_age=timedelta(minutes=30))
    assert outcome.status == "censored"
    assert outcome.reason == "incomplete_post_claim_followup"


def test_gap_censors_before_later_crossing():
    baseline, first, later = bin_(0, .4), bin_(5, .401), bin_(15, .43)
    outcome = event_study([baseline, first, later], T + timedelta(minutes=5), threshold=.01,
                          horizon=timedelta(minutes=20), max_pre_event_age=timedelta(minutes=30))
    assert outcome.status == "censored"
    assert outcome.reason == "post_claim_coverage_gap"
    assert outcome.direction is None
    assert outcome.baseline == baseline
    assert outcome.crossing is None
    assert outcome.interval_start is None
    assert outcome.interval_end is None
    assert outcome.delta is None


def test_initial_gap_after_baseline_censors_first_candidate():
    baseline, next_bin = bin_(-5, .4), bin_(20, .43)
    outcome = event_study([baseline, next_bin], T, threshold=.01,
                          horizon=timedelta(minutes=30), max_pre_event_age=timedelta(minutes=30))
    assert outcome.status == "censored"
    assert outcome.reason == "post_claim_coverage_gap"
    assert outcome.direction is None
    assert outcome.baseline == baseline
    assert outcome.crossing is None
    assert outcome.interval_start is None
    assert outcome.interval_end is None
    assert outcome.delta is None


def test_missing_first_post_claim_bucket_is_censored():
    baseline, rise = bin_(-5, .4), bin_(5, .43)
    outcome = event_study([baseline, rise], T, threshold=.01,
                          horizon=timedelta(minutes=10), max_pre_event_age=timedelta(minutes=30))
    assert outcome.status == "censored"
    assert outcome.reason == "post_claim_coverage_gap"
