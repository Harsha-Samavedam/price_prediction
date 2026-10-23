"""Retrospective event studies over bucketed Polymarket YES prices.

Stored timestamps are bucket starts. A bucket becomes observable only at its end.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

from .core import parse_time


@dataclass(frozen=True)
class PriceBin:
    market_slug: str
    bucket_start: datetime
    bucket_end: datetime
    price: float
    resolution_seconds: int


@dataclass(frozen=True)
class PriceChange:
    direction: str  # up, down, none
    delta: float


@dataclass(frozen=True)
class EventStudyOutcome:
    status: str  # crossed or censored
    reason: str
    direction: Optional[str]
    baseline: Optional[PriceBin]
    crossing: Optional[PriceBin]
    interval_start: Optional[datetime]
    interval_end: Optional[datetime]
    delta: Optional[float]


def load_market_bins(db: sqlite3.Connection, market_slug: str,
                     event_deadline: Optional[datetime] = None) -> list[PriceBin]:
    """Load ordered bins, dropping resolution rows and bins after endDate."""
    db.row_factory = sqlite3.Row
    market = db.execute("SELECT raw_json FROM markets WHERE slug = ?", (market_slug,)).fetchone()
    end_date = None
    if market:
        raw = json.loads(market["raw_json"])
        if raw.get("endDate"):
            raw_end_date = parse_time(raw["endDate"])
            end_date = raw_end_date
    if event_deadline is not None:
        end_date = event_deadline
    rows = db.execute(
        "SELECT timestamp, price, resolution_seconds FROM prices WHERE market_slug = ? ORDER BY timestamp",
        (market_slug,),
    ).fetchall()
    bins = []
    for row in rows:
        resolution = row["resolution_seconds"]
        if resolution is None or int(resolution) <= 0:
            continue
        start = parse_time(row["timestamp"])
        end = start + timedelta(seconds=int(resolution))
        if end_date is not None and end > end_date:
            continue
        bins.append(PriceBin(market_slug, start, end, float(row["price"]), int(resolution)))
    return bins


def split_contiguous_segments(bins: Iterable[PriceBin]) -> list[list[PriceBin]]:
    ordered = sorted(bins, key=lambda b: b.bucket_start)
    segments: list[list[PriceBin]] = []
    for item in ordered:
        if not segments or segments[-1][-1].bucket_end != item.bucket_start:
            segments.append([item])
        else:
            segments[-1].append(item)
    return segments


def classify_change(previous: PriceBin, current: PriceBin, threshold: float) -> PriceChange:
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    delta = current.price - previous.price
    direction = "up" if delta > 0 and delta + 1e-12 >= threshold else (
        "down" if delta < 0 and -delta + 1e-12 >= threshold else "none")
    return PriceChange(direction, delta)


def event_study(bins: Iterable[PriceBin], claim_time: datetime, *, threshold: float,
                horizon: timedelta, max_pre_event_age: timedelta) -> EventStudyOutcome:
    """Find first observable post-claim crossing; never infer an exact event time."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if horizon < timedelta(0) or max_pre_event_age < timedelta(0):
        raise ValueError("horizon and max_pre_event_age must be nonnegative")
    ordered = sorted(bins, key=lambda b: b.bucket_start)
    baseline_candidates = [b for b in ordered if b.bucket_end <= claim_time]
    if not baseline_candidates:
        return EventStudyOutcome(status="censored", reason="no_pre_event_baseline",
                                 direction=None, baseline=None, crossing=None,
                                 interval_start=None, interval_end=None, delta=None)
    baseline = baseline_candidates[-1]
    if claim_time - baseline.bucket_end > max_pre_event_age:
        return EventStudyOutcome(status="censored", reason="stale_pre_event_baseline",
                                 direction=None, baseline=baseline, crossing=None,
                                 interval_start=None, interval_end=None, delta=None)
    deadline = claim_time + horizon
    candidates = [b for b in ordered if b.bucket_start >= claim_time and b.bucket_end <= deadline]
    if not candidates:
        reason = "no_post_claim_coverage" if not any(b.bucket_start >= claim_time for b in ordered) else "no_bucket_within_horizon"
        return EventStudyOutcome(status="censored", reason=reason, direction=None,
                                 baseline=baseline, crossing=None, interval_start=None,
                                 interval_end=None, delta=None)
    first_gap = candidates[0].bucket_start - max(claim_time, baseline.bucket_end)
    if first_gap >= timedelta(seconds=candidates[0].resolution_seconds):
        return EventStudyOutcome(status="censored", reason="post_claim_coverage_gap",
                                 direction=None, baseline=baseline, crossing=None,
                                 interval_start=None, interval_end=None, delta=None)
    previous_post_claim: Optional[PriceBin] = None
    for bucket in candidates:
        if previous_post_claim is not None and previous_post_claim.bucket_end != bucket.bucket_start:
            return EventStudyOutcome(status="censored", reason="post_claim_coverage_gap",
                                     direction=None, baseline=baseline, crossing=None,
                                     interval_start=None, interval_end=None, delta=None)
        # Measure the cumulative move from the pre-event quote.
        change = classify_change(baseline, bucket, threshold)
        if change.direction != "none":
            return EventStudyOutcome("crossed", "threshold_crossing", change.direction, baseline,
                                     bucket, bucket.bucket_start, bucket.bucket_end, change.delta)
        previous_post_claim = bucket
    # A missing full bucket before the requested horizon makes this a
    # right-censored follow-up, even if the observed bins were flat.
    reason = ("incomplete_post_claim_followup"
              if deadline - candidates[-1].bucket_end >= timedelta(seconds=candidates[-1].resolution_seconds)
              else "no_threshold_crossing")
    return EventStudyOutcome(status="censored", reason=reason,
                             direction=None, baseline=baseline, crossing=None,
                             interval_start=None, interval_end=None, delta=None)
