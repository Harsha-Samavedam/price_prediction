from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

from .core import PricePoint, iso, parse_time


@dataclass(frozen=True)
class ReactionRow:
    claim_id: str
    market_slug: str
    claim_text: str
    article_url: Optional[str]
    available_at: str
    horizon_minutes: int
    pre_price_time: str
    pre_price: float
    post_price_time: str
    post_price: float
    delta_yes: float

    def to_dict(self) -> dict:
        return asdict(self)


def replay_claims(claims: Iterable[dict], prices: list[PricePoint],
                  horizon_minutes: int, max_price_age_minutes: int) -> tuple[list[ReactionRow], dict[str, int]]:
    if horizon_minutes <= 0 or max_price_age_minutes < 0:
        raise ValueError("horizon must be positive and maximum price age nonnegative")
    # An aggregated price can incorporate trades throughout its bucket. Treat
    # it as available only when the entire bucket has ended. CSV/CLOB points
    # without a resolution are treated as timestamped observations.
    def price_available_at(point: PricePoint) -> datetime:
        return point.timestamp + timedelta(seconds=point.resolution_seconds or 0)

    # The Data API V2 emits a settlement point with resolution_seconds=0.
    eligible = sorted((p for p in prices if p.resolution_seconds != 0),
                      key=lambda p: (price_available_at(p), p.timestamp))
    times = [price_available_at(p) for p in eligible]
    results: list[ReactionRow] = []
    excluded = {"no_pre_price": 0, "stale_pre_price": 0,
                "no_post_price": 0, "stale_post_price": 0}
    max_age = timedelta(minutes=max_price_age_minutes)
    for claim in claims:
        available_at = parse_time(claim["available_at"])
        target = available_at + timedelta(minutes=horizon_minutes)
        pre_idx = bisect.bisect_right(times, available_at) - 1
        if pre_idx < 0:
            excluded["no_pre_price"] += 1
            continue
        pre = eligible[pre_idx]
        if available_at - times[pre_idx] > max_age:
            excluded["stale_pre_price"] += 1
            continue
        post_idx = bisect.bisect_right(times, target) - 1
        if post_idx <= pre_idx or times[post_idx] <= available_at:
            excluded["no_post_price"] += 1
            continue
        post = eligible[post_idx]
        if target - times[post_idx] > max_age:
            excluded["stale_post_price"] += 1
            continue
        results.append(ReactionRow(
            claim_id=claim["claim_id"], market_slug=claim["market_slug"],
            claim_text=claim["claim_text"], article_url=claim["article_url"],
            available_at=iso(available_at) or "", horizon_minutes=horizon_minutes,
            pre_price_time=iso(times[pre_idx]) or "", pre_price=pre.price,
            post_price_time=iso(times[post_idx]) or "", post_price=post.price,
            delta_yes=round(post.price - pre.price, 8),
        ))
    return results, excluded
