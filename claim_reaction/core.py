from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


UTC = timezone.utc


def parse_time(value: Any) -> datetime:
    """Parse an epoch timestamp or an ISO-8601 timestamp with a timezone."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid timestamp: {value!r}")
    value = value.strip()
    if value.isdigit():
        if len(value) == 14:  # GDELT YYYYMMDDHHMMSS
            return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        return datetime.fromtimestamp(int(value), UTC)
    value = value.replace("Z", "+00:00")
    # Gamma closedTime values sometimes end in an hour-only UTC offset.
    value = re.sub(r"([+-]\d{2})$", r"\1:00", value)
    # Python 3.9 accepts only three or six fractional-second digits here;
    # Gamma timestamps can have other lengths (for example, four digits).
    value = re.sub(r"\.(\d{1,5})(?=[+-]\d{2}:\d{2}$)",
                   lambda match: "." + match.group(1).ljust(6, "0"), value)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value!r}")
    return parsed.astimezone(UTC)


def maybe_time(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    return parse_time(value)


def iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z") if value else None


def encoded_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected array or JSON-encoded array, got {value!r}")


@dataclass(frozen=True)
class Market:
    slug: str
    market_id: str
    question: str
    description: str
    yes_asset_id: str
    version: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class PricePoint:
    market_slug: str
    timestamp: datetime
    price: float
    resolution_seconds: Optional[int]
    source: str

    def __post_init__(self) -> None:
        if not 0 <= self.price <= 1:
            raise ValueError(f"Price outside [0, 1]: {self.price}")


@dataclass(frozen=True)
class NewsItem:
    source: str
    source_id: str
    market_slug: str
    query: str
    url: str
    title: str
    published_at: Optional[datetime]
    provider_seen_at: Optional[datetime]
    collected_at: datetime
    raw: dict[str, Any]


def parse_polymarket_market(data: dict[str, Any]) -> Market:
    labels = encoded_array(data["outcomes"])
    version = str(data.get("version") or "v1")
    if version == "v2":
        ids = encoded_array(data.get("positionIds"))
    elif version == "v1":
        ids = encoded_array(data.get("clobTokenIds"))
    else:
        raise ValueError(f"Unsupported Polymarket version: {version}")
    if len(labels) != 2 or len(ids) != 2:
        raise ValueError("Only binary markets with two outcome asset IDs are supported")
    yes_indices = [i for i, label in enumerate(labels) if str(label).lower() == "yes"]
    if len(yes_indices) != 1:
        raise ValueError("Could not identify exactly one YES outcome")
    yes_id = str(ids[yes_indices[0]])
    if not yes_id:
        raise ValueError("YES outcome has no asset ID")
    return Market(
        slug=str(data["slug"]),
        market_id=str(data["id"]),
        question=str(data.get("question") or ""),
        description=str(data.get("description") or ""),
        yes_asset_id=yes_id,
        version=version,
        raw=data,
    )
