"""Live, append-only observations of RSS items and Polymarket market events."""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import math
import sqlite3
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urljoin, urlparse

from .core import NewsItem, iso, maybe_time
from .sources import Polymarket
from .store import Store

UTC = timezone.utc
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DEFAULT_FEEDS = (
    "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "https://www.democrats.senate.gov/feed/",
)

OBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS observation_feeds(
 feed_url TEXT PRIMARY KEY, etag TEXT, last_modified TEXT, last_success_at TEXT,
 last_poll_at TEXT, last_status TEXT NOT NULL DEFAULT 'never', last_error TEXT
);
CREATE TABLE IF NOT EXISTS observation_feed_polls(
 poll_id INTEGER PRIMARY KEY, feed_url TEXT NOT NULL, started_at TEXT NOT NULL,
 completed_at TEXT NOT NULL, status TEXT NOT NULL, http_status INTEGER,
 error TEXT, raw_sha256 TEXT, raw_payload BLOB, run_id INTEGER
);
CREATE TABLE IF NOT EXISTS observation_feed_items(
 item_key TEXT PRIMARY KEY, feed_url TEXT NOT NULL, source_id TEXT NOT NULL,
 url TEXT NOT NULL, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
 first_poll_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS observation_feed_versions(
 version_id INTEGER PRIMARY KEY, item_key TEXT NOT NULL, feed_url TEXT NOT NULL,
 observed_at TEXT NOT NULL, published_at TEXT, updated_at TEXT,
 title TEXT NOT NULL, summary TEXT NOT NULL, item_url TEXT NOT NULL, raw_json TEXT NOT NULL,
 bootstrap INTEGER NOT NULL DEFAULT 0, interval_start TEXT,
 analysis_ready_at TEXT, run_id INTEGER,
 UNIQUE(item_key, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_observation_versions_item
 ON observation_feed_versions(item_key, version_id);
CREATE TABLE IF NOT EXISTS observation_body_snapshots(
 snapshot_id INTEGER PRIMARY KEY, feed_version_id INTEGER NOT NULL UNIQUE,
 item_url TEXT NOT NULL, fetch_started_at TEXT, fetch_completed_at TEXT,
 status TEXT NOT NULL, body TEXT, body_sha256 TEXT, error TEXT,
 bootstrap INTEGER NOT NULL, run_id INTEGER
);
CREATE TABLE IF NOT EXISTS observation_market_sessions(
 session_id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
 status TEXT NOT NULL, error TEXT, run_id INTEGER
);
CREATE TABLE IF NOT EXISTS observation_market_frames(
 frame_id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL,
 received_at TEXT NOT NULL, monotonic_received REAL NOT NULL,
 raw_payload TEXT NOT NULL, parse_error TEXT
);
CREATE TABLE IF NOT EXISTS observation_market_events(
 event_id INTEGER PRIMARY KEY, frame_id INTEGER NOT NULL, subindex INTEGER NOT NULL,
 event_type TEXT, asset_id TEXT, market_slug TEXT, exchange_time TEXT,
 bid REAL, ask REAL, midpoint REAL, last_trade_price REAL,
 trade_size REAL, trade_side TEXT, tick_size TEXT, error TEXT,
 normalized_json TEXT NOT NULL, received_at TEXT NOT NULL,
 UNIQUE(frame_id, subindex)
);
CREATE TABLE IF NOT EXISTS observation_runs(
 run_id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
 status TEXT NOT NULL, duration_seconds REAL NOT NULL,
 market_slugs_json TEXT NOT NULL, feed_urls_json TEXT NOT NULL, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_observation_market_events_time
 ON observation_market_events(exchange_time);
"""


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: Any) -> Optional[datetime]:
    """Polymarket wire timestamps are epoch milliseconds, including strings."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
            return datetime.fromtimestamp(float(value) / 1000.0, UTC)
        return maybe_time(str(value))
    except (ValueError, OverflowError, OSError, TypeError):
        return None


def _number(value: Any, low: float = 0.0, high: float = 1.0) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and low <= result <= high else None
    except (TypeError, ValueError):
        return None


def _wire_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return None


def _book_bbo(event: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    bids, asks = event.get("bids"), event.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list):
        return None, None

    def levels(side: list[Any]) -> list[float]:
        result = []
        for row in side:
            if not isinstance(row, dict):
                continue
            price, size = _number(row.get("price")), _number(row.get("size"), 0, float("inf"))
            if price is not None and size is not None and size > 0:
                result.append(price)
        return result

    bid_prices, ask_prices = levels(bids), levels(asks)
    return (max(bid_prices) if bid_prices else None,
            min(ask_prices) if ask_prices else None)


def normalize_market_message(message: Any, asset_to_slug: Optional[dict[str, str]] = None) -> list[dict[str, Any]]:
    """Normalize a wire frame to traceable per-event rows without carrying state."""
    events = message if isinstance(message, list) else [message]
    output: list[dict[str, Any]] = []
    for subindex, event in enumerate(events):
        row: dict[str, Any] = {
            "subindex": len(output), "event_type": None, "asset_id": None,
            "market_slug": None, "exchange_time": None, "bid": None, "ask": None,
            "midpoint": None, "last_trade_price": None, "trade_size": None,
            "trade_side": None, "tick_size": None, "error": None,
            "normalized": event if isinstance(event, dict) else {"value": event},
        }
        if not isinstance(event, dict):
            row["error"] = "event is not a JSON object"
            output.append(row)
            continue
        try:
            kind = event.get("event_type") or event.get("type")
            row["event_type"] = str(kind) if kind is not None else None
            raw_asset = event.get("asset_id") if event.get("asset_id") is not None else event.get("assetId")
            row["asset_id"] = _wire_text(raw_asset)
            if raw_asset is not None and row["asset_id"] is None:
                row["error"] = "asset_id is not scalar text"
            row["market_slug"] = _wire_text(event.get("slug"))
            if row["asset_id"] is not None and asset_to_slug:
                row["market_slug"] = asset_to_slug.get(str(row["asset_id"]), row["market_slug"])
            raw_timestamp = event.get("timestamp") if "timestamp" in event else event.get("time")
            row["exchange_time"] = iso(_timestamp(raw_timestamp))
            if raw_timestamp not in (None, "") and row["exchange_time"] is None:
                row["error"] = "invalid event timestamp"
            if kind == "book":
                row["bid"], row["ask"] = _book_bbo(event)
            elif kind == "price_change":
                changes = event.get("price_changes")
                if isinstance(changes, list):
                    for j, change in enumerate(changes):
                        if not isinstance(change, dict):
                            output.append({**row, "subindex": len(output), "error": "price_change entry is not an object", "normalized": {"event": event, "price_change": change}})
                            continue
                        derived = dict(row)
                        derived["subindex"] = len(output)
                        raw_asset = change.get("asset_id") if change.get("asset_id") is not None else row["asset_id"]
                        derived["asset_id"] = _wire_text(raw_asset)
                        if raw_asset is not None and derived["asset_id"] is None:
                            derived["error"] = "asset_id is not scalar text"
                        if derived["asset_id"] is not None and asset_to_slug:
                            derived["market_slug"] = asset_to_slug.get(str(derived["asset_id"]), row["market_slug"])
                        derived["bid"] = _number(change.get("best_bid"))
                        derived["ask"] = _number(change.get("best_ask"))
                        b, a = derived["bid"], derived["ask"]
                        derived["midpoint"] = (b + a) / 2 if b is not None and a is not None and b <= a else None
                        derived["normalized"] = {"event": event, "price_change": change}
                        output.append(derived)
                    if not changes:
                        row["error"] = "price_change has no price_changes entries"
                    elif all(not isinstance(change, dict) for change in changes):
                        continue
                    elif any(isinstance(change, dict) for change in changes):
                        continue
                    else:
                        row["error"] = "price_change has no valid price_changes entries"
                else:
                    row["error"] = "price_change has malformed price_changes"
            elif kind == "best_bid_ask":
                row["bid"] = _number(event.get("best_bid"))
                row["ask"] = _number(event.get("best_ask"))
            elif kind == "last_trade_price":
                row["last_trade_price"] = _number(event.get("price"))
                if event.get("price") is not None and row["last_trade_price"] is None:
                    row["error"] = "invalid trade price"
                try:
                    if isinstance(event.get("size"), bool):
                        raise ValueError("boolean trade size")
                    size = float(event["size"]) if event.get("size") is not None else None
                    row["trade_size"] = size if size is not None and math.isfinite(size) and size >= 0 else None
                except (TypeError, ValueError):
                    row["trade_size"] = None
                    if event.get("size") is not None and row["error"] is None:
                        row["error"] = "invalid trade size"
                row["trade_side"] = _wire_text(event.get("side"))
            elif kind == "tick_size_change":
                row["tick_size"] = _wire_text(event.get("new_tick_size") or event.get("tick_size"))
            elif kind in ("market_resolved", "new_market"):
                pass
            elif kind is None:
                row["error"] = "event has no event_type"
            else:
                row["error"] = f"unknown event_type: {kind}"
            b, a = row["bid"], row["ask"]
            if b is not None and a is not None and b <= a:
                row["midpoint"] = (b + a) / 2
        except Exception as exc:  # a malformed event must not kill the feed
            row["error"] = f"normalization error: {type(exc).__name__}: {exc}"
        row["subindex"] = len(output)
        output.append(row)
    return output


class ObservationStore(Store):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(OBS_SCHEMA)
        self.run_id: Optional[int] = None
        # Additive migrations for databases created by an earlier observer build.
        for table, column, declaration in (
            ("observation_feed_versions", "item_url", "TEXT NOT NULL DEFAULT ''"),
            ("observation_feed_versions", "analysis_ready_at", "TEXT"),
            ("observation_feed_versions", "run_id", "INTEGER"),
            ("observation_market_sessions", "run_id", "INTEGER"),
            ("observation_feed_polls", "run_id", "INTEGER"),
            ("observation_market_events", "received_at", "TEXT NOT NULL DEFAULT ''"),
        ):
            columns = {row["name"] for row in self.db.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def record_frame(self, session_id: int, received: datetime, mono: float,
                     raw: str, parse_error: Optional[str] = None) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO observation_market_frames(session_id,received_at,monotonic_received,raw_payload,parse_error) VALUES(?,?,?,?,?)",
                (session_id, iso(received), mono, raw, parse_error))
        return int(cur.lastrowid)

    def record_events(self, frame_id: int, rows: list[dict[str, Any]]) -> None:
        received = self.db.execute("SELECT received_at FROM observation_market_frames WHERE frame_id=?", (frame_id,)).fetchone()[0]
        with self.db:
            self.db.executemany(
                """INSERT INTO observation_market_events(frame_id,subindex,event_type,asset_id,market_slug,
                exchange_time,bid,ask,midpoint,last_trade_price,trade_size,trade_side,tick_size,error,normalized_json,received_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(frame_id, r["subindex"], r["event_type"], r["asset_id"], r["market_slug"],
                  r["exchange_time"], r["bid"], r["ask"], r["midpoint"], r["last_trade_price"],
                  r["trade_size"], r["trade_side"], r["tick_size"], r["error"],
                  json.dumps(r["normalized"], ensure_ascii=False, default=str), received) for r in rows])


def _feed_request(url: str, etag: Optional[str], modified: Optional[str]) -> tuple[int, bytes, dict[str, str], datetime]:
    headers = {"User-Agent": "claim-reaction-observer/0.1", "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"}
    if etag: headers["If-None-Match"] = etag
    if modified: headers["If-Modified-Since"] = modified
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("feed response exceeds 4 MB")
            return response.status, data, dict(response.headers.items()), _now()
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return 304, b"", dict(exc.headers.items()), _now()
        raise RuntimeError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(node: ET.Element, names: set[str]) -> Optional[str]:
    for child in node:
        if _local_name(child.tag) in names and child.text:
            value = child.text.strip()
            if value:
                return value
    return None


def _feed_date(*values: Optional[str]) -> tuple[Optional[datetime], Optional[str]]:
    for value in values:
        if not value:
            continue
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                return None, f"timezone missing from feed date: {value}"
            return dt.astimezone(UTC), None
        except (TypeError, ValueError, OverflowError):
            try:
                dt = maybe_time(value)
                if dt is not None:
                    return dt, None
            except ValueError:
                pass
            return None, f"invalid feed date: {value}"
    return None, None


def parse_feed(payload: bytes, feed_url: str) -> tuple[list[dict[str, Any]], list[str]]:
    root = ET.fromstring(payload)
    items, errors = [], []
    kind = _local_name(root.tag)
    if kind not in {"feed", "rss", "rdf"}:
        raise ValueError(f"unsupported feed root element: {kind}")
    nodes = list(root.iter()) if kind == "feed" else [n for n in root.iter() if _local_name(n.tag) == "item"]
    if kind == "feed":
        nodes = [n for n in nodes if _local_name(n.tag) == "entry"]
    seen_ids: set[str] = set()
    for node in nodes:
        title = _child_text(node, {"title"}) or ""
        summary = _child_text(node, {"description", "summary", "content", "encoded"}) or ""
        link = None
        for child in node:
            if _local_name(child.tag) == "link":
                link = child.attrib.get("href") or (child.text.strip() if child.text else None)
                if link and child.attrib.get("rel", "alternate") == "alternate":
                    break
        link = urljoin(feed_url, link) if link else ""
        identity = _child_text(node, {"guid", "id"}) or link
        if not identity:
            errors.append("feed item has neither guid/id nor link")
            continue
        if identity in seen_ids:
            errors.append(f"duplicate item identity in poll: {identity}")
            continue
        seen_ids.add(identity)
        pub_value = _child_text(node, {"pubdate", "published", "date"})
        published, pub_error = _feed_date(pub_value)
        if pub_value is None:
            pub_error = "feed item has no publication date"
        updated, upd_error = _feed_date(_child_text(node, {"updated", "modified"}))
        if pub_error: errors.append(pub_error)
        if upd_error: errors.append(upd_error)
        items.append({"source_id": identity, "url": link, "title": title, "summary": summary,
                      "published_at": iso(published), "updated_at": iso(updated),
                      "raw": ET.tostring(node, encoding="unicode")})
    return items, errors


async def _poll_feed(store: ObservationStore, url: str,
                     fetcher: Callable[..., tuple[int, bytes, dict[str, str]]],
                     market_slugs: tuple[str, ...],
                     on_new_version: Optional[Callable[[int, str, int], None]] = None) -> dict[str, Any]:
    started, received = _now(), _now()
    feed = store.db.execute("SELECT * FROM observation_feeds WHERE feed_url=?", (url,)).fetchone()
    if feed is None:
        with store.db:
            store.db.execute("INSERT INTO observation_feeds(feed_url,last_poll_at,last_status) VALUES(?,?,'polling')", (url, iso(started)))
        feed = store.db.execute("SELECT * FROM observation_feeds WHERE feed_url=?", (url,)).fetchone()
    poll_id = None
    version_count = 0
    try:
        try:
            response = await asyncio.to_thread(fetcher, url, feed["etag"], feed["last_modified"])
        except asyncio.CancelledError:
            cancelled_at = _now()
            with store.db:
                store.db.execute("INSERT INTO observation_feed_polls(feed_url,started_at,completed_at,status,error,run_id) VALUES(?,?,?,?,?,?)",
                                 (url, iso(started), iso(cancelled_at), "cancelled", "observation duration ended", store.run_id))
                store.db.execute("UPDATE observation_feeds SET last_poll_at=?,last_status='cancelled',last_error='observation duration ended' WHERE feed_url=?", (iso(cancelled_at), url))
            raise
        if len(response) == 4:
            status, payload, headers, received = response
        else:
            status, payload, headers = response
            received = _now()
        parsed_ok = True
        try:
            parsed, parse_errors = (parse_feed(payload, url) if status != 304 else ([], []))
        except Exception as exc:
            parsed_ok = False
            parsed = []
            parse_errors = [f"feed parse error: {type(exc).__name__}: {exc}"]
        digest = hashlib.sha256(payload).hexdigest() if payload else None
        poll_status = "partial" if parse_errors else "ok"
        normalized_headers = {str(key).lower(): value for key, value in headers.items()}
        with store.db:
            cur = store.db.execute("INSERT INTO observation_feed_polls(feed_url,started_at,completed_at,status,http_status,error,raw_sha256,raw_payload,run_id) VALUES(?,?,?,?,?,?,?,?,?)",
                                   (url, iso(started), iso(received), poll_status, status, "; ".join(parse_errors) or None, digest, payload or None, store.run_id))
            poll_id = int(cur.lastrowid)
            store.db.execute("UPDATE observation_feeds SET etag=?,last_modified=?,last_success_at=?,last_poll_at=?,last_status=?,last_error=? WHERE feed_url=?",
                             (normalized_headers.get("etag", feed["etag"]) if parsed_ok else feed["etag"],
                              normalized_headers.get("last-modified", feed["last_modified"]) if parsed_ok else feed["last_modified"],
                              iso(received) if parsed_ok else feed["last_success_at"], iso(received),
                              poll_status, "; ".join(parse_errors) or None, url))
            for entry in parsed:
                key = url + "\0" + entry["source_id"]
                old = store.db.execute("SELECT * FROM observation_feed_items WHERE item_key=?", (key,)).fetchone()
                if old:
                    store.db.execute("UPDATE observation_feed_items SET last_seen_at=? WHERE item_key=?", (iso(received), key))
                    prev = store.db.execute("SELECT title,summary,published_at,updated_at FROM observation_feed_versions WHERE item_key=? ORDER BY version_id DESC LIMIT 1", (key,)).fetchone()
                    previous = store.db.execute("SELECT title,summary,published_at,updated_at,item_url FROM observation_feed_versions WHERE item_key=? ORDER BY version_id DESC LIMIT 1", (key,)).fetchone()
                    changed = previous is None or tuple(previous) != (entry["title"], entry["summary"], entry["published_at"], entry["updated_at"], entry["url"])
                else:
                    store.db.execute("INSERT INTO observation_feed_items VALUES(?,?,?,?,?,?,?)",
                                     (key, url, entry["source_id"], entry["url"], iso(received), iso(received), poll_id))
                    changed = True
                if changed:
                    last_success = feed["last_success_at"]
                    bootstrap = int(last_success is None)
                    curv = store.db.execute("INSERT INTO observation_feed_versions(item_key,feed_url,observed_at,published_at,updated_at,title,summary,item_url,raw_json,bootstrap,interval_start,analysis_ready_at,run_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                            (key, url, iso(received), entry["published_at"], entry["updated_at"], entry["title"], entry["summary"], entry["url"], json.dumps(entry, ensure_ascii=False), bootstrap, last_success, None, store.run_id))
                    if on_new_version and entry["url"]:
                        parsed_url = urlparse(entry["url"])
                        if parsed_url.scheme in ("http", "https") and parsed_url.netloc:
                            on_new_version(int(curv.lastrowid), entry["url"], bootstrap)
                    version_count += 1
                    # Mirror every version to subscribed markets. Provider time is receipt time.
                    news_id = url + "#" + entry["source_id"] + "#v" + str(curv.lastrowid)
                    news = NewsItem("rss-live", news_id, "", "", entry["url"], entry["title"],
                                    maybe_time(entry["published_at"]), received, received, {**entry, "version_id": curv.lastrowid, "feed_url": url})
                    for slug in market_slugs:
                        store.db.execute("""INSERT INTO news(source,source_id,market_slug,query,url,title,published_at,provider_seen_at,first_collected_at,last_collected_at,raw_json)
                            VALUES('rss-live',?,?,?, ?,?,?,?,?,?,?) ON CONFLICT(source,source_id,market_slug) DO UPDATE SET
                            title=excluded.title,published_at=excluded.published_at,provider_seen_at=excluded.provider_seen_at,
                            last_collected_at=excluded.last_collected_at,raw_json=excluded.raw_json""",
                        (news_id, slug, "live RSS", entry["url"], entry["title"], entry["published_at"], iso(received), iso(received), iso(received), json.dumps(news.raw, ensure_ascii=False)))
            # A newly added market subscription receives the current observed
            # version even when its feed item has not changed since last poll.
            current = store.db.execute("SELECT v.*,i.source_id,i.first_seen_at FROM observation_feed_versions v JOIN observation_feed_items i USING(item_key) WHERE v.feed_url=? AND v.version_id=(SELECT MAX(v2.version_id) FROM observation_feed_versions v2 WHERE v2.item_key=v.item_key)", (url,)).fetchall()
            for row in current:
                for slug in market_slugs:
                    raw = json.loads(row["raw_json"])
                    raw.update({"version_id": row["version_id"], "feed_url": url, "first_seen_at": row["first_seen_at"]})
                    store.db.execute("""INSERT INTO news(source,source_id,market_slug,query,url,title,published_at,provider_seen_at,first_collected_at,last_collected_at,raw_json)
                        VALUES('rss-live',?,?,?, ?,?,?,?,?,?,?) ON CONFLICT(source,source_id,market_slug) DO UPDATE SET
                        last_collected_at=excluded.last_collected_at""",
                        (url + "#" + row["source_id"] + "#v" + str(row["version_id"]), slug, "live RSS", row["item_url"], row["title"], row["published_at"], row["observed_at"], row["observed_at"], row["observed_at"], json.dumps(raw, ensure_ascii=False)))
        return {"feed_url": url, "status": poll_status, "http_status": status, "items": len(parsed), "versions": version_count, "errors": parse_errors}
    except Exception as exc:
        received = _now()
        error = f"{type(exc).__name__}: {exc}"
        with store.db:
            cur = store.db.execute("INSERT INTO observation_feed_polls(feed_url,started_at,completed_at,status,error,run_id) VALUES(?,?,?,?,?,?)",
                                   (url, iso(started), iso(received), "error", error, store.run_id))
            poll_id = int(cur.lastrowid)
            store.db.execute("UPDATE observation_feeds SET last_poll_at=?,last_status='error',last_error=? WHERE feed_url=?", (iso(received), error, url))
        return {"feed_url": url, "status": "error", "error": error, "items": 0}


async def _websockets_connect(url: str, **kwargs: Any) -> Any:
    import websockets
    return await websockets.connect(url, **kwargs)


async def _market_loop(store: ObservationStore, slugs: tuple[str, ...], assets: dict[str, str],
                       duration_deadline: float, connect: Callable[..., Any],
                       stop_event: asyncio.Event) -> None:
    attempt = 0
    while not stop_event.is_set() and time.monotonic() < duration_deadline:
        started = _now()
        with store.db:
            sid = int(store.db.execute("INSERT INTO observation_market_sessions(started_at,status,run_id) VALUES(?, 'connecting',?)", (iso(started), store.run_id)).lastrowid)
        status, err = "closed", None
        try:
            remaining = max(0.001, duration_deadline - time.monotonic())
            connection = await asyncio.wait_for(
                connect(WS_URL, ping_interval=None, close_timeout=3, max_size=4 * 1024 * 1024),
                timeout=min(30, remaining))
            async with connection as ws:
                with store.db:
                    store.db.execute("UPDATE observation_market_sessions SET status='connected' WHERE session_id=?", (sid,))
                await ws.send(json.dumps({"assets_ids": list(assets.values()), "type": "market", "custom_feature_enabled": True}))
                attempt = 0
                async def pinger() -> None:
                    while not stop_event.is_set() and time.monotonic() < duration_deadline:
                        await asyncio.sleep(10)
                        if not stop_event.is_set():
                            if time.monotonic() - last_pong[0] > 30:
                                await ws.close()
                                return
                            await ws.send("PING")
                last_pong = [time.monotonic()]
                ping_task = asyncio.create_task(pinger())
                try:
                    while not stop_event.is_set() and time.monotonic() < duration_deadline:
                        try:
                            remaining = duration_deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            raw = await asyncio.wait_for(ws.recv(), timeout=min(30, remaining))
                        except asyncio.TimeoutError:
                            if time.monotonic() >= duration_deadline:
                                break
                            raise TimeoutError("websocket heartbeat deadline exceeded")
                        received, mono = _now(), time.monotonic()
                        if raw == "PONG":
                            last_pong[0] = time.monotonic()
                            continue
                        try:
                            message = json.loads(raw)
                            parse_error = None
                            rows = normalize_market_message(message, {asset: slug for slug, asset in assets.items()})
                        except Exception as exc:
                            message, rows = None, []
                            parse_error = f"{type(exc).__name__}: {exc}"
                        frame_id = store.record_frame(sid, received, mono, raw if isinstance(raw, str) else bytes(raw).decode("utf8", "replace"), parse_error)
                        if rows: store.record_events(frame_id, rows)
                finally:
                    ping_task.cancel()
                    await asyncio.gather(ping_task, return_exceptions=True)
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status, err = "disconnected", f"{type(exc).__name__}: {exc}"
        finally:
            with store.db:
                store.db.execute("UPDATE observation_market_sessions SET ended_at=?,status=?,error=? WHERE session_id=?",
                                 (iso(_now()), status, err, sid))
        if not stop_event.is_set() and time.monotonic() < duration_deadline:
            attempt += 1
            await asyncio.sleep(min(30, 2 ** min(attempt - 1, 5), max(0, duration_deadline - time.monotonic())))


async def run_observation(db_path: str, market_slugs: list[str] | tuple[str, ...],
                          feed_urls: list[str] | tuple[str, ...], duration_seconds: float,
                          poll_interval_seconds: float = 60, *,
                          polymarket: Optional[Polymarket] = None,
                          websocket_connect: Optional[Callable[..., Any]] = None,
                          feed_fetcher: Optional[Callable[..., tuple[int, bytes, dict[str, str]]]] = None,
                          snapshot_bodies: bool = False,
                          body_fetcher: Optional[Callable[[str], str]] = None) -> dict[str, Any]:
    """Observe markets and feeds for a polling period; started body fetches are drained."""
    if (not math.isfinite(float(duration_seconds)) or not math.isfinite(float(poll_interval_seconds))
            or duration_seconds <= 0 or poll_interval_seconds <= 0):
        raise ValueError("duration and poll interval must be finite and positive")
    for url in feed_urls:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"feed URL must use http or https: {url}")
    if market_slugs and websocket_connect is None:
        try:
            import websockets  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Live market observation needs websockets 15; install with `pip install -r requirements-observation.txt`") from exc
    store = ObservationStore(db_path)
    client = polymarket or Polymarket()
    connect = websocket_connect or _websockets_connect
    fetcher = feed_fetcher or _feed_request
    if snapshot_bodies and body_fetcher is None:
        from .retrieval import fetch_article_body
        body_fetcher = fetch_article_body
    slugs, feeds = tuple(dict.fromkeys(market_slugs)), tuple(dict.fromkeys(feed_urls))
    errors: list[str] = []
    assets: dict[str, str] = {}
    run_started = _now()
    run_started_mono = time.monotonic()
    with store.db:
        store.run_id = int(store.db.execute(
            "INSERT INTO observation_runs(started_at,status,duration_seconds,market_slugs_json,feed_urls_json) VALUES(?, 'running',?,?,?)",
            (iso(run_started), duration_seconds, json.dumps(slugs), json.dumps(feeds))).lastrowid)
    run_status = "error"
    body_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    body_queued_ids: set[int] = set()
    body_workers: list[asyncio.Task[Any]] = []
    body_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    def schedule_body(version_id: int, url: str, bootstrap: int) -> None:
        if version_id in body_queued_ids:
            return
        with store.db:
            store.db.execute("INSERT OR IGNORE INTO observation_body_snapshots(feed_version_id,item_url,fetch_started_at,status,bootstrap,run_id) VALUES(?,?,NULL,'queued',?,?)",
                             (version_id, url, bootstrap, store.run_id))
        body_queued_ids.add(version_id)
        body_queue.put_nowait((version_id, url))
    def persist_fetch(version_id: int, url: str) -> None:
        body = None
        error = None
        try:
            body = body_fetcher(url)  # type: ignore[misc]
            status = "ok"
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        except Exception as exc:
            status, digest = "error", None
            error = f"{type(exc).__name__}: {exc}"
        db = sqlite3.connect(db_path, timeout=5)
        try:
            with db:
                db.execute("UPDATE observation_body_snapshots SET fetch_completed_at=?,status=?,body=?,body_sha256=?,error=? WHERE feed_version_id=?",
                           (iso(_now()), status, body, digest, error, version_id))
        finally:
            db.close()
    async def body_worker() -> None:
        while True:
            version_id, url = await body_queue.get()
            if stop.is_set():
                with store.db:
                    store.db.execute("UPDATE observation_body_snapshots SET status='deferred',error='observer polling period ended before fetch started' WHERE feed_version_id=? AND status='queued'", (version_id,))
                body_queue.task_done()
                continue
            started = _now()
            with store.db:
                store.db.execute("UPDATE observation_body_snapshots SET fetch_started_at=?,status='running' WHERE feed_version_id=?",
                                 (iso(started), version_id))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(body_executor, persist_fetch, version_id, url)
            body_queue.task_done()
    try:
        for slug in slugs:
            market = await asyncio.to_thread(client.market_by_slug, slug)
            raw = market.raw
            for key in ("closed", "archived"):
                if raw.get(key) is True: raise ValueError(f"market {slug} is {key}")
            for key in ("active", "acceptingOrders", "enableOrderBook"):
                if raw.get(key) is False: raise ValueError(f"market {slug} is inactive or has no order book ({key}=false)")
            store.put_market(market)
            assets[slug] = market.yes_asset_id

        deadline = time.monotonic() + duration_seconds
        if snapshot_bodies:
            body_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="body-snapshot")
            body_workers = [asyncio.create_task(body_worker()) for _ in range(8)]
            deferred = (store.db.execute(
                "SELECT b.feed_version_id,b.item_url FROM observation_body_snapshots b "
                "JOIN observation_feed_versions v ON v.version_id=b.feed_version_id "
                f"WHERE b.status='deferred' AND v.feed_url IN ({','.join('?' for _ in feeds)}) "
                "ORDER BY b.snapshot_id", feeds).fetchall() if feeds else [])
            for row in deferred:
                with store.db:
                    store.db.execute("UPDATE observation_body_snapshots SET status='queued',run_id=?,fetch_started_at=NULL,fetch_completed_at=NULL,body=NULL,body_sha256=NULL,error=NULL WHERE feed_version_id=? AND status='deferred'",
                                     (store.run_id, row["feed_version_id"]))
                if store.db.execute("SELECT status FROM observation_body_snapshots WHERE feed_version_id=?", (row["feed_version_id"],)).fetchone()[0] == "queued":
                    body_queued_ids.add(row["feed_version_id"])
                    body_queue.put_nowait((row["feed_version_id"], row["item_url"]))
        stop = asyncio.Event()
        async def feed_runner(url: str) -> None:
            while not stop.is_set() and time.monotonic() < deadline:
                result = await _poll_feed(store, url, fetcher, slugs,
                                          schedule_body if snapshot_bodies else None)
                if result.get("status") != "ok": errors.append(f"{url}: {result.get('error')}")
                errors.extend(f"{url}: {e}" for e in result.get("errors", []))
                remaining = deadline - time.monotonic()
                if remaining <= 0: break
                try: await asyncio.wait_for(stop.wait(), timeout=min(poll_interval_seconds, remaining))
                except asyncio.TimeoutError: pass
        tasks = [asyncio.create_task(feed_runner(url)) for url in feeds]
        if assets: tasks.append(asyncio.create_task(_market_loop(store, slugs, assets, deadline, connect, stop)))
        try:
            if tasks:
                await asyncio.sleep(max(0, duration_seconds))
            stop.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            if snapshot_bodies:
                await body_queue.join()
                for worker in body_workers: worker.cancel()
                await asyncio.gather(*body_workers, return_exceptions=True)
                # Duration bounds collection; already-started article fetches are
                # drained here and each guarded fetch has its own 15-second timeout.
                body_executor.shutdown(wait=True, cancel_futures=False)
                body_workers.clear()
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                    errors.append(f"observation task failed: {type(outcome).__name__}: {outcome}")
        except asyncio.CancelledError:
            stop.set()
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if snapshot_bodies:
                for worker in body_workers: worker.cancel()
                await asyncio.gather(*body_workers, return_exceptions=True)
                with store.db:
                    store.db.execute("UPDATE observation_body_snapshots SET status='deferred',error='observation cancelled before fetch started' WHERE run_id=? AND status='queued'", (store.run_id,))
                    store.db.execute("UPDATE observation_body_snapshots SET error='observation cancelled while fetch was running; completion is pending' WHERE run_id=? AND status='running'", (store.run_id,))
                if body_executor is not None:
                    # Wait for running workers so asyncio shutdown cannot orphan them.
                    body_executor.shutdown(wait=True, cancel_futures=False)
            raise
        disconnected = store.db.execute("SELECT COUNT(*) FROM observation_market_sessions WHERE status='disconnected' AND run_id=?", (store.run_id,)).fetchone()[0]
        malformed_events = store.db.execute("SELECT COUNT(*) FROM observation_market_events e JOIN observation_market_frames f USING(frame_id) JOIN observation_market_sessions s USING(session_id) WHERE e.error IS NOT NULL AND s.run_id=?", (store.run_id,)).fetchone()[0]
        malformed_frames = store.db.execute("SELECT COUNT(*) FROM observation_market_frames f JOIN observation_market_sessions s USING(session_id) WHERE f.parse_error IS NOT NULL AND s.run_id=?", (store.run_id,)).fetchone()[0]
        if disconnected:
            errors.append(f"market websocket disconnected {disconnected} time(s)")
        if malformed_events:
            errors.append(f"{malformed_events} malformed or unknown market event(s)")
        if malformed_frames:
            errors.append(f"{malformed_frames} malformed websocket frame(s)")
        counts = {
            "feed_polls": store.db.execute("SELECT COUNT(*) FROM observation_feed_polls WHERE run_id=?", (store.run_id,)).fetchone()[0],
            "feed_versions": store.db.execute("SELECT COUNT(*) FROM observation_feed_versions WHERE run_id=?", (store.run_id,)).fetchone()[0],
            "market_sessions": store.db.execute("SELECT COUNT(*) FROM observation_market_sessions WHERE run_id=?", (store.run_id,)).fetchone()[0],
            "market_frames": store.db.execute("SELECT COUNT(*) FROM observation_market_frames f JOIN observation_market_sessions s USING(session_id) WHERE s.run_id=?", (store.run_id,)).fetchone()[0],
            "market_events": store.db.execute("SELECT COUNT(*) FROM observation_market_events e JOIN observation_market_frames f USING(frame_id) JOIN observation_market_sessions s USING(session_id) WHERE s.run_id=?", (store.run_id,)).fetchone()[0],
        }
        snapshot_counts = {row["status"]: row["n"] for row in store.db.execute(
            "SELECT status,COUNT(*) AS n FROM observation_body_snapshots WHERE run_id=? GROUP BY status",
            (store.run_id,)).fetchall()}
        elapsed_seconds = time.monotonic() - run_started_mono
        run_status = "ok" if not errors else "partial"
        with store.db:
            store.db.execute("UPDATE observation_runs SET ended_at=?,status=?,error=? WHERE run_id=?",
                             (iso(_now()), run_status, "; ".join(errors) or None, store.run_id))
        return {"duration_seconds": duration_seconds, "elapsed_seconds": elapsed_seconds,
                "body_snapshots": {"completed": snapshot_counts.get("ok", 0),
                                   "failed": snapshot_counts.get("error", 0),
                                   "deferred": snapshot_counts.get("deferred", 0),
                                   "running": snapshot_counts.get("running", 0)},
                "markets": list(assets), "feeds": list(feeds),
                "run_id": store.run_id, "counts": counts, "errors": errors, "status": run_status}
    except asyncio.CancelledError:
        run_status = "cancelled"
        raise
    except Exception as exc:
        with store.db:
            store.db.execute("UPDATE observation_runs SET error=? WHERE run_id=?",
                             (f"{type(exc).__name__}: {exc}", store.run_id))
        raise
    finally:
        with store.db:
            store.db.execute("UPDATE observation_runs SET ended_at=COALESCE(ended_at,?),status=CASE WHEN status='running' THEN ? ELSE status END WHERE run_id=?",
                             (iso(_now()), run_status, store.run_id))
        store.close()


def observation_status(db_path: str) -> dict[str, Any]:
    """Read existing observation state without creating a database or using network."""
    import os
    if not os.path.exists(db_path):
        return {"exists": False, "feeds": [], "sessions": [], "frames": 0, "events": 0}
    db_uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(db_uri, uri=True)
    db.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "observation_feeds" not in tables:
            return {"exists": True, "feeds": [], "sessions": [], "frames": 0, "events": 0}
        feeds = [dict(r) for r in db.execute("SELECT * FROM observation_feeds ORDER BY feed_url")]
        sessions = [dict(r) for r in db.execute("SELECT * FROM observation_market_sessions ORDER BY session_id DESC LIMIT 20")]
        runs = ([dict(r) for r in db.execute("SELECT * FROM observation_runs ORDER BY run_id DESC LIMIT 20")]
                if "observation_runs" in tables else [])
        return {"exists": True, "feeds": feeds, "sessions": sessions,
                "runs": runs,
                "frames": db.execute("SELECT COUNT(*) FROM observation_market_frames").fetchone()[0],
                "events": db.execute("SELECT COUNT(*) FROM observation_market_events").fetchone()[0],
                "feed_polls": db.execute("SELECT COUNT(*) FROM observation_feed_polls").fetchone()[0],
                "feed_versions": db.execute("SELECT COUNT(*) FROM observation_feed_versions").fetchone()[0]}
    finally:
        db.close()
