from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .core import Market, NewsItem, PricePoint, iso, parse_time


SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    slug TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    question TEXT NOT NULL,
    description TEXT NOT NULL,
    yes_asset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prices (
    market_slug TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    price REAL NOT NULL CHECK (price >= 0 AND price <= 1),
    resolution_seconds INTEGER,
    source TEXT NOT NULL,
    PRIMARY KEY (market_slug, timestamp, source)
);
CREATE INDEX IF NOT EXISTS idx_prices_time ON prices(market_slug, timestamp);
CREATE TABLE IF NOT EXISTS news (
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    query TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    published_at TEXT,
    provider_seen_at TEXT,
    first_collected_at TEXT NOT NULL,
    last_collected_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (source, source_id, market_slug)
);
CREATE INDEX IF NOT EXISTS idx_news_market_time ON news(market_slug, provider_seen_at);
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    market_slug TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    available_at TEXT NOT NULL,
    article_url TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_market_time ON claims(market_slug, available_at);
CREATE TABLE IF NOT EXISTS article_index (
    url TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    first_collected_at TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS article_fts USING fts5(
    url UNINDEXED, title, tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS article_embeddings (
    url TEXT NOT NULL,
    model_id TEXT NOT NULL,
    title TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    PRIMARY KEY (url, model_id)
);
CREATE TABLE IF NOT EXISTS article_bodies (
    url TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    body TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def put_market(self, market: Market) -> None:
        with self.db:
            self.db.execute(
                """INSERT OR REPLACE INTO markets VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (market.slug, market.market_id, market.question, market.description,
                 market.yes_asset_id, market.version, json.dumps(market.raw)),
            )

    def get_market(self, slug: str) -> Optional[Market]:
        row = self.db.execute("SELECT * FROM markets WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return None
        return Market(row["slug"], row["market_id"], row["question"], row["description"],
                      row["yes_asset_id"], row["version"], json.loads(row["raw_json"]))

    def put_prices(self, points: Iterable[PricePoint]) -> int:
        rows = [(p.market_slug, iso(p.timestamp), p.price, p.resolution_seconds, p.source)
                for p in points]
        with self.db:
            self.db.executemany(
                """INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?)""", rows
            )
        return len(rows)

    def put_news(self, items: Iterable[NewsItem]) -> int:
        rows = [
            (n.source, n.source_id, n.market_slug, n.query, n.url, n.title,
             iso(n.published_at), iso(n.provider_seen_at), iso(n.collected_at),
             iso(n.collected_at), json.dumps(n.raw))
            for n in items
        ]
        with self.db:
            self.db.executemany(
                """INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, source_id, market_slug) DO UPDATE SET
                    last_collected_at = excluded.last_collected_at,
                    provider_seen_at = COALESCE(news.provider_seen_at, excluded.provider_seen_at),
                    published_at = COALESCE(news.published_at, excluded.published_at),
                    title = CASE WHEN news.title = '' THEN excluded.title ELSE news.title END""",
                rows,
            )
        return len(rows)

    def put_claim(self, claim_id: str, market_slug: str, claim_text: str,
                  available_at: datetime, article_url: Optional[str]) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?)",
                (claim_id, market_slug, claim_text, iso(available_at), article_url),
            )

    def claims(self, market_slug: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM claims WHERE market_slug = ? ORDER BY available_at, claim_id",
            (market_slug,),
        ).fetchall()

    def prices(self, market_slug: str, source: str) -> list[PricePoint]:
        rows = self.db.execute(
            "SELECT * FROM prices WHERE market_slug = ? AND source = ? ORDER BY timestamp",
            (market_slug, source),
        ).fetchall()
        return [PricePoint(r["market_slug"], parse_time(r["timestamp"]), r["price"],
                           r["resolution_seconds"], r["source"]) for r in rows]

    def news_counts(self, market_slug: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT source, COUNT(*) AS count, MIN(published_at) AS earliest_publication, "
            "MIN(provider_seen_at) AS earliest_provider_seen FROM news "
            "WHERE market_slug = ? GROUP BY source ORDER BY source",
            (market_slug,),
        ).fetchall()
