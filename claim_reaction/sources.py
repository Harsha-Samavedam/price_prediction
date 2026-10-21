from __future__ import annotations

import csv
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Optional

from .core import Market, NewsItem, PricePoint, maybe_time, parse_polymarket_market, parse_time


UTC = timezone.utc


def get_bytes(url: str, params: Optional[dict[str, Any]] = None,
              headers: Optional[dict[str, str]] = None) -> bytes:
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={
        "User-Agent": "claim-reaction-research/0.1",
        **(headers or {}),
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(300).decode("utf-8", "replace").strip()
        raise RuntimeError(f"HTTP {exc.code} from {urllib.parse.urlparse(url).netloc}: {detail}") from exc


def get_json(url: str, params: Optional[dict[str, Any]] = None,
             headers: Optional[dict[str, str]] = None) -> Any:
    return json.loads(get_bytes(url, params, headers))


def require_key(name: str) -> str:
    key = os.getenv(name)
    if not key:
        raise ValueError(f"Set {name} to use this source")
    return key


def item(source: str, source_id: str, market_slug: str, query: str,
         url: str, title: str, published_at: Optional[datetime],
         provider_seen_at: Optional[datetime], raw: dict[str, Any]) -> NewsItem:
    return NewsItem(source, source_id, market_slug, query, url, title,
                    published_at, provider_seen_at, datetime.now(UTC), raw)


def gdelt_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if len(value) == 16 and value[8] == "T" and value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return parse_time(value)


class Polymarket:
    GAMMA = "https://gamma-api.polymarket.com"
    DATA = "https://data-api.polymarket.com"
    CLOB = "https://clob.polymarket.com"

    def market_by_slug(self, slug: str) -> Market:
        data = get_json(self.GAMMA + "/markets/slug/" + urllib.parse.quote(slug, safe=""))
        return parse_polymarket_market(data)

    def prices(self, market: Market, start: datetime, end: datetime,
               api: str = "data-v2") -> list[PricePoint]:
        if end <= start:
            raise ValueError("end must be after start")
        if api == "clob":
            response = get_json(self.CLOB + "/prices-history", {
                "market": market.yes_asset_id,
                "startTs": int(start.timestamp()),
                "endTs": int(end.timestamp()),
                "fidelity": 1,
            })
            return [PricePoint(market.slug, parse_time(p["t"]), float(p["p"]),
                               None, "polymarket-clob") for p in response.get("history", [])]
        if api != "data-v2":
            raise ValueError(f"Unknown price API: {api}")
        # Data API V2 limits explicit windows to 15 days. Splitting also makes
        # retention gaps visible instead of silently substituting another grain.
        points: dict[int, PricePoint] = {}
        cursor_start = start
        while cursor_start < end:
            cursor_end = min(cursor_start + timedelta(days=15), end)
            params: dict[str, Any] = {
                "token_id": market.yes_asset_id,
                "start": int(cursor_start.timestamp()),
                "end": int(cursor_end.timestamp()),
                "limit": 2500,
            }
            while True:
                response = get_json(self.DATA + "/v2/prices-history", params)
                for row in response.get("data", []):
                    resolution = int(row["resolution_seconds"])
                    if resolution == 0:  # Settlement point is not a market reaction.
                        continue
                    timestamp = int(row["timestamp"])
                    points[timestamp] = PricePoint(
                        market.slug, parse_time(timestamp), float(row["price"]),
                        resolution, "polymarket-data-v2",
                    )
                page = response.get("pagination") or {}
                if not page.get("has_more"):
                    break
                cursor = page.get("next_cursor")
                if not cursor:
                    raise RuntimeError("Price history says more pages exist without a cursor")
                params["cursor"] = cursor
            cursor_start = cursor_end
        return [points[t] for t in sorted(points)]


def gdelt_doc(query: str, start: datetime, end: datetime,
              market_slug: str) -> Iterable[NewsItem]:
    if end <= start:
        raise ValueError("end must be after start")

    def fetch_window(window_start: datetime, window_end: datetime) -> Iterable[dict[str, Any]]:
        response = get_json("https://api.gdeltproject.org/api/v2/doc/doc", {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "startdatetime": window_start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": window_end.strftime("%Y%m%d%H%M%S"),
            "maxrecords": 250,
        })
        articles = response.get("articles", [])
        if len(articles) < 250:
            yield from articles
            return
        if window_end - window_start <= timedelta(minutes=15):
            raise RuntimeError("GDELT DOC still hits 250 records in a 15-minute window")
        midpoint = window_start + (window_end - window_start) / 2
        # Overlap one second to avoid losing a boundary article if either end
        # is exclusive. The outer iterator deduplicates URLs.
        yield from fetch_window(window_start, midpoint + timedelta(seconds=1))
        yield from fetch_window(midpoint - timedelta(seconds=1), window_end)

    seen_urls: set[str] = set()
    cursor = start
    while cursor < end:
        # ArticleList searches only the last three months of each requested
        # window. Two-month chunks stay below that limit in every calendar.
        window_end = min(cursor + timedelta(days=60), end)
        # The API uses strict bounds; a one-second lead includes records at
        # the requested start and at boundaries between adjacent chunks.
        window_start = cursor - timedelta(seconds=1)
        for row in fetch_window(window_start, window_end):
            url = row.get("url")
            if not url or url in seen_urls:
                continue
            seen_at = gdelt_time(row.get("seendate"))
            if seen_at is not None and not start <= seen_at < end:
                continue
            seen_urls.add(url)
            yield item("gdelt-doc", url, market_slug, query, url,
                       row.get("title") or "", None, seen_at, row)
        cursor = window_end


def gdelt_gkg_archive(query: str, start: datetime, end: datetime,
                      market_slug: str) -> Iterable[NewsItem]:
    # GKG gives URLs and entity/theme metadata, not article headlines or full text.
    # This is a coarse metadata filter, not claim extraction.
    csv.field_size_limit(10 * 1024 * 1024)
    tick = start.replace(minute=(start.minute // 15) * 15, second=0, microsecond=0)
    needle = query.casefold()
    available_batches = 0
    while tick < end:
        stamp = tick.strftime("%Y%m%d%H%M%S")
        url = f"https://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
        try:
            raw_zip = get_bytes(url)
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                tick += timedelta(minutes=15)
                continue
            raise
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
            available_batches += 1
            with archive.open(archive.namelist()[0]) as raw_file:
                reader = csv.reader(io.TextIOWrapper(raw_file, encoding="utf-8"), delimiter="\t")
                for row in reader:
                    if len(row) < 27:
                        continue
                    article_url = row[4]
                    search_fields = " ".join((article_url, row[7], row[9], row[11], row[13]))
                    if needle not in search_fields.casefold():
                        continue
                    seen = parse_time(row[1])
                    if not start <= seen < end:
                        continue
                    raw = {"gkg_record_id": row[0], "batch": stamp,
                           "source": row[3], "themes": row[7],
                           "persons": row[11], "organizations": row[13],
                           "match_basis": "metadata_substring"}
                    yield item("gdelt-gkg", article_url, market_slug, query,
                               article_url, "", None, seen, raw)
        tick += timedelta(minutes=15)
    if available_batches == 0:
        raise RuntimeError("No GDELT GKG archive batches exist in the requested interval")


def guardian(query: str, start: datetime, end: datetime,
             market_slug: str) -> Iterable[NewsItem]:
    key = require_key("GUARDIAN_API_KEY")
    page = 1
    while True:
        response = get_json("https://content.guardianapis.com/search", {
            "q": query,
            "from-date": start.date().isoformat(),
            "to-date": end.date().isoformat(),
            "page-size": 200,
            "page": page,
            "api-key": key,
        })["response"]
        for row in response.get("results", []):
            published = maybe_time(row.get("webPublicationDate"))
            if published is None or not start <= published < end:
                continue
            yield item("guardian", row["id"], market_slug, query,
                       row["webUrl"], row.get("webTitle") or "", published, None, row)
        if page >= response.get("pages", page):
            break
        page += 1


def nyt(query: str, start: datetime, end: datetime,
        market_slug: str) -> Iterable[NewsItem]:
    key = require_key("NYT_API_KEY")
    page = 0
    while True:
        response = get_json("https://api.nytimes.com/svc/search/v2/articlesearch.json", {
            "q": query,
            "begin_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "page": page,
            "api-key": key,
        })["response"]
        docs = response.get("docs", [])
        for row in docs:
            published = maybe_time(row.get("pub_date"))
            if published is None or not start <= published < end:
                continue
            url = row.get("web_url") or ""
            if not url:
                continue
            yield item("nyt", row.get("_id") or url, market_slug, query,
                       url, (row.get("headline") or {}).get("main") or "",
                       published, None, row)
        if not docs or (page + 1) * 10 >= response.get("meta", {}).get("hits", 0):
            break
        page += 1
        if page >= 100:
            raise RuntimeError("NYT Article Search reached its page limit; narrow the date window")


def newsapi(query: str, start: datetime, end: datetime,
            market_slug: str) -> Iterable[NewsItem]:
    key = require_key("NEWSAPI_KEY")
    page = 1
    while True:
        response = get_json("https://newsapi.org/v2/everything", {
            "q": query,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "sortBy": "publishedAt",
            "pageSize": 100,
            "page": page,
        }, {"X-Api-Key": key})
        articles = response.get("articles", [])
        for row in articles:
            url = row.get("url")
            if not url:
                continue
            published = maybe_time(row.get("publishedAt"))
            if published is None or not start <= published < end:
                continue
            yield item("newsapi", url, market_slug, query, url,
                       row.get("title") or "", published, None, row)
        if page * 100 >= response.get("totalResults", 0):
            break
        page += 1
        if page > 100:
            raise RuntimeError("NewsAPI reached its page limit; narrow the date window")


def mediacloud(query: str, start: datetime, end: datetime,
               market_slug: str, collection_id: int) -> Iterable[NewsItem]:
    key = require_key("MEDIACLOUD_API_KEY")
    cursor: Optional[str] = None
    while True:
        params: dict[str, Any] = {
            "q": query,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "platform": "onlinenews-mediacloud",
            "cs": collection_id,
            "page_size": 1000,
        }
        if cursor:
            params["pagination_token"] = cursor
        response = get_json("https://search.mediacloud.org/api/search/story-list", params,
                            {"Authorization": f"Token {key}"})
        for row in response.get("stories", []):
            url = row.get("url")
            if not url:
                continue
            indexed = maybe_time(row.get("indexed_date"))
            # Media Cloud often provides only a publication date, not a time.
            published = row.get("publish_date")
            if published and len(published) > 10:
                published_at = maybe_time(published)
            else:
                published_at = None
            yield item("mediacloud", str(row.get("id") or url), market_slug, query,
                       url, row.get("title") or "", published_at, indexed, row)
        next_cursor = response.get("pagination_token")
        if not next_cursor:
            break
        if next_cursor == cursor:
            raise RuntimeError("Media Cloud returned the same pagination token twice")
        cursor = next_cursor


def rss(feed_url: str, market_slug: str, query: str) -> Iterable[NewsItem]:
    root = ET.fromstring(get_bytes(feed_url))
    for entry in root.findall(".//item") + root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title = entry.findtext("title") or entry.findtext("{http://www.w3.org/2005/Atom}title") or ""
        url = entry.findtext("link") or ""
        if not url:
            link = entry.find("{http://www.w3.org/2005/Atom}link")
            url = link.get("href", "") if link is not None else ""
        if not url or query.casefold() not in title.casefold():
            continue
        published_raw = (entry.findtext("pubDate") or
                         entry.findtext("{http://www.w3.org/2005/Atom}published") or
                         entry.findtext("{http://www.w3.org/2005/Atom}updated"))
        published = None
        if published_raw:
            try:
                published = parsedate_to_datetime(published_raw).astimezone(UTC)
            except (TypeError, ValueError):
                published = maybe_time(published_raw)
        yield item("rss", url, market_slug, query, url, title,
                   published, None, {"feed_url": feed_url})
