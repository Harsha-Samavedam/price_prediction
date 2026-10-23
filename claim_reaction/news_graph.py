"""Text-only graph from multilingual headlines to claims and market questions.

The module consumes already collected GDELT records and a caller supplied
multilingual embedder. It does not inspect outcomes, prices, or market data
beyond market identifiers and question text.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], role: str) -> ArrayLike: ...


@dataclass(frozen=True)
class Article:
    url: str
    title: str
    provider_seen_at: str
    market_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClaimNode:
    claim_id: str
    cluster_id: str
    is_first_report: bool
    utc_day: str
    representative_title: str
    event_title: str
    event_article_url: str
    earliest_seen_at: str
    event_time: str
    members: tuple[Article, ...]


@dataclass(frozen=True)
class MarketEdge:
    claim_id: str
    market_id: str
    cosine_similarity: float


@dataclass(frozen=True)
class LinkCandidate:
    claim_id: str
    market_id: str
    cosine_similarity: float
    retrieved_member: bool


@dataclass(frozen=True)
class NewsGraph:
    claims: tuple[ClaimNode, ...]
    market_edges: tuple[MarketEdge, ...]
    link_candidates: tuple[LinkCandidate, ...]
    article_provenance: tuple[tuple[str, tuple[Article, ...]], ...] = ()


def _time(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("provider_seen_at must be an ISO timestamp or datetime")
    if dt.tzinfo is None:
        raise ValueError("provider_seen_at must include a timezone")
    return dt.astimezone(timezone.utc)


def _market_ids(record: Mapping[str, Any]) -> set[str]:
    values = record.get("market_ids", record.get("market_slugs", record.get("market_slug", ())))
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values if value is not None and str(value)}


def _normalize_rows(vectors: ArrayLike, expected: int) -> NDArray[np.float64]:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != expected or arr.shape[1] == 0:
        raise ValueError("embedder must return one vector per input text")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embedding vectors must be finite")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must be nonzero")
    return arr / norms[:, None]


def build_news_graph(records: Sequence[Mapping[str, Any]],
                     markets: Sequence[Mapping[str, Any]], embedder: Embedder,
                     cluster_cosine_threshold: float, top_k: int) -> NewsGraph:
    """Build deterministic online same-day headline clusters and market links.

    Records require ``url``, ``title``, and ``provider_seen_at``; retrieved
    associations may be supplied as ``market_ids``, ``market_slugs``, or
    ``market_slug``. Each new unique article emits a snapshot at its provider
    timestamp. Later headlines and duplicate-URL associations cannot change
    that snapshot; duplicate records remain available in article_provenance.
    Markets require ``market_id`` (or ``market_slug``) and ``question``.
    """
    if not -1 <= cluster_cosine_threshold <= 1:
        raise ValueError("cluster cosine threshold must be in [-1, 1]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    raw_by_url: dict[str, list[tuple[datetime, str, tuple[str, ...]]]] = {}
    for row in records:
        url, title = str(row.get("url", "")), str(row.get("title", ""))
        if not url or not title:
            raise ValueError("each record requires a nonempty url and title")
        seen = _time(row["provider_seen_at"])
        raw_by_url.setdefault(url, []).append((seen, title, tuple(sorted(_market_ids(row)))))
    articles: list[Article] = []
    provenance: list[tuple[str, tuple[Article, ...]]] = []
    for url, versions in sorted(raw_by_url.items()):
        first_seen = min(row[0] for row in versions)
        first_rows = [row for row in versions if row[0] == first_seen]
        # Same-time duplicate rows are jointly available; deterministic title
        # selection removes dependence on input row order.
        first_title = min(row[1] for row in first_rows)
        first_markets = tuple(sorted({mid for row in first_rows for mid in row[2]}))
        articles.append(Article(url, first_title, first_seen.isoformat().replace("+00:00", "Z"), first_markets))
        all_versions = tuple(Article(url, title, seen.isoformat().replace("+00:00", "Z"), mids)
                             for seen, title, mids in sorted(versions, key=lambda row: (row[0], row[1], row[2])))
        provenance.append((url, all_versions))
    if not articles:
        return NewsGraph((), (), (), ())
    mrows: list[tuple[str, str]] = []
    for row in markets:
        mid = str(row.get("market_id", row.get("market_slug", "")))
        question = str(row.get("question", ""))
        if not mid or not question:
            raise ValueError("each market requires market_id/market_slug and question")
        mrows.append((mid, question))
    if len({mid for mid, _ in mrows}) != len(mrows):
        raise ValueError("market identifiers must be unique")
    headline_vectors = _normalize_rows(embedder.encode([a.title for a in articles], "document"), len(articles))
    market_vectors = (_normalize_rows(embedder.encode([q for _, q in mrows], "query"), len(mrows))
                      if mrows else np.empty((0, headline_vectors.shape[1])))
    if mrows and market_vectors.shape[1] != headline_vectors.shape[1]:
        raise ValueError("headline and market embeddings must have the same dimension")

    article_times = [_time(a.provider_seen_at) for a in articles]
    claims: list[ClaimNode] = []
    edges: list[MarketEdge] = []
    candidates: list[LinkCandidate] = []
    # Entries retain only headlines available so far. A new article joins an
    # existing component only when it clears the threshold against every
    # member, with URL-stable ties.
    clusters: dict[str, list[list[int]]] = {}
    article_order = sorted(range(len(articles)), key=lambda i: (article_times[i], articles[i].url))
    for i in article_order:
        day = article_times[i].date().isoformat()
        day_clusters = clusters.setdefault(day, [])
        eligible = []
        for cluster_index, members in enumerate(day_clusters):
            similarity = min(float(headline_vectors[i] @ headline_vectors[j]) for j in members)
            if similarity >= cluster_cosine_threshold:
                eligible.append((-similarity, min(articles[j].url for j in members), cluster_index))
        if eligible:
            cluster_index = min(eligible)[2]
            day_clusters[cluster_index].append(i)
        else:
            cluster_index = len(day_clusters)
            day_clusters.append([i])
        members = day_clusters[cluster_index]
        event_time = article_times[i].isoformat().replace("+00:00", "Z")
        claim_id = f"{day}-{sha1(articles[i].url.encode()).hexdigest()[:12]}"
        cluster_articles = tuple(articles[j] for j in sorted(members, key=lambda k: articles[k].url))
        representative = min(members, key=lambda j: (article_times[j], articles[j].url))
        cluster_first = min(members, key=lambda j: (article_times[j], articles[j].url))
        cluster_id = f"{day}-{sha1(articles[cluster_first].url.encode()).hexdigest()[:12]}"
        # This snapshot's availability time is its current event timestamp.
        # Later reports must not be backdated to the cluster's first report.
        claims.append(ClaimNode(claim_id, cluster_id, len(members) == 1, day,
                                articles[representative].title, articles[i].title,
                                articles[i].url, event_time,
                                event_time, cluster_articles))
        cluster_vector = headline_vectors[members].mean(axis=0)
        norm = np.linalg.norm(cluster_vector)
        if norm <= np.finfo(float).eps:
            # Opposing vectors can cancel; the earliest member is a stable
            # nonzero representative of the available cluster text.
            cluster_vector = headline_vectors[representative].copy()
        else:
            cluster_vector /= norm
        sims = market_vectors @ cluster_vector if mrows else np.empty(0)
        retrieved = {mid for i in members for mid in articles[i].market_ids}
        ranked = sorted(range(len(mrows)), key=lambda j: (-float(sims[j]), mrows[j][0]))
        selected = set(ranked[:min(top_k, len(ranked))])
        for j, (mid, _) in enumerate(mrows):
            edges.append(MarketEdge(claim_id, mid, float(sims[j])))
            if j in selected or mid in retrieved:
                candidates.append(LinkCandidate(claim_id, mid, float(sims[j]), mid in retrieved))
    claims.sort(key=lambda c: (c.event_time, c.claim_id))
    edges.sort(key=lambda e: (e.claim_id, e.market_id))
    candidates.sort(key=lambda c: (c.claim_id, -c.cosine_similarity, c.market_id))
    return NewsGraph(tuple(claims), tuple(edges), tuple(candidates), tuple(provenance))
