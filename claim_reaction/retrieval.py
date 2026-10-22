"""Retrieve source-backed, market-specific details from a global news index."""

from __future__ import annotations

import json
import ipaddress
import math
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .core import iso, parse_time
from .store import Store


@dataclass(frozen=True)
class Article:
    url: str
    title: str
    first_seen_at: str
    first_collected_at: str
    source: str


def refresh_article_index(store: Store) -> int:
    """Build one article per URL across all collection queries and markets."""
    rows = store.db.execute(
        "SELECT url, title, source, provider_seen_at, first_collected_at "
        "FROM news WHERE url != '' AND title != '' "
        "ORDER BY COALESCE(provider_seen_at, first_collected_at), first_collected_at, url"
    ).fetchall()
    articles: dict[str, Article] = {}
    for row in rows:
        seen = row["provider_seen_at"] or row["first_collected_at"]
        if row["url"] not in articles:
            articles[row["url"]] = Article(
                row["url"], row["title"], seen, row["first_collected_at"], row["source"]
            )
    with store.db:
        store.db.execute("DELETE FROM article_fts")
        store.db.execute("DELETE FROM article_index")
        store.db.executemany(
            "INSERT INTO article_index VALUES (?, ?, ?, ?, ?)",
            [(a.url, a.title, a.first_seen_at, a.first_collected_at, a.source)
             for a in articles.values()],
        )
        store.db.executemany(
            "INSERT INTO article_fts(url, title) VALUES (?, ?)",
            [(a.url, a.title) for a in articles.values()],
        )
    return len(articles)


def _eligible_articles(store: Store, cutoff: datetime) -> list[Article]:
    rows = store.db.execute(
        "SELECT * FROM article_index WHERE first_seen_at <= ? ORDER BY first_seen_at, url",
        (iso(cutoff),),
    ).fetchall()
    return [Article(**dict(row)) for row in rows]


def _bm25_hits(store: Store, question: str, cutoff: datetime, limit: int) -> list[str]:
    terms = list(dict.fromkeys(re.findall(r"\w+", question.casefold(), re.UNICODE)))
    if not terms or limit <= 0:
        return []
    expression = " OR ".join('"' + term.replace('"', '') + '"' for term in terms)
    rows = store.db.execute(
        "SELECT article_fts.url FROM article_fts "
        "JOIN article_index ON article_index.url = article_fts.url "
        "WHERE article_fts MATCH ? AND article_index.first_seen_at <= ? "
        "ORDER BY bm25(article_fts) LIMIT ?",
        (expression, iso(cutoff), limit),
    ).fetchall()
    return [row["url"] for row in rows]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


class HuggingFaceEmbedder:
    def __init__(self, model_id: str) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for local embeddings") from exc
        self.torch = torch
        self.model_id = model_id
        self.e5_prefixes = "e5" in model_id.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], role: str) -> list[list[float]]:
        if role not in ("query", "document", "symmetric"):
            raise ValueError("Embedding role must be query, document, or symmetric")
        if not texts:
            return []
        if self.e5_prefixes:
            prefix = "passage: " if role == "document" else "query: "
            texts = [prefix + value for value in texts]
        vectors = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            tokens = self.tokenizer(batch, padding=True, truncation=True,
                                    return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                output = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors


def _article_vectors(store: Store, articles: list[Article], embedder: Any) -> dict[str, list[float]]:
    cached_rows = store.db.execute(
        "SELECT url, title, vector_json FROM article_embeddings WHERE model_id = ?",
        (embedder.model_id,),
    ).fetchall()
    cached = {row["url"]: (row["title"], json.loads(row["vector_json"]))
              for row in cached_rows}
    missing = [a for a in articles if a.url not in cached or cached[a.url][0] != a.title]
    if missing:
        vectors = embedder.encode([a.title for a in missing], "document")
        with store.db:
            store.db.executemany(
                "INSERT OR REPLACE INTO article_embeddings VALUES (?, ?, ?, ?)",
                [(a.url, embedder.model_id, a.title, json.dumps(v))
                 for a, v in zip(missing, vectors)],
            )
        cached.update({a.url: (a.title, v) for a, v in zip(missing, vectors)})
    return {a.url: cached[a.url][1] for a in articles}


def hybrid_candidates(store: Store, question: str, cutoff: datetime,
                      embedder: Any, bm25_limit: int, dense_limit: int,
                      fusion_k: int) -> list[tuple[Article, float, list[float]]]:
    articles = _eligible_articles(store, cutoff)
    by_url = {a.url: a for a in articles}
    if not articles:
        return []
    bm25 = _bm25_hits(store, question, cutoff, bm25_limit)
    vectors = _article_vectors(store, articles, embedder)
    query_vector = embedder.encode([question], "query")[0]
    dense = sorted(articles, key=lambda a: (-_cosine(query_vector, vectors[a.url]), a.url))
    dense_urls = [a.url for a in dense[:dense_limit]]
    scores: dict[str, float] = {}
    for ranking in (bm25, dense_urls):
        for rank, url in enumerate(ranking, 1):
            scores[url] = scores.get(url, 0.0) + 1.0 / (fusion_k + rank)
    ordered = sorted(scores, key=lambda url: (-scores[url], url))
    return [(by_url[url], scores[url], vectors[url]) for url in ordered]


def semantic_clusters(items: Iterable[tuple[Any, list[float]]],
                      threshold: float) -> list[list[Any]]:
    clusters: list[list[Any]] = []
    representatives: list[list[float]] = []
    for item, vector in items:
        best = max(range(len(representatives)),
                   key=lambda idx: _cosine(vector, representatives[idx]), default=None)
        if best is not None and _cosine(vector, representatives[best]) >= threshold:
            clusters[best].append(item)
        else:
            clusters.append([item])
            representatives.append(vector)
    return clusters


def _require_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Article URL must be public HTTP(S) without credentials")
    try:
        addresses = socket.getaddrinfo(parsed.hostname,
                                       parsed.port or (443 if parsed.scheme == "https" else 80),
                                       type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("Article hostname could not be resolved") from exc
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Article URL resolves to a non-public address")


class _PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_article_body(url: str) -> str:
    _require_public_url(url)
    try:
        import trafilatura
    except ImportError as exc:
        raise RuntimeError("Install trafilatura for article body extraction") from exc
    request = Request(url, headers={"User-Agent": "ClaimReactionResearch/1.0"})
    with build_opener(_PublicRedirects()).open(request, timeout=15) as response:
        _require_public_url(response.geturl())
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("Article exceeds 2 MB fetch limit")
    html = payload.decode("utf-8", errors="replace")
    body = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not body:
        raise ValueError("No readable article body")
    return body


def cached_body(store: Store, url: str) -> tuple[str, str]:
    row = store.db.execute("SELECT body, fetched_at FROM article_bodies WHERE url = ?",
                           (url,)).fetchone()
    if row is not None:
        return row["body"], row["fetched_at"]
    body = fetch_article_body(url)
    fetched_at = iso(datetime.now(timezone.utc)) or ""
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO article_bodies VALUES (?, ?, ?)",
                         (url, fetched_at, body))
    return body, fetched_at


def _json_value(text: str) -> dict[str, Any] | list[Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise ValueError("Local model did not return a JSON object or array")


class HuggingFaceGenerator:
    def __init__(self, model_id: str, max_input_tokens: int, max_new_tokens: int) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for local generation") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        self.encoder_decoder = bool(config.is_encoder_decoder)
        model_class = AutoModelForSeq2SeqLM if self.encoder_decoder else AutoModelForCausalLM
        self.model = model_class.from_pretrained(model_id)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        if not self.encoder_decoder and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "Return only valid JSON matching the user's requested schema."},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.max_input_tokens).to(self.device)
        with self.torch.no_grad():
            generated = self.model.generate(**inputs,
                                            max_new_tokens=self.max_new_tokens,
                                            do_sample=False,
                                            pad_token_id=self.tokenizer.pad_token_id)
        tokens = generated[0]
        if not self.encoder_decoder:
            tokens = tokens[inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def body_chunks(self, body: str, prompt_prefix: str) -> list[str]:
        if not self.encoder_decoder and self.tokenizer.chat_template:
            prompt_prefix = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "Return only valid JSON matching the user's requested schema."},
                 {"role": "user", "content": prompt_prefix}],
                tokenize=False, add_generation_prompt=True)
        prefix_size = len(self.tokenizer.encode(prompt_prefix, add_special_tokens=True))
        room = self.max_input_tokens - prefix_size - 8
        if room <= 0:
            raise ValueError("max_input_tokens is too small for the extraction prompt")
        tokens = self.tokenizer.encode(body, add_special_tokens=False)
        return [self.tokenizer.decode(tokens[start:start + room], skip_special_tokens=True)
                for start in range(0, len(tokens), room)]


def extract_details(generator: Any, body: str,
                    min_detail_characters: int) -> list[dict[str, str]]:
    prefix = ("Extract distinct, complete factual clauses copied verbatim from the text. "
              "Each clause must express one fact and be an exact contiguous span of the text. "
              "Skip questions, opinions, predictions, and fragments. Do not paraphrase or infer. "
              "Return only a JSON object whose details key is an array of clause strings. "
              "Use an empty array if there are no factual clauses. Text:\n")
    chunks = generator.body_chunks(body, prefix)
    details = []
    seen = set()
    for chunk in chunks:
        try:
            response = _json_value(generator.generate(prefix + chunk))
        except ValueError:
            continue
        values = response.get("details") if isinstance(response, dict) else response
        if not isinstance(values, list):
            continue
        for value in values:
            detail = str(value).strip() if isinstance(value, str) else ""
            key = detail.casefold()
            if (len(detail) >= min_detail_characters
                    and not detail.rstrip().endswith(("?", "？"))
                    and detail.casefold() in chunk.casefold() and key not in seen):
                details.append({"text": detail, "support": detail})
                seen.add(key)
    return details


def score_detail(generator: Any, question: str, detail: str) -> float:
    prompt = ("For the binary prediction market question, score how strongly the factual "
              "detail supports YES versus NO. Return only JSON with a numeric strength "
              "between -1 and 1. Positive supports YES, negative supports NO, zero is "
              "irrelevant or insufficient. This is evidence strength, not predicted price "
              "change. Question: " + question + "\nDetail: " + detail +
              "\nReturn the JSON object now:")
    try:
        response = _json_value(generator.generate(prompt))
        value = response.get("strength") if isinstance(response, dict) else None
        if isinstance(value, bool):
            return 0.0
        score = float(value)
    except (ValueError, TypeError):
        return 0.0
    return score if math.isfinite(score) and -1 <= score <= 1 else 0.0


def retrieve_details(store: Store, market_slug: str, cutoff: datetime,
                     embedder: Any, generator: Any, *, bm25_limit: int,
                     dense_limit: int, fusion_k: int, max_clusters: int,
                     bodies_per_cluster: int, article_similarity_threshold: float,
                     detail_similarity_threshold: float,
                     min_detail_characters: int) -> dict[str, Any]:
    market = store.get_market(market_slug)
    if market is None:
        raise ValueError("Market not found")
    refresh_article_index(store)
    candidates = hybrid_candidates(store, market.question, cutoff, embedder,
                                   bm25_limit, dense_limit, fusion_k)
    clustering_vectors = embedder.encode([article.title for article, _, _ in candidates],
                                         "symmetric")
    article_clusters = semantic_clusters(
        [((article, rank), vector)
         for (article, rank, _), vector in zip(candidates, clustering_vectors)],
        article_similarity_threshold,
    )
    chosen = article_clusters[:max_clusters]
    observations: list[dict[str, Any]] = []
    extracted: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(chosen):
        for article, rank in cluster[:bodies_per_cluster]:
            record: dict[str, Any] = {
                "url": article.url, "title": article.title, "source": article.source,
                "provider_first_seen_at": article.first_seen_at,
                "headline_first_collected_at": article.first_collected_at,
                "article_cluster": cluster_index, "retrieval_score": rank,
                "headline_snapshot_unverified": (
                    parse_time(article.first_collected_at) > cutoff),
            }
            try:
                body, fetched_at = cached_body(store, article.url)
                record["body_fetched_at"] = fetched_at
                record["body_retrospective"] = parse_time(fetched_at) > cutoff
                record["text_scope"] = "body"
                text = body
            except (OSError, ValueError) as exc:
                record["fetch_error"] = f"{type(exc).__name__}: {exc}"
                record["body_retrospective"] = False
                record["text_scope"] = "headline"
                text = article.title
            record["point_in_time_eligible"] = (
                not record["body_retrospective"]
                and not record["headline_snapshot_unverified"])
            details = extract_details(generator, text, min_detail_characters)
            record["details_extracted"] = len(details)
            for detail in details:
                extracted.append({"text": detail["text"], "support": detail["support"],
                                  "source": record.copy()})
            observations.append(record)
    for item in extracted:
        item["strength"] = score_detail(generator, market.question, item["text"])
    detail_clusters = []
    for direction in (-1, 0, 1):
        same_direction = [item for item in extracted
                          if (item["strength"] > 0) - (item["strength"] < 0) == direction]
        if same_direction:
            vectors = embedder.encode([item["text"] for item in same_direction], "symmetric")
            detail_clusters.extend(semantic_clusters(zip(same_direction, vectors),
                                                     detail_similarity_threshold))
    ranked = []
    for cluster in detail_clusters:
        representative = max(cluster, key=lambda item: (abs(item["strength"]), item["text"]))
        strength = representative["strength"]
        sources = [item["source"] | {"support": item["support"]} for item in cluster]
        ranked.append({"detail": representative["text"], "strength": strength,
                       "direction": "yes" if strength > 0 else "no" if strength < 0 else "unclear",
                       "sources": sources,
                       "any_retrospective_body": any(s["body_retrospective"] for s in sources),
                       "point_in_time_eligible": all(s["point_in_time_eligible"] for s in sources)})
    ranked.sort(key=lambda row: (-abs(row["strength"]), row["detail"]))
    attempted = {row["url"]: row["text_scope"] for row in observations}
    cluster_members = [
        [{"url": article.url, "title": article.title,
          "provider_first_seen_at": article.first_seen_at,
          "retrieval_score": rank,
          "retrieved_text_scope": attempted.get(article.url)}
         for article, rank in cluster]
        for cluster in chosen
    ]
    return {"market_slug": market_slug, "question": market.question,
            "cutoff": iso(cutoff), "candidate_articles": len(candidates),
            "article_clusters": len(article_clusters),
            "selected_article_clusters": len(chosen),
            "selected_articles": observations, "selected_cluster_members": cluster_members,
            "details": ranked,
            "historical_body_policy": "Bodies fetched after cutoff are retrospective and not point-in-time evidence"}
