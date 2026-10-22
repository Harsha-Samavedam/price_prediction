from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from .core import PricePoint, parse_time
from .replay import ReactionRow, replay_claims
from .retrieval import HuggingFaceEmbedder, HuggingFaceGenerator, retrieve_details
from .sources import (Polymarket, gdelt_doc, gdelt_gkg_archive, guardian,
                      mediacloud, newsapi, nyt, rss)
from .store import Store


NEWS_SOURCES = ("gdelt-doc", "gdelt-gkg", "guardian", "nyt", "newsapi", "mediacloud", "rss")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m claim_reaction")
    p.add_argument("--db", required=True, help="SQLite database path")
    sub = p.add_subparsers(dest="command", required=True)

    market = sub.add_parser("market", help="Ingest one Polymarket market by slug")
    market.add_argument("--slug", required=True)

    prices_api = sub.add_parser("prices-api", help="Fetch YES-price history")
    prices_api.add_argument("--slug", required=True)
    prices_api.add_argument("--start", required=True, help="ISO-8601 timestamp with timezone")
    prices_api.add_argument("--end", required=True)
    prices_api.add_argument("--api", required=True, choices=("data-v2", "clob"))

    prices_csv = sub.add_parser("prices-csv", help="Import timestamp,price CSV")
    prices_csv.add_argument("--slug", required=True)
    prices_csv.add_argument("--file", required=True)

    news = sub.add_parser("news", help="Ingest candidate news observations")
    news.add_argument("--slug", required=True)
    news.add_argument("--source", required=True, choices=NEWS_SOURCES)
    news.add_argument("--query", required=True)
    news.add_argument("--start", help="ISO-8601 timestamp with timezone")
    news.add_argument("--end", help="ISO-8601 timestamp with timezone")
    news.add_argument("--collection-id", type=int, help="Required for Media Cloud")
    news.add_argument("--feed-url", help="Required for RSS")

    claims = sub.add_parser("claims-csv", help="Import manually verified claims")
    claims.add_argument("--file", required=True,
                        help="Columns: market_slug,claim_text,available_at,article_url[,claim_id]")

    news_csv = sub.add_parser("news-csv", help="Export candidate articles for review")
    news_csv.add_argument("--slug", required=True)
    news_csv.add_argument("--out", required=True)

    replay = sub.add_parser("replay", help="Label claim reactions from stored prices")
    replay.add_argument("--slug", required=True)
    replay.add_argument("--price-source", required=True,
                        choices=("csv", "polymarket-data-v2", "polymarket-clob"))
    replay.add_argument("--horizon-minutes", required=True, type=int)
    replay.add_argument("--max-price-age-minutes", required=True, type=int)
    replay.add_argument("--out", required=True, help="Output CSV path")

    audit = sub.add_parser("audit", help="Summarize ingested coverage")
    audit.add_argument("--slug", required=True)

    details = sub.add_parser("retrieve-details", help="Retrieve and rank source-backed details")
    details.add_argument("--slug", required=True)
    details.add_argument("--cutoff", required=True, help="ISO-8601 timestamp with timezone")
    details.add_argument("--embedding-model", default="intfloat/multilingual-e5-small",
                         help="Hugging Face encoder model ID")
    details.add_argument("--generation-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                         help="Hugging Face generation model ID")
    details.add_argument("--bm25-candidates", required=True, type=int)
    details.add_argument("--dense-candidates", required=True, type=int)
    details.add_argument("--fusion-k", required=True, type=int)
    details.add_argument("--max-clusters", required=True, type=int)
    details.add_argument("--bodies-per-cluster", required=True, type=int)
    details.add_argument("--article-similarity-threshold", required=True, type=float)
    details.add_argument("--detail-similarity-threshold", required=True, type=float)
    details.add_argument("--generation-max-input-tokens", required=True, type=int)
    details.add_argument("--generation-max-new-tokens", required=True, type=int)
    details.add_argument("--min-detail-characters", required=True, type=int)
    details.add_argument("--out", required=True, help="Output JSON path")
    return p


def run(args: argparse.Namespace) -> dict:
    if args.command == "retrieve-details":
        positive = ("bm25_candidates", "dense_candidates", "fusion_k", "max_clusters",
                    "bodies_per_cluster", "generation_max_input_tokens",
                    "generation_max_new_tokens", "min_detail_characters")
        if any(getattr(args, key) <= 0 for key in positive):
            raise ValueError("Retrieval limits and token counts must be positive")
        for key in ("article_similarity_threshold", "detail_similarity_threshold"):
            if not -1 <= getattr(args, key) <= 1:
                raise ValueError(f"{key} must be between -1 and 1")
        embedder = HuggingFaceEmbedder(args.embedding_model)
        generator = HuggingFaceGenerator(args.generation_model,
                                        args.generation_max_input_tokens,
                                        args.generation_max_new_tokens)
        with Store(args.db) as store:
            result = retrieve_details(
                store, args.slug, parse_time(args.cutoff), embedder, generator,
                bm25_limit=args.bm25_candidates, dense_limit=args.dense_candidates,
                fusion_k=args.fusion_k, max_clusters=args.max_clusters,
                bodies_per_cluster=args.bodies_per_cluster,
                article_similarity_threshold=args.article_similarity_threshold,
                detail_similarity_threshold=args.detail_similarity_threshold,
                min_detail_characters=args.min_detail_characters,
            )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"market_slug": args.slug, "candidate_articles": result["candidate_articles"],
                "details": len(result["details"]), "output": str(out.resolve())}
    with Store(args.db) as store:
        if args.command == "market":
            market = Polymarket().market_by_slug(args.slug)
            store.put_market(market)
            return {"market_slug": market.slug, "question": market.question,
                    "yes_asset_id": market.yes_asset_id}

        if args.command == "prices-api":
            market = store.get_market(args.slug)
            if market is None:
                raise ValueError("Ingest the market before fetching its prices")
            points = Polymarket().prices(market, parse_time(args.start),
                                         parse_time(args.end), args.api)
            return {"market_slug": args.slug, "source": args.api,
                    "price_points": store.put_prices(points)}

        if args.command == "prices-csv":
            with open(args.file, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            points = [PricePoint(args.slug, parse_time(r["timestamp"]),
                                 float(r["price"]),
                                 int(r["resolution_seconds"]) if r.get("resolution_seconds") else None,
                                 "csv") for r in rows]
            return {"market_slug": args.slug, "price_points": store.put_prices(points)}

        if args.command == "news":
            if args.source == "rss":
                if not args.feed_url:
                    raise ValueError("--feed-url is required for RSS")
                records = rss(args.feed_url, args.slug, args.query)
            else:
                if not args.start or not args.end:
                    raise ValueError("--start and --end are required for historical news")
                start, end = parse_time(args.start), parse_time(args.end)
                if end <= start:
                    raise ValueError("end must be after start")
                if args.source == "gdelt-doc":
                    records = gdelt_doc(args.query, start, end, args.slug)
                elif args.source == "gdelt-gkg":
                    records = gdelt_gkg_archive(args.query, start, end, args.slug)
                elif args.source == "guardian":
                    records = guardian(args.query, start, end, args.slug)
                elif args.source == "nyt":
                    records = nyt(args.query, start, end, args.slug)
                elif args.source == "newsapi":
                    records = newsapi(args.query, start, end, args.slug)
                elif args.source == "mediacloud":
                    if args.collection_id is None:
                        raise ValueError("--collection-id is required for Media Cloud")
                    records = mediacloud(args.query, start, end, args.slug, args.collection_id)
                else:
                    raise ValueError(f"Unsupported source: {args.source}")
            count = store.put_news(records)
            return {"market_slug": args.slug, "source": args.source,
                    "candidate_articles": count}

        if args.command == "claims-csv":
            count = 0
            with open(args.file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    slug = row["market_slug"]
                    claim_text = row["claim_text"]
                    available_at = parse_time(row["available_at"])
                    article_url = row.get("article_url") or None
                    claim_id = row.get("claim_id") or hashlib.sha256(
                        f"{slug}\0{claim_text}\0{available_at.isoformat()}".encode()
                    ).hexdigest()[:20]
                    store.put_claim(claim_id, slug, claim_text, available_at, article_url)
                    count += 1
            return {"claims_imported": count}

        if args.command == "news-csv":
            rows = store.db.execute(
                "SELECT source, source_id, market_slug, query, url, title, "
                "published_at, provider_seen_at, first_collected_at "
                "FROM news WHERE market_slug = ? ORDER BY COALESCE(provider_seen_at, "
                "first_collected_at), source, source_id",
                (args.slug,),
            ).fetchall()
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=("source", "source_id", "market_slug",
                                                       "query", "url", "title", "published_at",
                                                       "provider_seen_at", "first_collected_at"))
                writer.writeheader()
                writer.writerows(dict(row) for row in rows)
            return {"market_slug": args.slug, "candidate_articles": len(rows),
                    "output": str(out.resolve())}

        if args.command == "replay":
            rows, excluded = replay_claims(
                store.claims(args.slug), store.prices(args.slug, args.price_source),
                args.horizon_minutes, args.max_price_age_minutes,
            )
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(ReactionRow.__dataclass_fields__))
                writer.writeheader()
                writer.writerows(row.to_dict() for row in rows)
            return {"market_slug": args.slug, "labeled_claims": len(rows),
                    "excluded": excluded, "output": str(out.resolve())}

        if args.command == "audit":
            market = store.get_market(args.slug)
            prices = store.db.execute(
                "SELECT source, COUNT(*) AS count, MIN(timestamp) AS first, "
                "MAX(timestamp) AS last FROM prices WHERE market_slug = ? GROUP BY source",
                (args.slug,),
            ).fetchall()
            return {"market_slug": args.slug,
                    "market_question": market.question if market else None,
                    "prices": [dict(r) for r in prices],
                    "news": [dict(r) for r in store.news_counts(args.slug)],
                    "claims": len(store.claims(args.slug))}

    raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    args = parser().parse_args()
    try:
        print(json.dumps(run(args), indent=2))
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
