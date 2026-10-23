"""Build a deterministic, market-balanced article body sample for research."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claim_reaction.core import iso, parse_time
from claim_reaction.forecast import HORIZONS_MINUTES, load_forecast_rows
from claim_reaction.retrieval import fetch_article_body


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _select_round_robin(rows: list[Any], limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select an even market allocation, spread across each market's time range."""
    by_market: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        slug, url = row.market_slug, row.article_url
        if not slug or not url or url in by_market[slug]:
            continue
        by_market[slug][url] = {
            "market_slug": slug,
            "article_url": url,
            "headline": row.headline,
            "provider_seen_at": row.available_at,
            "first_collected_at": row.first_collected_at,
        }

    ordered_markets = sorted(by_market)
    per_market = {
        slug: sorted(by_market[slug].values(),
                     key=lambda item: (item["provider_seen_at"], item["article_url"], item["headline"]))
        for slug in ordered_markets
    }
    quotas = {slug: 0 for slug in ordered_markets}
    while sum(quotas.values()) < limit:
        added = False
        for slug in ordered_markets:
            if quotas[slug] < len(per_market[slug]):
                quotas[slug] += 1
                added = True
                if sum(quotas.values()) == limit:
                    break
        if not added:
            break

    selected: list[dict[str, Any]] = []
    for slug in ordered_markets:
        items = per_market[slug]
        count = quotas[slug]
        if count == 1:
            indices = [(len(items) - 1) // 2]
        elif count > 1:
            indices = [round(position * (len(items) - 1) / (count - 1))
                       for position in range(count)]
        else:
            indices = []
        selected.extend(items[index] for index in indices)
    return selected, {slug: len(items) for slug, items in per_market.items()}


def _fetch_one(url: str, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            body = fetch_article_body(url)
            fetched_at = iso(datetime.now(timezone.utc)) or ""
            return {"body": body, "body_fetched_at": fetched_at,
                    "body_fetch_status": "fetched", "body_fetch_error": None}
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(0.25 * (2 ** attempt), 2.0))
    assert last_error is not None
    return {"body": None, "body_fetched_at": None, "body_fetch_status": "failed",
            "body_fetch_error": f"{type(last_error).__name__}: {last_error}"[:500]}


def build(db_path: Path, out_path: Path, limit: int = 200,
          horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES,
          workers: int = 4, retries: int = 2,
          reuse_documents_path: Path | None = None) -> dict[str, Any]:
    if out_path.exists():
        raise FileExistsError(f"Output already exists: {out_path}")
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if limit <= 0 or workers <= 0 or retries < 0:
        raise ValueError("limit and workers must be positive; retries must be nonnegative")

    forecast_rows, excluded = load_forecast_rows(str(db_path), horizons_minutes=horizons_minutes)
    selected, eligible_by_market = _select_round_robin(forecast_rows, limit)
    urls = sorted({row["article_url"] for row in selected})
    body_by_url: dict[str, dict[str, Any]] = {}
    reusable: dict[str, dict[str, Any]] = {}
    if reuse_documents_path is not None and reuse_documents_path.is_file():
        with reuse_documents_path.open(encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                if item.get("body_fetch_status") == "fetched" and item.get("body") and item.get("body_fetched_at"):
                    reusable.setdefault(item["url"], item)
    for url in urls:
        if url in reusable:
            item = reusable[url]
            body_by_url[url] = {"body": item["body"], "body_fetched_at": item["body_fetched_at"],
                                "body_fetch_status": "fetched", "body_fetch_error": None,
                                "body_fetch_origin": "starter_v1_snapshot"}
    remaining_urls = [url for url in urls if url not in body_by_url]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, url, retries): url for url in remaining_urls}
        for future in as_completed(futures):
            result = future.result()
            result["body_fetch_origin"] = "network_fetch"
            body_by_url[futures[future]] = result

    documents = []
    for selection in selected:
        body = body_by_url[selection["article_url"]]
        fetched_at = body["body_fetched_at"]
        retrospective = (parse_time(fetched_at) > parse_time(selection["provider_seen_at"])
                         if fetched_at else None)
        documents.append({**selection, **body, "body_retrospective": retrospective})
    documents.sort(key=lambda row: (row["market_slug"], row["provider_seen_at"], row["article_url"]))

    counts_by_market = Counter(row["market_slug"] for row in selected)
    unique_urls = len(urls)
    manifest = {
        "schema_version": 1,
        "created_at": iso(datetime.now(timezone.utc)),
        "provenance": {
            "source_database": str(db_path.resolve()),
            "source_database_sha256": file_sha256(db_path),
            "eligibility_source": "claim_reaction.forecast.load_forecast_rows; rows with a reaction label are eligible",
            "selection": "Allocate pairs evenly across sorted markets; within each market, select evenly spaced observations spanning provider time. Outcomes not read.",
            "horizons_minutes": list(horizons_minutes),
            "body_fetcher": "claim_reaction.retrieval.fetch_article_body",
            "body_retrospective_rule": "true when body_fetched_at is later than provider_seen_at",
            "workers": workers,
            "retries": retries,
            "reused_body_source": (str(reuse_documents_path.resolve())
                                   if reuse_documents_path is not None and reuse_documents_path.is_file()
                                   else None),
            "reused_body_source_sha256": (file_sha256(reuse_documents_path)
                                          if reuse_documents_path is not None and reuse_documents_path.is_file()
                                          else None),
        },
        "selection": {
            "target_count": limit,
            "eligible_labeled_pairs": len({(row.market_slug, row.article_url) for row in forecast_rows}),
            "eligible_pairs_by_market": eligible_by_market,
            "selected_pairs_by_market": dict(sorted(counts_by_market.items())),
            "forecast_exclusions": excluded,
        },
        "coverage": {
            "selected_pairs": len(documents),
            "markets_with_selected_pairs": len(counts_by_market),
            "unique_urls_fetched": unique_urls,
            "reused_urls": sum(url in reusable for url in urls),
        },
        "counts": {
            "body_fetched": sum(value["body_fetch_status"] == "fetched" for value in body_by_url.values()),
            "body_failed": sum(value["body_fetch_status"] == "failed" for value in body_by_url.values()),
            "retrospective_bodies": sum(row["body_retrospective"] is True for row in documents),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Claim the destination atomically so an existing sample can never be replaced.
    out_path.mkdir()
    try:
        documents_path = out_path / "documents.jsonl"
        write_jsonl(documents_path, documents)
        manifest["files_sha256"] = {"documents.jsonl": file_sha256(documents_path)}
        (out_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(out_path, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/political_25.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("data/body_sample_v1"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--reuse-documents", type=Path,
                        default=Path("data/evaluation/starter_v1/documents.jsonl"))
    parser.add_argument("--horizons-minutes", type=int, nargs="+", default=list(HORIZONS_MINUTES))
    args = parser.parse_args()
    print(json.dumps(build(args.db, args.out, args.limit, tuple(args.horizons_minutes),
                           args.workers, args.retries, args.reuse_documents), sort_keys=True))


if __name__ == "__main__":
    main()
