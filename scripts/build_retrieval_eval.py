"""Snapshot explicitly selected market queries and news for retrieval evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claim_reaction.core import iso, parse_time
from claim_reaction.retrieval import fetch_article_body


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id(url: str, title: str, provider_seen_at: str) -> str:
    identity = json.dumps([url, title, provider_seen_at],
                          ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(db_path: Path, selection_path: Path, out_path: Path,
          skip_fetch: bool = False) -> dict[str, Any]:
    if out_path.exists():
        raise FileExistsError(f"Output already exists: {out_path}")
    if not db_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Database and selection JSON must both exist")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or not isinstance(selection.get("queries"), list) \
            or not isinstance(selection.get("examples"), list):
        raise ValueError("Selection must contain queries and examples arrays")

    queries_by_id: dict[str, dict[str, Any]] = {}
    for item in selection["queries"]:
        if not isinstance(item, dict):
            raise ValueError("Each query must be an object")
        query_id, slug, cutoff, topic = (item.get(key) for key in
                                          ("query_id", "market_slug", "cutoff", "topic"))
        if not all(isinstance(value, str) and value for value in
                   (query_id, slug, cutoff, topic)):
            raise ValueError("Each query needs query_id, market_slug, cutoff, and topic")
        if query_id in queries_by_id:
            raise ValueError(f"Duplicate query_id: {query_id}")
        parse_time(cutoff)
        queries_by_id[query_id] = item

    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        queries: list[dict[str, Any]] = []
        for query_id, item in queries_by_id.items():
            market = connection.execute("SELECT question FROM markets WHERE slug = ?",
                                        (item["market_slug"],)).fetchone()
            if market is None:
                raise ValueError(f"Market not found: {item['market_slug']}")
            queries.append({"query_id": query_id, "market_slug": item["market_slug"],
                            "question": market["question"], "cutoff": iso(parse_time(item["cutoff"])),
                            "topic": item["topic"]})

        documents_by_id: dict[str, dict[str, Any]] = {}
        judgment_inputs: list[tuple[str, str, dict[str, Any], str]] = []
        seen_judgments: set[tuple[str, str]] = set()
        for item in selection["examples"]:
            if not isinstance(item, dict):
                raise ValueError("Each example must be an object")
            query_id = item.get("query_id")
            if query_id not in queries_by_id:
                raise ValueError(f"Unknown example query_id: {query_id}")
            source_id, source, source_market_slug = (item.get(key) for key in
                                                     ("source_id", "source", "source_market_slug"))
            if not all(isinstance(value, str) and value for value in
                       (source_id, source, source_market_slug)):
                raise ValueError("Each example needs source_id, source, and source_market_slug")
            annotation = item.get("annotation")
            if not isinstance(annotation, dict):
                raise ValueError("Each example needs an annotation object")
            news = connection.execute(
                "SELECT url, title, provider_seen_at, first_collected_at "
                "FROM news WHERE source = ? AND source_id = ? AND market_slug = ?",
                (source, source_id, source_market_slug),
            ).fetchone()
            if news is None:
                raise ValueError(f"News not found: {source}/{source_id}/{source_market_slug}")
            if not news["url"] or not news["title"] or not news["provider_seen_at"]:
                raise ValueError(f"News has no URL, headline, or provider time: {source_id}")
            if parse_time(news["provider_seen_at"]) > parse_time(queries_by_id[query_id]["cutoff"]):
                raise ValueError(f"News was first seen after query cutoff: {query_id}/{source_id}")
            doc_id = document_id(news["url"], news["title"], news["provider_seen_at"])
            if (query_id, doc_id) in seen_judgments:
                raise ValueError(f"Duplicate judgment for {query_id}/{doc_id}")
            seen_judgments.add((query_id, doc_id))
            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = {
                    "doc_id": doc_id, "url": news["url"], "title": news["title"],
                    "provider_seen_at": news["provider_seen_at"],
                    "first_collected_at": news["first_collected_at"],
                    "source_observations": [],
                }
            else:
                documents_by_id[doc_id]["first_collected_at"] = min(
                    documents_by_id[doc_id]["first_collected_at"],
                    news["first_collected_at"],
                )
            observation = {"source": source, "source_id": source_id,
                           "source_market_slug": source_market_slug,
                           "first_collected_at": news["first_collected_at"]}
            if observation not in documents_by_id[doc_id]["source_observations"]:
                documents_by_id[doc_id]["source_observations"].append(observation)
            judgment_inputs.append((query_id, doc_id, annotation,
                                    news["first_collected_at"]))
    finally:
        connection.close()

    def fetch_one(url: str) -> dict[str, Any]:
        try:
            body = fetch_article_body(url)
            return {"body": body,
                    "body_fetched_at": iso(datetime.now(timezone.utc)),
                    "body_fetch_status": "fetched", "body_fetch_error": None}
        except (OSError, ValueError, RuntimeError) as exc:
            return {"body": None, "body_fetched_at": None,
                    "body_fetch_status": "failed",
                    "body_fetch_error": f"{type(exc).__name__}: {exc}"}

    urls = sorted({document["url"] for document in documents_by_id.values()})
    body_by_url: dict[str, dict[str, Any]] = {}
    if skip_fetch:
        body_by_url = {
            url: {"body": None, "body_fetched_at": None,
                  "body_fetch_status": "skipped", "body_fetch_error": None}
            for url in urls
        }
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch_one, url): url for url in urls}
            for future in as_completed(futures):
                body_by_url[futures[future]] = future.result()

    for document in documents_by_id.values():
        document.update(body_by_url[document["url"]])
        document["source_observations"].sort(key=lambda row: (row["source"], row["source_id"],
                                                              row["source_market_slug"]))
    query_cutoffs = {row["query_id"]: parse_time(row["cutoff"]) for row in queries}
    judgments = []
    for query_id, doc_id, annotation, first_collected_at in judgment_inputs:
        fetched_at = documents_by_id[doc_id]["body_fetched_at"]
        body_retrospective = (parse_time(fetched_at) > query_cutoffs[query_id]
                              if fetched_at else None)
        headline_unverified = parse_time(first_collected_at) > query_cutoffs[query_id]
        judgments.append({"query_id": query_id, "doc_id": doc_id,
                          "annotation": annotation,
                          "headline_first_collected_at": first_collected_at,
                          "headline_snapshot_unverified": headline_unverified,
                          "body_retrospective": body_retrospective,
                          "point_in_time_eligible": not headline_unverified
                                                    and body_retrospective is not True})

    queries.sort(key=lambda row: row["query_id"])
    documents = sorted(documents_by_id.values(), key=lambda row: row["doc_id"])
    judgments.sort(key=lambda row: (row["query_id"], row["doc_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".retrieval-eval-", dir=str(out_path.parent)))
    try:
        outputs = {"queries.jsonl": queries, "documents.jsonl": documents,
                   "judgments.jsonl": judgments}
        for name, rows in outputs.items():
            write_jsonl(staging / name, rows)
        manifest = {
            "schema_version": 1,
            "created_at": iso(datetime.now(timezone.utc)),
            "evaluation_scope": {
                "selection": "Curated sample; not representative of the full news pool.",
                "unlisted_pairs": "Unjudged; not negative examples.",
                "headline_labels": "Apply only to headline snapshots, not fetched article bodies.",
                "historical_headlines": "Provider seen time can predate collection; the stored headline may have changed before collection. Check each judgment's headline_snapshot_unverified flag.",
                "recall": "No recall claim outside the judged pool.",
                "splits": "No training or test split is implied.",
            },
            "source_database_sha256": file_sha256(db_path),
            "selection_sha256": file_sha256(selection_path),
            "counts": {"queries": len(queries), "documents": len(documents),
                       "judgments": len(judgments), "unique_urls": len(body_by_url),
                       "body_fetched": sum(row["body_fetch_status"] == "fetched"
                                           for row in body_by_url.values()),
                       "body_failed": sum(row["body_fetch_status"] == "failed"
                                          for row in body_by_url.values()),
                       "body_skipped": sum(row["body_fetch_status"] == "skipped"
                                           for row in body_by_url.values()),
                       "headline_snapshot_unverified": sum(
                           row["headline_snapshot_unverified"] for row in judgments),
                       "point_in_time_eligible": sum(
                           row["point_in_time_eligible"] for row in judgments)},
            "files_sha256": {name: file_sha256(staging / name) for name in outputs},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if out_path.exists():
            raise FileExistsError(f"Output already exists: {out_path}")
        os.replace(staging, out_path)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot selected retrieval judgments")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.db, args.selection, args.out, args.skip_fetch),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
