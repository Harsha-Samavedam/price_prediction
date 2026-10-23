"""Build retrospective claim and market graph from saved GDELT headlines."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claim_reaction.news_graph import build_news_graph
from claim_reaction.retrieval import HuggingFaceEmbedder


def load_graph_inputs(db_path: str) -> tuple[list[dict], list[dict]]:
    db = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        records = [dict(row) for row in db.execute(
            "SELECT url,title,provider_seen_at,market_slug FROM news "
            "WHERE source='gdelt-doc' AND provider_seen_at IS NOT NULL AND url!='' AND title!='' "
            "ORDER BY provider_seen_at,url,market_slug"
        )]
        markets = [dict(row) for row in db.execute(
            "SELECT slug AS market_slug,question FROM markets ORDER BY slug"
        )]
        return records, markets
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cluster-threshold", required=True, type=float)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()
    records, markets = load_graph_inputs(args.db)
    graph = build_news_graph(records, markets,
                             HuggingFaceEmbedder(args.embedding_model),
                             args.cluster_threshold, args.top_k)
    report = {
        "provenance": {
            "source_database": str(Path(args.db).resolve()),
            "source": "retrospectively collected GDELT headlines",
            "event_time": "provider_seen_at (GDELT indexed time, not local observation)",
            "embedding_model": args.embedding_model,
            "cluster_cosine_threshold": args.cluster_threshold,
            "candidate_top_k": args.top_k,
            "market_edge_rule": "all markets retain cosine; NLI candidates are top-k plus original query matches",
            "outcomes_or_prices_used": False,
        },
        "counts": {"articles": len({r["url"] for r in records}),
                   "claim_snapshots": len(graph.claims),
                   "distinct_claims": sum(item.is_first_report for item in graph.claims),
                   "market_edges": len(graph.market_edges),
                   "link_candidates": len(graph.link_candidates)},
        "claims": [asdict(item) for item in graph.claims],
        "market_edges": [asdict(item) for item in graph.market_edges],
        "link_candidates": [asdict(item) for item in graph.link_candidates],
        "article_provenance": [asdict(article) for _, versions in graph.article_provenance
                               for article in versions],
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["counts"], "output": str(target.resolve())}, indent=2))


if __name__ == "__main__":
    main()
