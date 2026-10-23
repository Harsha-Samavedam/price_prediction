"""Compare retrospective article-body evidence with headline forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from claim_reaction.body_forecast import (LocalNLIClassifier, add_body_features, add_relevance_features,
                                         compare_body_and_headline,
                                         select_usable_body_rows)
from claim_reaction.forecast import load_forecast_rows
from claim_reaction.retrieval import HuggingFaceEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/political_25.sqlite")
    parser.add_argument("--documents", required=True, help="Body documents JSONL")
    parser.add_argument("--out", default="body_forecast_results.json")
    parser.add_argument("--cache", default="body_forecast_cache.json")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--nli-model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--hypotheses", default="data/market_hypotheses_v1.json")
    parser.add_argument("--min-detail-characters", type=int, default=20)
    args = parser.parse_args()
    if args.min_detail_characters <= 0:
        parser.error("minimum detail length must be positive")

    all_rows, excluded = load_forecast_rows(args.db)
    rows, body_coverage = select_usable_body_rows(all_rows, args.documents)
    embedder = HuggingFaceEmbedder(args.embedding_model)
    classifier = LocalNLIClassifier(args.nli_model)
    headline_features, headline_names = add_relevance_features(rows, embedder)
    body_features, body_names, provenance = add_body_features(
        rows, args.documents, embedder, classifier, args.hypotheses,
        args.cache, args.min_detail_characters)
    report = compare_body_and_headline(rows, body_features, headline_features)
    report.update({"event_source": "GDELT indexed headlines at provider_seen_at",
                   "label_source": "Polymarket YES-share price history",
                   "label_limitation": "Retrospective GDELT observations are provisional event times, not live news arrival times.",
                   "feature_names": {"body_based": body_names, "headline_only": headline_names},
                   "body_feature_provenance": provenance, "body_sample_coverage": body_coverage,
                   "excluded": excluded,
                   "configuration": {"database": str(Path(args.db).resolve()),
                                     "documents": str(Path(args.documents).resolve()),
                                     "embedding_model": args.embedding_model,
                                     "nli_model": args.nli_model,
                                     "hypotheses": str(Path(args.hypotheses).resolve()),
                                     "min_detail_characters": args.min_detail_characters}})
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print(json.dumps({"rows": len(rows), "body_details": sum(p["detail_count"] for p in provenance),
                      "body_folds": len(report["body_based"]["folds"]),
                      "headline_folds": len(report["headline_only"]["folds"]),
                      "output": args.out}, indent=2))


if __name__ == "__main__":
    main()
