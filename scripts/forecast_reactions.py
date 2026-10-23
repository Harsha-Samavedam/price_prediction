"""Run the offline GDELT headline YES-price reaction evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# Support both `python -m scripts.forecast_reactions` and direct script launch
# from any working directory without requiring a package installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from claim_reaction.forecast import (HORIZONS_MINUTES, MAX_PRICE_AGE_MINUTES,
                                    RIDGE_ALPHAS, add_relevance_features,
                                    evaluate_leave_one_market_out,
                                    load_forecast_rows)
from claim_reaction.retrieval import HuggingFaceEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/political_25.sqlite")
    parser.add_argument("--out", default="forecast_results.json")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()
    rows, excluded = load_forecast_rows(args.db)
    embedder = HuggingFaceEmbedder(args.embedding_model)
    features, names = add_relevance_features(rows, embedder)
    report = evaluate_leave_one_market_out(rows, features)
    db_path = Path(args.db).resolve()
    db_hash = hashlib.sha256()
    with db_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            db_hash.update(chunk)
    report.update({"feature_names": names, "rows": len(rows), "excluded": excluded,
                   "event_source": "GDELT indexed headlines at provider_seen_at",
                   "label_source": "Polymarket YES-share price history",
                   "label_limitation": "Retrospective GDELT observations are provisional event times, not live news arrival times.",
                   "configuration": {"database": str(db_path),
                                     "database_sha256": db_hash.hexdigest(),
                                     "embedding_model": args.embedding_model,
                                     "horizons_minutes": list(HORIZONS_MINUTES),
                                     "max_price_age_minutes": MAX_PRICE_AGE_MINUTES,
                                     "ridge_alphas": list(RIDGE_ALPHAS)}})
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "excluded": excluded,
                      "folds": len(report["folds"]), "output": args.out}, indent=2))


if __name__ == "__main__":
    main()
