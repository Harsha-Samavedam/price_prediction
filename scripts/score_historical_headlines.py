"""Cache local NLI scores for retrospectively indexed GDELT headlines."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claim_reaction.body_forecast import LocalNLIClassifier, _score_pair, signed_entailment_score

SCORER_VERSION = "entailment_v2"


def headline_rows(db_path: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(
            "SELECT source,source_id,market_slug,url,title,provider_seen_at,"
            "published_at,first_collected_at FROM news "
            "WHERE source='gdelt-doc' AND provider_seen_at IS NOT NULL "
            "AND title!='' AND url!='' "
            "ORDER BY market_slug,provider_seen_at,url"
        )]
    finally:
        connection.close()


def score_key(row: dict[str, Any], model_id: str, hypotheses: dict[str, str]) -> str:
    payload = [SCORER_VERSION, model_id, hypotheses["yes"], hypotheses["no"],
               row["source"], row["source_id"], row["market_slug"],
               row["url"], row["title"], row["provider_seen_at"]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _legacy_score_key(row: dict[str, Any], model_id: str,
                      hypotheses: dict[str, str]) -> str:
    payload = [model_id, hypotheses["yes"], hypotheses["no"], row["source"],
               row["source_id"], row["market_slug"], row["url"], row["title"],
               row["provider_seen_at"]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def score_headlines(db_path: str, hypotheses_path: str, output_path: str,
                    classifier: Any,
                    prior_headline_scores_paths: list[str] | None = None) -> dict[str, Any]:
    hypotheses = json.loads(Path(hypotheses_path).read_text(encoding="utf-8"))
    if not isinstance(hypotheses, dict):
        raise ValueError("Hypotheses must be a market-slug mapping")
    rows = headline_rows(db_path)
    model_id = str(classifier.model_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    legacy: dict[str, dict[str, Any]] = {}
    v2_lines: list[str] = []
    if output.exists():
        with output.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        cached = json.loads(line)
                        if cached.get("scorer_version") == SCORER_VERSION:
                            existing.add(str(cached["score_key"]))
                            v2_lines.append(line.rstrip("\n"))
                        elif cached.get("scorer_version") is None:
                            legacy[str(cached["score_key"])] = cached
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"Invalid score cache line {line_number}") from exc
    for prior_path in prior_headline_scores_paths or []:
        with Path(prior_path).open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                cached = json.loads(line)
                if cached.get("scorer_version") is None:
                    legacy[str(cached.get("score_key", ""))] = cached
    if output.exists() and len(v2_lines) != sum(bool(line.strip()) for line in output.read_text(
            encoding="utf-8").splitlines()):
        output.write_text("".join(line + "\n" for line in v2_lines), encoding="utf-8")
    scored = 0
    skipped = 0
    with output.open("a", encoding="utf-8", buffering=1) as target:
        for row in rows:
            market_hypotheses = hypotheses.get(row["market_slug"])
            if (not isinstance(market_hypotheses, dict)
                    or not all(isinstance(market_hypotheses.get(k), str)
                               and market_hypotheses[k].strip() for k in ("yes", "no"))):
                raise ValueError(f"Missing YES/NO hypotheses for {row['market_slug']}")
            key = score_key(row, model_id, market_hypotheses)
            if key in existing:
                skipped += 1
                continue
            old_key = _legacy_score_key(row, model_id, market_hypotheses)
            old = legacy.get(old_key)
            if (old is not None and old.get("model_id") == model_id
                    and isinstance(old.get("probabilities"), dict)):
                probabilities = old["probabilities"]
                score = signed_entailment_score(probabilities)
            else:
                score, probabilities = _score_pair(
                    classifier, row["title"], market_hypotheses["yes"],
                    market_hypotheses["no"])
            record = {**row, "score_key": key, "model_id": model_id,
                      "scorer_version": SCORER_VERSION,
                      "signed_score": score, "probabilities": probabilities,
                      "time_basis": "GDELT provider_seen_at",
                      "retrospectively_collected": True}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.add(key)
            scored += 1
    return {"total_rows": len(rows), "new_scores": scored, "cached_scores": skipped,
            "output": str(output.resolve()), "model_id": model_id,
            "time_basis": "retrospective GDELT provider_seen_at"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--hypotheses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-headline-scores", action="append", default=[])
    parser.add_argument("--model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    args = parser.parse_args()
    result = score_headlines(args.db, args.hypotheses, args.output,
                             LocalNLIClassifier(args.model),
                             args.prior_headline_scores)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
