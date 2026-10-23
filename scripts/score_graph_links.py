"""Score retrospective claim-to-market candidate links with local NLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claim_reaction.body_forecast import (LocalNLIClassifier, _score_pair,
                                          signed_entailment_score)
from scripts.score_historical_headlines import SCORER_VERSION, score_key as headline_score_key


def _key(model: str, title: str, market_slug: str, yes: str, no: str) -> str:
    return hashlib.sha256(json.dumps([SCORER_VERSION, model, title, market_slug, yes, no],
                                   ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _legacy_key(model: str, title: str, market_slug: str, yes: str, no: str) -> str:
    return hashlib.sha256(json.dumps([model, title, market_slug, yes, no],
                                   ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _legacy_headline_score_key(row: dict[str, Any], model: str,
                               hypotheses: dict[str, Any]) -> str:
    payload = [model, hypotheses["yes"], hypotheses["no"], row["source"],
               row["source_id"], row["market_slug"], row["url"], row["title"],
               row["provider_seen_at"]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _prior_scores(path: str | None, model: str, hypotheses: dict[str, Any]
                  ) -> dict[str, tuple[float, dict[str, Any]]]:
    cached: dict[str, tuple[float, dict[str, Any]]] = {}
    if not path:
        return cached
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            hyp = hypotheses.get(row.get("market_slug"))
            if row.get("model_id") != model or not isinstance(hyp, dict):
                continue
            version = row.get("scorer_version")
            if version not in (None, SCORER_VERSION):
                continue
            expected_key = (headline_score_key(row, model, hyp)
                            if version == SCORER_VERSION
                            else _legacy_headline_score_key(row, model, hyp))
            if row.get("score_key") != expected_key:
                continue
            probabilities = row.get("probabilities")
            if not isinstance(probabilities, dict):
                continue
            score = (float(row["signed_score"]) if version == SCORER_VERSION
                     else signed_entailment_score(probabilities))
            key = _key(model, row["title"], row["market_slug"], hyp["yes"], hyp["no"])
            cached[key] = (score, probabilities)
    return cached


def _prior_link_scores(paths: list[str], model: str
                      ) -> dict[str, tuple[float, dict[str, Any]]]:
    cached: dict[str, tuple[float, dict[str, Any]]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("model_id") != model:
                    continue
                yes, no = row.get("yes_hypothesis"), row.get("no_hypothesis")
                if not isinstance(yes, str) or not isinstance(no, str):
                    continue
                version = row.get("scorer_version")
                if version not in (None, SCORER_VERSION):
                    continue
                key = _key(model, row["headline"], row["market_slug"], yes, no)
                expected_key = key if version == SCORER_VERSION else _legacy_key(
                    model, row["headline"], row["market_slug"], yes, no)
                probabilities = row.get("probabilities")
                if row.get("input_key") == expected_key and isinstance(probabilities, dict):
                    score = (float(row["signed_score"]) if version == SCORER_VERSION
                             else signed_entailment_score(probabilities))
                    cached[key] = (score, probabilities)
    return cached


def score_links(graph_path: str, hypotheses_path: str, output_path: str,
                classifier: Any, prior_scores_path: str | None = None,
                prior_link_scores_paths: list[str] | None = None,
                all_snapshots: bool = False) -> dict[str, Any]:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    hypotheses = json.loads(Path(hypotheses_path).read_text(encoding="utf-8"))
    if not isinstance(hypotheses, dict):
        raise ValueError("Hypotheses must be a market-slug mapping")
    claims = {row["claim_id"]: row for row in graph["claims"]
              if all_snapshots or row["is_first_report"]}
    candidates = [edge for edge in graph["link_candidates"]
                  if edge["claim_id"] in claims]
    model = str(classifier.model_id)
    headline_memo = _prior_scores(prior_scores_path, model, hypotheses)
    link_memo = _prior_link_scores(prior_link_scores_paths or [], model)
    memo = {**headline_memo, **link_memo}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[tuple[str, str, str], dict[str, Any]] = {}
    if output.exists():
        with output.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                        if row.get("scorer_version") == SCORER_VERSION:
                            completed[(row["claim_id"], row["market_slug"], row["model_id"])] = row
                        elif row.get("scorer_version") is None and row.get("model_id") == model:
                            yes, no = row.get("yes_hypothesis"), row.get("no_hypothesis")
                            probabilities = row.get("probabilities")
                            if (isinstance(yes, str) and isinstance(no, str)
                                    and isinstance(probabilities, dict)
                                    and row.get("input_key") == _legacy_key(
                                        model, row["headline"], row["market_slug"], yes, no)):
                                key = _key(model, row["headline"], row["market_slug"], yes, no)
                                score = signed_entailment_score(probabilities)
                                memo[key] = (score, probabilities)
                                link_memo[key] = (score, probabilities)
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"Invalid link cache line {line_number}") from exc
    new, reused_headlines, reused_links, skipped = 0, 0, 0, 0
    with output.open("a", encoding="utf-8", buffering=1) as target:
        for edge in candidates:
            claim_id, slug = edge["claim_id"], edge["market_id"]
            claim = claims[claim_id]
            hyp = hypotheses.get(slug)
            if not isinstance(hyp, dict) or not all(
                    isinstance(hyp.get(side), str) and hyp[side].strip()
                    for side in ("yes", "no")):
                raise ValueError(f"Missing YES/NO hypotheses for {slug}")
            title = (claim.get("event_title", claim["representative_title"])
                     if all_snapshots else claim["representative_title"])
            key = _key(model, title, slug, hyp["yes"], hyp["no"])
            old = completed.get((claim_id, slug, model))
            if (old is not None and old.get("input_key") == key
                    and old.get("first_seen_at") == claim["event_time"]
                    and old.get("cosine_similarity") == edge["cosine_similarity"]
                    and old.get("retrieved_member") == edge["retrieved_member"]):
                skipped += 1
                continue
            if key in memo:
                score, probabilities = memo[key]
                if key in link_memo:
                    reused_links += 1
                else:
                    reused_headlines += 1
            else:
                score, probabilities = _score_pair(classifier, title, hyp["yes"], hyp["no"])
                memo[key] = (score, probabilities)
                new += 1
            record = {"claim_id": claim_id, "cluster_id": claim["cluster_id"],
                      "market_slug": slug,
                      "event_time": claim["event_time"],
                      "first_seen_at": claim["event_time"],
                      "earliest_seen_at": claim.get("earliest_seen_at", claim["event_time"]),
                      "utc_day": claim.get("utc_day"),
                      "members": claim.get("members", []),
                      "headline": title,
                      "cosine_similarity": edge["cosine_similarity"],
                      "retrieved_member": edge["retrieved_member"],
                      "signed_score": score, "probabilities": probabilities,
                      "model_id": model, "scorer_version": SCORER_VERSION, "input_key": key,
                      "yes_hypothesis": hyp["yes"], "no_hypothesis": hyp["no"],
                      "retrospectively_collected": True,
                      "time_basis": "GDELT provider_seen_at"}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed[(claim_id, slug, model)] = record
    return {"candidate_links": len(candidates),
            "new_nli_scores": new, "reused_headline_scores": reused_headlines,
            "reused_prior_link_scores": reused_links,
            "cached_links": skipped, "output": str(output.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--hypotheses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-headline-scores")
    parser.add_argument("--prior-link-scores", action="append", default=[])
    parser.add_argument("--all-snapshots", action="store_true",
                        help="score candidate links for every claim snapshot")
    parser.add_argument("--model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    args = parser.parse_args()
    print(json.dumps(score_links(args.graph, args.hypotheses, args.output,
                                 LocalNLIClassifier(args.model),
                                 args.prior_headline_scores,
                                 args.prior_link_scores,
                                 args.all_snapshots), indent=2))


if __name__ == "__main__":
    main()
