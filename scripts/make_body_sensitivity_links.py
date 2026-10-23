"""Build a retrospective body-score sensitivity copy of headline link scores.

This substitutes fetched article-body scores only where the first-report claim
contains that article URL for the candidate market. Body fetches happened later,
so these rows are explicitly labeled retrospective and are not historical features.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _member_pairs(graph: dict[str, Any]) -> dict[str, set[str]]:
    """Map (conceptually) each first-report claim to its eligible article URLs."""
    result: dict[str, set[str]] = {}
    for claim in graph.get("claims", []):
        if not claim.get("is_first_report"):
            continue
        urls: set[str] = set()
        for member in claim.get("members", []):
            url = member.get("url")
            if not isinstance(url, str) or not url:
                continue
            market_ids = member.get("market_ids", [])
            for slug in market_ids if isinstance(market_ids, list) else []:
                if isinstance(slug, str):
                    urls.add(f"{slug}\0{url}")
        result[str(claim["claim_id"])] = urls
    return result


def _snapshot_pairs(graph: dict[str, Any]) -> dict[str, set[str]]:
    """Map every scored snapshot claim to its own event article URL by market."""
    result: dict[str, set[str]] = {}
    for claim in graph.get("claims", []):
        claim_id = claim.get("claim_id")
        url = claim.get("event_article_url")
        if not isinstance(claim_id, str) or not isinstance(url, str) or not url:
            continue
        pairs: set[str] = set()
        for member in claim.get("members", []):
            if member.get("url") != url:
                continue
            market_ids = member.get("market_ids", [])
            for slug in market_ids if isinstance(market_ids, list) else []:
                if isinstance(slug, str):
                    pairs.add(f"{slug}\0{url}")
        result[claim_id] = pairs
    return result


def _body_scores(experiment: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in experiment.get("body_feature_provenance", []):
        if not isinstance(row, dict) or row.get("status") != "fetched":
            continue
        score = row.get("aggregate_signed_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        url, slug = row.get("article_url"), row.get("market_slug")
        if isinstance(url, str) and url and isinstance(slug, str) and slug:
            result[(slug, url)] = row
    return result


def make_body_sensitivity_links(graph_path: str | Path,
                                headline_scores_path: str | Path,
                                body_experiment_path: str | Path,
                                output_path: str | Path,
                                counts_path: str | Path,
                                all_snapshots: bool = False) -> dict[str, Any]:
    """Write a complete transformed JSONL and a JSON summary of substitutions."""
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    experiment = json.loads(Path(body_experiment_path).read_text(encoding="utf-8"))
    member_urls = _snapshot_pairs(graph) if all_snapshots else _member_pairs(graph)
    candidates = {(str(edge["claim_id"]), str(edge["market_id"]))
                  for edge in graph.get("link_candidates", [])
                  if (all_snapshots or str(edge.get("claim_id")) in member_urls)}
    bodies = _body_scores(experiment)

    output = Path(output_path)
    counts_file = Path(counts_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {
        "label": "retrospective_body_sensitivity",
        "retrospective": True,
        "time_basis": ("each snapshot event retained; its article body fetched retrospectively"
                       if all_snapshots else
                       "first-report chronology retained; article bodies fetched retrospectively"),
        "headline_links": 0,
        "candidate_links": len(candidates),
        "substituted_links": 0,
        "without_matching_first_report_body": 0,
        "non_candidate_links": 0,
        "output": str(output.resolve()),
    }
    with Path(headline_scores_path).open(encoding="utf-8") as source, output.open(
            "w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid headline score JSONL line {line_number}") from exc
            counts["headline_links"] += 1
            key = (str(row.get("claim_id")), str(row.get("market_slug")))
            if key not in candidates:
                counts["non_candidate_links"] += 1
            else:
                matched = None
                for pair in member_urls.get(key[0], set()):
                    slug, url = pair.split("\0", 1)
                    if slug == key[1] and (slug, url) in bodies:
                        matched = bodies[(slug, url)]
                        break
                if matched is None:
                    counts["without_matching_first_report_body"] += 1
                else:
                    prior = row.get("signed_score")
                    row["headline_signed_score"] = prior
                    row["signed_score"] = matched["aggregate_signed_score"]
                    row["body_score_substituted"] = True
                    row["body_article_url"] = matched["article_url"]
                    row["body_fetched_at"] = matched.get("body_fetched_at")
                    row["body_retrospective"] = True
                    row["retrospectively_collected"] = True
                    row["sensitivity_time_basis"] = (
                        ("snapshot event time retained; body score fetched after event"
                         if all_snapshots else
                         "first-report event time retained; body score fetched after event"))
                    counts["substituted_links"] += 1
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    counts_file.parent.mkdir(parents=True, exist_ok=True)
    counts["counts_output"] = str(counts_file.resolve())
    counts_file.write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--headline-scores", required=True)
    parser.add_argument("--body-experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--all-snapshots", action="store_true",
                        help="match each claim only to its own event_article_url")
    args = parser.parse_args()
    print(json.dumps(make_body_sensitivity_links(args.graph, args.headline_scores,
                                                  args.body_experiment, args.output,
                                                  args.counts, args.all_snapshots), indent=2))


if __name__ == "__main__":
    main()
