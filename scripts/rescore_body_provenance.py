"""Reaggregate stored body NLI probabilities with the entailment-only scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claim_reaction.body_forecast import signed_entailment_score


def rescore_experiment(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with fetched body provenance reaggregated from stored NLI."""
    result = dict(data)
    provenance = data.get("body_feature_provenance")
    if not isinstance(provenance, list):
        raise ValueError("body_feature_provenance must be a list")

    rescored: list[dict[str, Any]] = []
    for row in provenance:
        if not isinstance(row, dict):
            raise ValueError("body_feature_provenance entries must be objects")
        updated = dict(row)
        if updated.get("status") == "fetched":
            sentences = updated.get("selected_sentences")
            if not isinstance(sentences, list):
                raise ValueError("fetched provenance must contain selected_sentences")
            scores: list[float] = []
            new_sentences: list[dict[str, Any]] = []
            for sentence in sentences:
                if not isinstance(sentence, dict) or not isinstance(sentence.get("probabilities"), dict):
                    raise ValueError("selected sentences must contain raw probabilities")
                rescored_sentence = dict(sentence)
                score = signed_entailment_score(sentence["probabilities"])
                rescored_sentence["signed_score"] = score
                scores.append(score)
                new_sentences.append(rescored_sentence)
            updated["selected_sentences"] = new_sentences
            updated["signed_detail_scores"] = scores
            updated["aggregate_signed_score"] = sum(scores) / len(scores) if scores else 0.0
            updated["scorer_version"] = "entailment_v2"
            updated["retrospective"] = True
        rescored.append(updated)

    result["body_feature_provenance"] = rescored
    result["scorer_version"] = "entailment_v2"
    result["retrospective"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rescored = rescore_experiment(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescored, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
