"""Run a retrospective, topic-group-held-out Hawkes price-move experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claim_reaction.historical_experiment import run_historical_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--link-scores", required=True)
    parser.add_argument("--topic-groups", required=True)
    parser.add_argument("--deadlines")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", required=True, action="append", type=float)
    parser.add_argument("--decay-seconds", required=True, action="append", type=float)
    parser.add_argument("--resolution", action="append", type=int,
                        help="bucket resolution in seconds; repeat to run separate resolutions")
    parser.add_argument("--evidence-updates", action="store_true",
                        help="include first later YES and NO evidence updates per cluster and market")
    args = parser.parse_args()
    result = run_historical_experiment(
        args.db, args.graph, args.link_scores, args.topic_groups,
        thresholds=args.threshold, decay_seconds=args.decay_seconds,
        deadline_overrides_path=args.deadlines,
        evidence_updates=args.evidence_updates,
        resolutions=args.resolution or (300, 1800, 10800),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target.resolve()),
                      "folds": len(result["folds"]),
                      "aggregate_metrics": len(result["aggregate_oof_metrics"]),
                      "event_studies": len(result["event_study"]["outcomes"])}, indent=2))


if __name__ == "__main__":
    main()
