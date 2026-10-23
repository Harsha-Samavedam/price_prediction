"""Topic-held-out retrospective price-move experiments.

The graph and its GDELT ``provider_seen_at`` timestamps are retrospective
features. They are not publication times or locally observed arrival times and
this experiment makes no causal claim.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import iso, parse_time
from .hawkes import (Mode, fit_from_features, history_features, score_hawkes)
from .historical_prices import (PriceBin, classify_change, event_study,
                                load_market_bins, split_contiguous_segments)

MODES: tuple[Mode, ...] = ("constant", "price_history", "timestamp_only", "signed_news")
DECAY_GRID_SECONDS = (900.0, 3600.0, 14400.0, 86400.0)
EVENT_HORIZONS = (("5m", timedelta(minutes=5)), ("1h", timedelta(hours=1)),
                  ("1d", timedelta(days=1)))


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _deadlines(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = _json(path)
    if isinstance(value, dict) and isinstance(value.get("deadlines"), dict):
        value = value["deadlines"]
    if not isinstance(value, dict):
        raise ValueError("deadline overrides must be a market-slug-to-timestamp mapping")
    return {str(slug): parse_time(ts) for slug, ts in value.items()}


def _topic_map(path: str | Path) -> dict[str, str]:
    raw = _json(path)
    groups = raw.get("groups") if isinstance(raw, dict) else None
    if not isinstance(groups, dict):
        raise ValueError("topic groups JSON must contain a groups mapping")
    result: dict[str, str] = {}
    for group, item in groups.items():
        markets = item.get("markets") if isinstance(item, dict) else None
        if not isinstance(markets, list):
            raise ValueError(f"topic group {group!r} must contain a markets list")
        for slug in markets:
            slug = str(slug)
            if slug in result:
                raise ValueError(f"market {slug!r} belongs to multiple topic groups")
            result[slug] = str(group)
    return result


def _load_graph_events(graph_path: str | Path, scores_path: str | Path,
                       deadlines: dict[str, Any], *, evidence_updates: bool = False
                       ) -> tuple[dict[str, list[tuple[Any, float]]], list[dict], dict[str, int]]:
    graph = _json(graph_path)
    claims = graph.get("claims")
    candidates = graph.get("link_candidates")
    if not isinstance(claims, list) or not isinstance(candidates, list):
        raise ValueError("graph JSON must contain claims and link_candidates arrays")
    claim_by_id = {row["claim_id"]: row for row in claims}
    if len(claim_by_id) != len(claims):
        raise ValueError("graph contains duplicate claim IDs")
    scores: dict[tuple[str, str], tuple[Any, float]] = {}
    with Path(scores_path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (str(row["claim_id"]), str(row["market_slug"]))
                scores[key] = (parse_time(row["first_seen_at"]), float(row["signed_score"]))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid link score line {line_no}") from exc
    events: dict[str, list[tuple[Any, float]]] = defaultdict(list)
    rows: list[dict] = []
    suppressed = 0
    required: set[tuple[str, str]] = set()
    first_report_ids = {claim_id for claim_id, claim in claim_by_id.items()
                        if claim.get("is_first_report", True)}
    for edge in candidates:
        claim_id, slug = str(edge["claim_id"]), str(edge["market_id"])
        if claim_id not in claim_by_id:
            raise ValueError(f"link candidate references unknown claim {claim_id!r}")
        if not evidence_updates and claim_id not in first_report_ids:
            continue
        key = (claim_id, slug)
        required.add(key)
    missing = sorted(required - set(scores))
    if missing:
        preview = ", ".join(f"{claim}/{slug}" for claim, slug in missing[:5])
        raise ValueError(f"missing link-score coverage for {len(missing)} candidate(s): {preview}")
    if evidence_updates:
        grouped: dict[tuple[str, str], list[tuple[Any, str, float, bool]]] = defaultdict(list)
        for claim_id, slug in sorted(required):
            claim = claim_by_id[claim_id]
            event_time = parse_time(claim.get("event_time") or claim["earliest_seen_at"])
            grouped[(str(claim.get("cluster_id", claim_id)), slug)].append(
                (event_time, claim_id, scores[(claim_id, slug)][1], claim_id in first_report_ids))
        for (_, slug), snapshots in grouped.items():
            snapshots.sort(key=lambda item: (item[0], item[1]))
            first_reports = [item for item in snapshots if item[3]]
            if not first_reports:
                # This market may only become a candidate after the cluster's
                # first report. Emit the first observed nonzero evidence for
                # each direction without inventing an earlier report event.
                chosen = []
                seen_directions: set[str] = set()
                for item in snapshots:
                    if item[2] == 0:
                        continue
                    direction = "yes" if item[2] > 0 else "no"
                    if direction not in seen_directions:
                        chosen.append((item, "positive_update" if direction == "yes" else "negative_update"))
                        seen_directions.add(direction)
                    else:
                        suppressed += 1
            else:
                first = first_reports[0]
                chosen = [(first, "first_report")]
                seen_directions = {"yes" if first[2] > 0 else "no"} if first[2] != 0 else set()
                for item in snapshots:
                    if (item[0], item[1]) <= (first[0], first[1]) or item[2] == 0:
                        continue
                    direction = "yes" if item[2] > 0 else "no"
                    if direction not in seen_directions:
                        chosen.append((item, "positive_update" if direction == "yes" else "negative_update"))
                        seen_directions.add(direction)
                    else:
                        suppressed += 1
            for item, kind in chosen:
                time, claim_id, mark, _ = item
                if slug not in deadlines or time < deadlines[slug]:
                    events[slug].append((time, mark))
                    rows.append({"claim_id": claim_id, "market_slug": slug, "time": time,
                                 "event_kind": kind, "signed_mark": mark})
    else:
        for claim_id, slug in sorted(required):
            _, mark = scores[(claim_id, slug)]
            claim = claim_by_id[claim_id]
            claim_time = parse_time(claim.get("event_time") or claim["earliest_seen_at"])
            if slug not in deadlines or claim_time < deadlines[slug]:
                events[slug].append((claim_time, mark))
                rows.append({"claim_id": claim_id, "market_slug": slug, "time": claim_time,
                             "event_kind": "first_report", "signed_mark": mark})
    # Only candidate links are used; unrelated scored rows are ignored.
    for slug in events:
        events[slug].sort(key=lambda item: item[0])
    return events, rows, {"repeated_reports_suppressed": suppressed}


def _num_time(value: Any) -> float:
    return value.timestamp()


def _class_counts(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(label): counts[label] for label in range(3)}


def _market_segments(bins: Iterable[PriceBin], threshold: float):
    """Yield segment bins and adjacent-change labels; first bin is baseline."""
    for segment in split_contiguous_segments(bins):
        if not segment:
            continue
        labels = np.zeros(len(segment), dtype=int)
        for i in range(1, len(segment)):
            direction = classify_change(segment[i - 1], segment[i], threshold).direction
            labels[i] = {"none": 0, "up": 1, "down": 2}[direction]
        yield segment, labels


def run_historical_experiment(db_path: str | Path, graph_path: str | Path,
                              link_scores_path: str | Path, topic_groups_path: str | Path,
                              *, thresholds: Iterable[float] | float,
                              decay_seconds: Iterable[float] | float,
                              deadline_overrides_path: str | Path | None = None,
                              evidence_updates: bool = False,
                              resolutions: Iterable[int] = (300, 1800, 10800)) -> dict[str, Any]:
    """Run topic-group held-out folds with separately constructed market histories.

    ``decay_seconds`` supplies the candidate grid. The caller should provide
    the documented grid (900, 3600, 14400, 86400 seconds) for the full run.
    """
    thresholds = ((float(thresholds),) if np.isscalar(thresholds)
                  else tuple(float(x) for x in thresholds))
    requested_decays = ((float(decay_seconds),) if np.isscalar(decay_seconds)
                        else tuple(float(x) for x in decay_seconds))
    resolutions = tuple(int(x) for x in resolutions)
    if not thresholds or any(x <= 0 for x in thresholds):
        raise ValueError("at least one positive threshold is required")
    if not requested_decays or any(x <= 0 for x in requested_decays):
        raise ValueError("at least one positive decay_seconds value is required")
    if len(set(requested_decays)) != len(requested_decays):
        raise ValueError("decay_seconds values must be unique")
    decays = tuple(requested_decays)
    if not resolutions or any(x <= 0 for x in resolutions):
        raise ValueError("at least one positive resolution is required")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must be unique")

    groups = _topic_map(topic_groups_path)
    deadlines = _deadlines(deadline_overrides_path)
    news_events, linked_claims, event_counts = _load_graph_events(
        graph_path, link_scores_path, deadlines, evidence_updates=evidence_updates)
    db = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    deadline_exclusions: dict[str, dict[str, Any]] = {}
    try:
        markets: dict[str, list[PriceBin]] = {}
        for slug in groups:
            raw_bins = load_market_bins(db, slug)
            kept = load_market_bins(db, slug, deadlines.get(slug))
            markets[slug] = kept
            if slug in deadlines:
                raw_keys = {(b.bucket_start, b.resolution_seconds) for b in raw_bins}
                kept_keys = {(b.bucket_start, b.resolution_seconds) for b in kept}
                removed = raw_keys - kept_keys
                added = kept_keys - raw_keys
                deadline_exclusions[slug] = {
                    "deadline_excluded_price_bins": len(removed),
                    "deadline_excluded_bins_by_resolution": dict(sorted(Counter(
                        str(res) for _, res in removed).items())),
                    "raw_end_price_bin_count": len(raw_bins),
                    "override_end_price_bin_count": len(kept),
                    "added_price_bin_count": len(added),
                    "removed_price_bin_count": len(removed),
                    "added_bins_by_resolution": dict(sorted(Counter(str(res) for _, res in added).items())),
                    "removed_bins_by_resolution": dict(sorted(Counter(str(res) for _, res in removed).items())),
                    "override_timestamp": iso(deadlines[slug]),
                }
    finally:
        db.close()

    output_folds: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    event_results: list[dict[str, Any]] = []
    event_window_coverage: dict[str, dict[str, int]] = {}
    for resolution in resolutions:
        selected = {slug: [b for b in bins if b.resolution_seconds == resolution]
                    for slug, bins in markets.items()}
        scoped_claims: dict[str, list[dict]] = defaultdict(list)
        scope_counts: Counter[str] = Counter()
        for claim in linked_claims:
            bins = selected.get(claim["market_slug"], [])
            if not bins:
                scope_counts["no_price_bins"] += 1
            elif claim["time"] < bins[0].bucket_end:
                scope_counts["before_first_price"] += 1
            elif claim["time"] >= bins[-1].bucket_end:
                scope_counts["after_last_price"] += 1
            else:
                scoped_claims[claim["market_slug"]].append(claim)
                scope_counts["within_price_window"] += 1
        event_window_coverage[str(resolution)] = dict(sorted(scope_counts.items()))
        for threshold in thresholds:
            # Each feature set is constructed within its market/segment; the
            # outer held-out group is never consulted during decay selection.
            feature_rows: dict[float, dict[str, dict[str, list[tuple[np.ndarray, int]]]]] = {
                decay: {mode: defaultdict(list) for mode in MODES} for decay in decays}
            mode_names: dict[str, tuple[str, ...]] = {}
            inner_model_cache: dict[tuple[tuple[str, ...], str, float], Any] = {}
            for decay in decays:
                if decay != decays[0]:
                    feature_rows[decay]["constant"] = feature_rows[decays[0]]["constant"]
                for slug, bins in selected.items():
                    for segment, labels in _market_segments(bins, threshold):
                        starts = np.array([_num_time(b.bucket_start) for b in segment])
                        market_events = news_events.get(slug, [])
                        nt = np.array([_num_time(t) for t, _ in market_events], dtype=float)
                        marks = np.array([m for _, m in market_events], dtype=float)
                        for mode in MODES:
                            if mode == "constant" and decay != decays[0]:
                                continue
                            x, names = history_features(starts, nt, marks, labels,
                                                        resolution, decay, mode)
                            mode_names[mode] = names
                            # First bin initializes history and has no adjacent
                            # price-change outcome, so it is never a fitted/scored row.
                            for i in range(1, len(segment)):
                                feature_rows[decay][mode][groups[slug]].append((x[i], int(labels[i])))

            active_groups = [g for g in sorted(set(groups.values()))
                             if any(feature_rows[decays[0]][mode].get(g)
                                    for mode in MODES)]
            for held_group in active_groups:
                train_groups = [g for g in active_groups if g != held_group]
                for mode in MODES:
                    inner_scores = []
                    for decay in decays:
                        if mode == "constant" and inner_scores:
                            # Constant-mode features do not depend on decay.
                            first = inner_scores[0]
                            inner_scores.append({**first, "decay_seconds": decay})
                            continue
                        losses = []
                        # Inner validation is leave-one-topic-group-out over
                        # the outer training groups only.
                        for inner_group in train_groups:
                            inner_train = [row for group in train_groups if group != inner_group
                                           for row in feature_rows[decay][mode].get(group, [])]
                            inner_test = feature_rows[decay][mode].get(inner_group, [])
                            if not inner_train or not inner_test:
                                continue
                            xi = np.vstack([x for x, _ in inner_train])
                            yi = np.array([y for _, y in inner_train])
                            xv = np.vstack([x for x, _ in inner_test])
                            yv = np.array([y for _, y in inner_test])
                            train_group_key = tuple(g for g in train_groups if g != inner_group)
                            cache_key = (train_group_key, mode, decay)
                            model = inner_model_cache.get(cache_key)
                            if model is None:
                                model = fit_from_features(xi, yi, resolution, mode,
                                                          decay, mode_names[mode])
                                inner_model_cache[cache_key] = model
                            score = score_hawkes(model, xv, yv, resolution)
                            losses.append((score.log_loss, len(inner_test)))
                        total_inner = sum(n for _, n in losses)
                        inner_loss = (sum(loss * n for loss, n in losses) / total_inner
                                      if total_inner else None)
                        inner_scores.append({"decay_seconds": decay,
                                             "inner_jump_log_loss": inner_loss,
                                             "inner_held_out_bin_count": total_inner})
                    valid_inner = [s for s in inner_scores if s["inner_jump_log_loss"] is not None]
                    chosen_decay = (min(valid_inner,
                                        key=lambda s: (s["inner_jump_log_loss"], s["decay_seconds"])
                                        )["decay_seconds"] if valid_inner else min(decays))
                    train = [row for group in train_groups
                             for row in feature_rows[chosen_decay][mode].get(group, [])]
                    test = feature_rows[chosen_decay][mode].get(held_group, [])
                    record: dict[str, Any] = {
                        "resolution_seconds": resolution, "threshold": threshold,
                        "decay_seconds": chosen_decay, "held_out_topic_group": held_group,
                        "mode": mode, "train_bin_count": len(train), "held_out_bin_count": len(test),
                        "train_class_counts": _class_counts(y for _, y in train),
                        "observed_class_counts": _class_counts(y for _, y in test),
                        "inner_decay_scores": inner_scores,
                        "model_jump_log_loss": None, "model_jump_multiclass_brier": None,
                    }
                    if train and test:
                        x_train = np.vstack([x for x, _ in train])
                        y_train = np.array([y for _, y in train])
                        x_test = np.vstack([x for x, _ in test])
                        y_test = np.array([y for _, y in test])
                        model = fit_from_features(x_train, y_train, resolution, mode,
                                                  chosen_decay, mode_names[mode])
                        score = score_hawkes(model, x_test, y_test, resolution)
                        record["model_jump_log_loss"] = score.log_loss
                        record["model_jump_multiclass_brier"] = score.brier
                    output_folds.append(record)

            for mode in MODES:
                scored = [f for f in output_folds
                          if f["resolution_seconds"] == resolution and f["threshold"] == threshold
                          and f["mode"] == mode and f["held_out_bin_count"] > 0
                          and f["model_jump_log_loss"] is not None]
                total = sum(f["held_out_bin_count"] for f in scored)
                aggregate.append({
                    "resolution_seconds": resolution, "threshold": threshold,
                    "mode": mode, "held_out_bin_count": total,
                    "model_jump_log_loss": (sum(f["model_jump_log_loss"] * f["held_out_bin_count"] for f in scored) / total if total else None),
                    "model_jump_multiclass_brier": (sum(f["model_jump_multiclass_brier"] * f["held_out_bin_count"] for f in scored) / total if total else None),
                    "observed_class_counts": _class_counts(y for f in scored
                                                           for label, count in f["observed_class_counts"].items()
                                                           for y in [int(label)] * count),
                })

            # Descriptive cumulative post-news event study, distinct from
            # adjacent-bin model jump labels.
            for slug, bins in selected.items():
                for claim in scoped_claims[slug]:
                    for horizon_name, horizon in EVENT_HORIZONS:
                        outcome = event_study(bins, claim["time"], threshold=threshold,
                                              horizon=horizon,
                                              max_pre_event_age=timedelta(minutes=30))
                        event_results.append({
                            "resolution_seconds": resolution, "threshold": threshold,
                            "market_slug": slug, "claim_id": claim["claim_id"],
                            "event_kind": claim["event_kind"], "signed_mark": claim["signed_mark"],
                            "claim_time": iso(claim["time"]),
                            "horizon": horizon_name, "status": outcome.status,
                            "censor_reason": outcome.reason if outcome.status == "censored" else None,
                            "direction": outcome.direction,
                            "baseline_bucket_start": iso(outcome.baseline.bucket_start) if outcome.baseline else None,
                            "baseline_bucket_end": iso(outcome.baseline.bucket_end) if outcome.baseline else None,
                            "baseline_price": outcome.baseline.price if outcome.baseline else None,
                            "crossing_bucket_start": iso(outcome.interval_start),
                            "crossing_bucket_end": iso(outcome.interval_end),
                            "crossing_delta": outcome.delta,
                            "latency_lower_bound_seconds": ((outcome.interval_start - claim["time"]).total_seconds()
                                                             if outcome.interval_start else None),
                            "latency_upper_bound_seconds": ((outcome.interval_end - claim["time"]).total_seconds()
                                                             if outcome.interval_end else None),
                        })

    event_summary: list[dict[str, Any]] = []
    event_keys = sorted({(r["resolution_seconds"], r["threshold"], r["horizon"])
                         for r in event_results})
    for resolution, threshold, horizon in event_keys:
        rows = [r for r in event_results if r["resolution_seconds"] == resolution
                and r["threshold"] == threshold and r["horizon"] == horizon]
        if evidence_updates:
            rows = [r for r in rows if r["signed_mark"] != 0]
        censored = Counter(r["censor_reason"] for r in rows if r["status"] == "censored")
        observed_lags = sorted(r["latency_upper_bound_seconds"] for r in rows
                               if r["latency_upper_bound_seconds"] is not None)
        mid = len(observed_lags) // 2
        median_lag = (None if not observed_lags else observed_lags[mid] if len(observed_lags) % 2
                      else (observed_lags[mid - 1] + observed_lags[mid]) / 2)
        signed_rows = [r for r in rows if r["signed_mark"] != 0]
        aligned = [r for r in signed_rows if r["status"] == "crossed" and
                   ((r["signed_mark"] > 0 and r["direction"] == "up")
                    or (r["signed_mark"] < 0 and r["direction"] == "down"))]
        opposite = [r for r in signed_rows if r["status"] == "crossed" and
                    ((r["signed_mark"] > 0 and r["direction"] == "down")
                     or (r["signed_mark"] < 0 and r["direction"] == "up"))]

        def median_upper_lag(items):
            lags = sorted(r["latency_upper_bound_seconds"] for r in items
                          if r["status"] == "crossed" and r["latency_upper_bound_seconds"] is not None)
            if not lags:
                return None
            m = len(lags) // 2
            return lags[m] if len(lags) % 2 else (lags[m - 1] + lags[m]) / 2

        event_summary.append({
            "resolution_seconds": resolution, "threshold": threshold, "horizon": horizon,
            "total": len(rows),
            "crossed_up": sum(r["status"] == "crossed" and r["direction"] == "up" for r in rows),
            "crossed_down": sum(r["status"] == "crossed" and r["direction"] == "down" for r in rows),
            "censored_by_reason": dict(sorted(censored.items())),
            "median_observed_crossing_upper_bound_lag_seconds": median_lag,
            "aligned_crossed": sum(r["status"] == "crossed" for r in aligned),
            "aligned_denominator": len(signed_rows),
            "median_aligned_crossing_upper_bound_lag_seconds": median_upper_lag(aligned),
            "opposite_crossed": sum(r["status"] == "crossed" for r in opposite),
            "opposite_denominator": len(signed_rows),
            "median_opposite_crossing_upper_bound_lag_seconds": median_upper_lag(opposite),
            "no_threshold_crossing_censored": sum(r["status"] == "censored" for r in signed_rows),
        })

    return {
        "provenance": {
            "event_time": "GDELT provider_seen_at / graph earliest_seen_at",
            "caveat": "Retrospective provider_seen_at, not publication time or local first observation; no causal claim.",
            "feature_availability": "price bucket labels only after bucket end; news events strictly before bucket start",
            "model_jump_metrics": "adjacent price-bucket threshold labels; excludes each contiguous segment's first baseline bucket",
            "event_study_metrics": "first cumulative post-news threshold crossing from pre-event quote; descriptive only",
        },
        "deadline_exclusions": deadline_exclusions,
        "folds": output_folds,
        "aggregate_oof_metrics": aggregate,
        "event_study": {
            "evidence_updates_enabled": evidence_updates,
            "event_counts": {
                "zero_first_reports": sum(r["event_kind"] == "first_report" and r["signed_mark"] == 0 for r in linked_claims),
                "positive_updates": sum(r["event_kind"] == "positive_update" for r in linked_claims),
                "negative_updates": sum(r["event_kind"] == "negative_update" for r in linked_claims),
                **event_counts,
            },
            "max_pre_event_age_seconds": 1800,
            "price_window_coverage": event_window_coverage,
            "horizons": [name for name, _ in EVENT_HORIZONS],
            "outcomes": event_results,
            "summary": event_summary,
            "censoring_counts": dict(sorted(Counter(
                f"{row['resolution_seconds']}:{row['threshold']}:{row['horizon']}:{row['censor_reason']}"
                for row in event_results if row["status"] == "censored").items())),
        },
    }
