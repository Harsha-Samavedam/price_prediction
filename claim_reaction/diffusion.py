"""Receipt-time event study for RSS versions and YES midpoint observations."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def analyze(db_path: str, feed_markets: dict[str, list[str]], horizons: list[int] | None = None,
            thresholds: list[float] | None = None) -> dict[str, Any]:
    """Analyze first non-bootstrap RSS observation per item against configured markets.

    Event time and quote time use local receipt timestamps. interval_start and
    published_at remain in the evidence as uncertainty/context, not substitutes for t0.
    """
    horizons = sorted(set(horizons if horizons is not None else [60, 300, 900]))
    thresholds = sorted(set(thresholds if thresholds is not None else [0.25, 0.5, 0.75]))
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    events: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    try:
        versions = conn.execute("""SELECT * FROM observation_feed_versions
            ORDER BY feed_url,item_key,observed_at,version_id""").fetchall()
        seen_items: set[tuple[str, str]] = set()
        for row in versions:
            key = (row['feed_url'], row['item_key'])
            if row['feed_url'] not in feed_markets:
                continue
            if key in seen_items:
                skipped.append({'version_id': row['version_id'], 'reason': 'duplicate_or_revision'})
                continue
            seen_items.add(key)
            if row['bootstrap']:
                skipped.append({'version_id': row['version_id'], 'reason': 'bootstrap'})
                continue
            t0 = _dt(row['observed_at'])
            if t0 is None:
                skipped.append({'version_id': row['version_id'], 'reason': 'invalid_observed_at'})
                continue
            for slug in feed_markets[row['feed_url']]:
                events.append({'version_id': row['version_id'], 'item_key': row['item_key'],
                    'feed_url': row['feed_url'], 'market_slug': slug, 't0': t0,
                    'published_at': row['published_at'], 'interval_start': row['interval_start'],
                    'title': row['title']})

        output = []
        for event in events:
            t0 = event.pop('t0')
            sess = conn.execute("""SELECT s.session_id,s.started_at,s.ended_at,s.status FROM observation_market_sessions s
                WHERE s.status='connected' AND s.started_at<=? AND (s.ended_at IS NULL OR s.ended_at>=?)
                ORDER BY s.started_at DESC LIMIT 1""", (t0.isoformat().replace('+00:00','Z'), t0.isoformat().replace('+00:00','Z'))).fetchone()
            result = {**event, 'observed_at': t0.isoformat().replace('+00:00','Z'), 'session_id': sess['session_id'] if sess else None,
                      'baseline': None, 'horizons': {}, 'threshold_crossings': {}, 'skip_reason': None}
            if not sess:
                result['skip_reason'] = 'no_connected_session_at_event'
                output.append(result); continue
            quotes = conn.execute("""SELECT e.event_id,e.received_at,e.midpoint,e.bid,e.ask,e.event_type,
                f.session_id FROM observation_market_events e JOIN observation_market_frames f ON f.frame_id=e.frame_id
                WHERE f.session_id=? AND e.market_slug=? AND e.midpoint IS NOT NULL
                  AND e.bid IS NOT NULL AND e.ask IS NOT NULL AND e.bid<=e.ask AND e.error IS NULL
                ORDER BY e.received_at,e.event_id""", (sess['session_id'], event['market_slug'])).fetchall()
            qt = [(_dt(q['received_at']), q) for q in quotes]
            prior = [(d,q) for d,q in qt if d and d <= t0]
            if not prior:
                result['skip_reason'] = 'no_pre_event_quote'
                output.append(result); continue
            base_dt, base = prior[-1]
            result['baseline'] = {'midpoint': base['midpoint'], 'received_at': base['received_at'],
                                  'event_id': base['event_id'], 'bid':base['bid'],'ask':base['ask']}
            for seconds in horizons:
                target = t0 + timedelta(seconds=seconds)
                eligible = [(d,q) for d,q in qt if d and t0 < d <= target]
                last = eligible[-1] if eligible else None
                covered = bool(sess['ended_at'] is None or (_dt(sess['ended_at']) and _dt(sess['ended_at']) >= target))
                q = last[1] if last else None
                result['horizons'][str(seconds)] = {'covered': covered, 'midpoint': q['midpoint'] if q else None,
                    'delta': (q['midpoint']-base['midpoint']) if q else None,
                    'received_at': q['received_at'] if q else None, 'event_id': q['event_id'] if q else None,
                    'quote_count': len(eligible), 'incomplete_reason': None if covered else 'session_ended_before_horizon'}
            for threshold in thresholds:
                crossing = next(((d,q) for d,q in qt if d and d > t0 and q['midpoint'] >= threshold), None)
                result['threshold_crossings'][str(threshold)] = None if not crossing else {
                    'seconds_after_event': (crossing[0]-t0).total_seconds(), 'received_at': crossing[1]['received_at'],
                    'midpoint': crossing[1]['midpoint'], 'event_id': crossing[1]['event_id']}
            output.append(result)
        return {'time_basis':'local_receipt_time','causal_claim':False,'horizons_seconds':horizons,
            'absolute_midpoint_thresholds':thresholds,'aggregate':{'candidate_events':len(events),
            'analyzed':sum(x['baseline'] is not None for x in output), 'skipped_versions':len(skipped),
            'events_without_baseline':sum(x['baseline'] is None for x in output),
            'incomplete_horizon_observations':sum(not h['covered'] for x in output for h in x['horizons'].values())},
            'skipped_versions':skipped,'events':output}
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='SQLite observation database')
    parser.add_argument('--mapping', required=True, help='JSON object: feed URL to list of market slugs')
    parser.add_argument('--horizons', default='60,300,900', help='comma-separated seconds')
    parser.add_argument('--thresholds', default='0.25,0.5,0.75', help='comma-separated absolute YES midpoint levels')
    parser.add_argument('--output', help='write JSON to this path; defaults to stdout')
    args = parser.parse_args()
    mapping = json.loads(Path(args.mapping).read_text())
    result = analyze(args.db, mapping, [int(x) for x in args.horizons.split(',') if x],
                     [float(x) for x in args.thresholds.split(',') if x])
    rendered = json.dumps(result, indent=2)
    if args.output: Path(args.output).write_text(rendered + '\n')
    else: print(rendered)


if __name__ == '__main__':
    main()
