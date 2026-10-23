import sqlite3
import tempfile
import unittest
from pathlib import Path

from claim_reaction.diffusion import analyze


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / 'obs.sqlite')
        db = sqlite3.connect(self.path)
        db.executescript('''
        CREATE TABLE observation_feed_versions(version_id INTEGER PRIMARY KEY,item_key TEXT,feed_url TEXT,observed_at TEXT,published_at TEXT,updated_at TEXT,title TEXT,summary TEXT,item_url TEXT,raw_json TEXT,bootstrap INTEGER,interval_start TEXT,analysis_ready_at TEXT,run_id INTEGER);
        CREATE TABLE observation_market_sessions(session_id INTEGER PRIMARY KEY,started_at TEXT,ended_at TEXT,status TEXT,error TEXT,run_id INTEGER);
        CREATE TABLE observation_market_frames(frame_id INTEGER PRIMARY KEY,session_id INTEGER,received_at TEXT,monotonic_received REAL,raw_payload TEXT,parse_error TEXT);
        CREATE TABLE observation_market_events(event_id INTEGER PRIMARY KEY,frame_id INTEGER,subindex INTEGER,event_type TEXT,asset_id TEXT,market_slug TEXT,exchange_time TEXT,bid REAL,ask REAL,midpoint REAL,last_trade_price REAL,trade_size REAL,trade_side TEXT,tick_size TEXT,error TEXT,normalized_json TEXT,received_at TEXT);
        ''')
        db.execute("INSERT INTO observation_market_sessions VALUES(1,'2026-01-01T00:00:00Z','2026-01-01T00:10:00Z','connected',NULL,NULL)")
        self.add_version(db, 1, 'item', 'feed', '2026-01-01T00:01:00Z')
        self.add_quote(db, 1, '2026-01-01T00:00:50Z', .4)
        self.add_quote(db, 1, '2026-01-01T00:01:20Z', .52)
        self.add_quote(db, 1, '2026-01-01T00:02:00Z', .65)
        db.commit(); db.close()

    def tearDown(self): self.tmp.cleanup()

    @staticmethod
    def add_version(db, vid, item, feed, observed, bootstrap=0, interval=None):
        db.execute('INSERT INTO observation_feed_versions VALUES(?,?,?,?,NULL,NULL,?,\'\',\'\',\'{}\',?,?,NULL,NULL)',
                   (vid,item,feed,observed,f'headline-{vid}',bootstrap,interval))

    @staticmethod
    def add_quote(db, session, at, mid):
        cur=db.execute('INSERT INTO observation_market_frames(session_id,received_at,monotonic_received,raw_payload) VALUES(?,?,0,\'{}\')',(session,at))
        db.execute('INSERT INTO observation_market_events(frame_id,event_type,market_slug,bid,ask,midpoint,error,normalized_json,received_at) VALUES(?,\'best_bid_ask\',\'m\',?,?,?,NULL,\'{}\',?)',
                   (cur.lastrowid,mid-.01,mid+.01,mid,at))

    def test_threshold_timing_and_baseline_delta(self):
        result=analyze(self.path, {'feed':['m']}, [60,180], [.5,.6])
        event=result['events'][0]
        self.assertEqual(event['baseline']['midpoint'], .4)
        self.assertEqual(event['horizons']['60']['delta'], .25)
        self.assertEqual(event['threshold_crossings']['0.5']['seconds_after_event'], 20)
        self.assertEqual(event['threshold_crossings']['0.6']['seconds_after_event'], 60)

    def test_unrelated_feed_bootstrap_duplicate_and_no_prior_quote(self):
        db=sqlite3.connect(self.path)
        self.add_version(db,2,'other','unmapped','2026-01-01T00:01:00Z')
        self.add_version(db,3,'boot','feed','2026-01-01T00:01:00Z',1)
        self.add_version(db,4,'boot','feed','2026-01-01T00:02:00Z')
        self.add_version(db,5,'late','feed','2026-01-01T00:00:40Z')
        db.commit();db.close()
        result=analyze(self.path, {'feed':['m']})
        self.assertEqual([e['item_key'] for e in result['events']],['item','late'])
        self.assertEqual(result['events'][1]['skip_reason'],'no_pre_event_quote')
        self.assertIn('bootstrap',[x['reason'] for x in result['skipped_versions']])
        self.assertIn('duplicate_or_revision',[x['reason'] for x in result['skipped_versions']])

    def test_session_gap_and_missing_horizon_coverage(self):
        db=sqlite3.connect(self.path)
        db.execute("UPDATE observation_market_sessions SET ended_at='2026-01-01T00:02:00Z'")
        db.commit();db.close()
        event=analyze(self.path, {'feed':['m']}, [60,120])['events'][0]
        self.assertTrue(event['horizons']['60']['covered'])
        self.assertFalse(event['horizons']['120']['covered'])


if __name__ == '__main__': unittest.main()
