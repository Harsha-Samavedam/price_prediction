import asyncio
import tempfile
import unittest
from pathlib import Path

from claim_reaction.observation import (
    ObservationStore,
    _poll_feed,
    _timestamp,
    normalize_market_message,
    observation_status,
    parse_feed,
    run_observation,
)
from claim_reaction.core import Market


class ObservationTests(unittest.TestCase):
    def test_epoch_wire_timestamps_are_milliseconds(self):
        self.assertEqual(_timestamp("1700000000123").timestamp(), 1700000000.123)
        self.assertEqual(_timestamp("2023-11-14T22:13:20Z").timestamp(), 1700000000)
        self.assertIsNone(_timestamp("bad"))
        self.assertIsNone(_timestamp(True))

    def test_batch_normalization_preserves_trade_and_rejects_crossed_midpoint(self):
        rows = normalize_market_message([
            {"event_type": "book", "asset_id": "yes", "timestamp": "1700000000000",
             "bids": [{"price": "0.4", "size": "2"}, {"price": "0.8", "size": "0"}],
             "asks": [{"price": "0.7", "size": "1"}]},
            {"event_type": "best_bid_ask", "best_bid": "0.9", "best_ask": "0.7"},
            {"event_type": "last_trade_price", "price": "0.65", "size": "4", "side": "BUY"},
            {"event_type": "unknown_future_event"},
        ])
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["subindex"] for row in rows], [0, 1, 2, 3])
        self.assertEqual((rows[0]["bid"], rows[0]["ask"], rows[0]["midpoint"]), (0.4, 0.7, 0.55))
        self.assertIsNone(rows[1]["midpoint"])
        self.assertEqual((rows[2]["trade_size"], rows[2]["trade_side"]), (4.0, "BUY"))
        self.assertEqual(rows[3]["error"], "unknown event_type: unknown_future_event")

    def test_price_change_expands_subitems_empty_book_has_null_mid(self):
        rows = normalize_market_message({"event_type": "price_change", "market": "condition-id", "timestamp": "1700000000000",
            "price_changes": [{"asset_id": "a", "best_bid": "0.2", "best_ask": "0.8"},
                              {"asset_id": "b", "best_bid": None, "best_ask": "0.8"}]}, {"a": "market-a", "b": "market-b"})
        self.assertEqual([r["asset_id"] for r in rows], ["a", "b"])
        self.assertEqual([r["market_slug"] for r in rows], ["market-a", "market-b"])
        self.assertEqual([r["subindex"] for r in rows], [0, 1])
        self.assertEqual(rows[0]["midpoint"], 0.5)
        self.assertIsNone(rows[1]["midpoint"])
        empty = normalize_market_message({"event_type": "book", "bids": [], "asks": []})[0]
        self.assertEqual((empty["bid"], empty["ask"], empty["midpoint"]), (None, None, None))
        malformed = normalize_market_message({"event_type": "price_change", "price_changes": [42]})
        self.assertEqual(len(malformed), 1)
        self.assertIn("not an object", malformed[0]["error"])

    def test_bad_event_scalars_are_isolated_and_raw_frame_commits(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(str(Path(tmp) / "x.sqlite"))
            sid = store.db.execute("INSERT INTO observation_market_sessions(started_at,status) VALUES('now','connected')").lastrowid
            frame = store.record_frame(sid, datetime.now(timezone.utc), 1.0, "raw")
            rows = normalize_market_message([
                {"event_type": "last_trade_price", "asset_id": {}, "price": 0.4, "size": 2, "side": "BUY"},
                {"event_type": "last_trade_price", "asset_id": "yes", "price": 0.5, "size": 3, "side": "SELL"},
            ])
            store.record_events(frame, rows)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM observation_market_frames").fetchone()[0], 1)
            events = store.db.execute("SELECT asset_id,error FROM observation_market_events ORDER BY subindex").fetchall()
            self.assertEqual(len(events), 2)
            self.assertIsNone(events[0]["asset_id"])
            self.assertIn("asset_id", events[0]["error"])
            self.assertEqual(events[1]["asset_id"], "yes")
            store.close()

    def test_rss_and_atom_parse_dates_links_and_bad_dates(self):
        rss = b'''<rss><channel><item><guid>g1</guid><title>Headline</title>
          <link>/story</link><description>Summary</description>
          <pubDate>Tue, 14 Nov 2023 22:13:20 GMT</pubDate></item></channel></rss>'''
        parsed, errors = parse_feed(rss, "https://example.test/feed")
        self.assertFalse(errors)
        self.assertEqual(parsed[0]["url"], "https://example.test/story")
        self.assertEqual(parsed[0]["published_at"], "2023-11-14T22:13:20Z")
        atom = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>x</id><title>Atom</title>
          <link rel="alternate" href="/x"/><updated>broken</updated></entry></feed>'''
        parsed, errors = parse_feed(atom, "https://example.test/feed")
        self.assertEqual(parsed[0]["url"], "https://example.test/x")
        self.assertIsNone(parsed[0]["updated_at"])
        self.assertTrue(errors)

    def test_poll_versions_survive_reopen_and_304_does_not_duplicate(self):
        payloads = [b'''<rss><channel><item><guid>id</guid><title>One</title><link>https://e.test/1</link>
          <description>Summary</description><pubDate>Tue, 14 Nov 2023 22:13:20 GMT</pubDate></item></channel></rss>''',
                    b'''<rss><channel><item><guid>id</guid><title>Two</title><link>https://e.test/1</link>
          <description>Summary revised</description><pubDate>Tue, 14 Nov 2023 22:13:20 GMT</pubDate></item></channel></rss>''']
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "observe.sqlite")
            store = ObservationStore(db_path)
            calls = []
            def fetch(url, etag, modified):
                calls.append((etag, modified))
                if len(calls) == 1: return 200, payloads[0], {"ETag": "v1"}
                if len(calls) == 2: return 304, b"", {}
                return 200, payloads[1], {"ETag": "v2"}
            first = asyncio.run(_poll_feed(store, "https://e.test/rss", fetch, ("m",)))
            store.close()
            store = ObservationStore(db_path)
            second = asyncio.run(_poll_feed(store, "https://e.test/rss", fetch, ("m",)))
            third = asyncio.run(_poll_feed(store, "https://e.test/rss", fetch, ("m",)))
            rows = store.db.execute("SELECT title,bootstrap,interval_start FROM observation_feed_versions ORDER BY version_id").fetchall()
            self.assertEqual([r["title"] for r in rows], ["One", "Two"])
            self.assertEqual(rows[0]["bootstrap"], 1)
            self.assertEqual(rows[1]["bootstrap"], 0)
            self.assertIsNotNone(rows[1]["interval_start"])
            self.assertEqual((first["versions"], second["versions"], third["versions"]), (1, 0, 1))
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[1][0], "v1")
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM news WHERE source='rss-live'").fetchone()[0], 2)
            self.assertEqual([r[0] for r in store.db.execute("SELECT title FROM news WHERE source='rss-live' ORDER BY title")], ["One", "Two"])
            store.close()

    def test_invalid_xml_persists_response_evidence_and_lowercase_etag(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(str(Path(tmp) / "x.sqlite"))
            received = datetime(2026, 1, 1, tzinfo=timezone.utc)
            def invalid(*_): return 200, b"<html>login</html>", {"etag": "lowercase"}, received
            result = asyncio.run(_poll_feed(store, "https://bad.test/rss", invalid, ()))
            self.assertEqual(result["status"], "partial")
            poll = store.db.execute("SELECT http_status,status,raw_payload,raw_sha256,error FROM observation_feed_polls").fetchone()
            self.assertEqual((poll["http_status"], poll["status"], poll["raw_payload"]), (200, "partial", b"<html>login</html>"))
            self.assertIsNotNone(poll["raw_sha256"])
            self.assertIn("unsupported feed root", poll["error"])
            feed = store.db.execute("SELECT etag,last_success_at FROM observation_feeds").fetchone()
            self.assertIsNone(feed["etag"])
            self.assertIsNone(feed["last_success_at"])
            def valid(_, etag, modified):
                self.assertIsNone(etag)
                return 200, (b"<rss><channel><item><guid>old</guid><title>Old story</title>"
                             b"<link>https://bad.test/old</link>"
                             b"<pubDate>Thu, 01 Jan 2026 00:00:00 GMT</pubDate>"
                             b"</item></channel></rss>"), {}, received
            second = asyncio.run(_poll_feed(store, "https://bad.test/rss", valid, ()))
            self.assertEqual(second["status"], "ok")
            version = store.db.execute("SELECT bootstrap,interval_start FROM observation_feed_versions").fetchone()
            self.assertEqual(version["bootstrap"], 1)
            self.assertIsNone(version["interval_start"])
            store.close()

    def test_poll_failure_is_persisted_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(str(Path(tmp) / "x.sqlite"))
            def fail(*_): raise OSError("offline")
            result = asyncio.run(_poll_feed(store, "https://bad.test/rss", fail, ()))
            self.assertEqual(result["status"], "error")
            self.assertEqual(store.db.execute("SELECT last_status FROM observation_feeds").fetchone()[0], "error")
            self.assertEqual(store.db.execute("SELECT status FROM observation_feed_polls").fetchone()[0], "error")
            store.close()

    def test_opt_in_body_snapshot_records_version_and_hash(self):
        payload = (b'<rss><channel><item><guid>story</guid><title>Story</title>'
                   b'<link>https://example.test/story</link><description>Summary</description>'
                   b'<pubDate>Thu, 01 Jan 2026 00:00:00 GMT</pubDate></item></channel></rss>')
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "observe.sqlite")
            def feed(*_): return 200, payload, {}
            result = asyncio.run(run_observation(path, [], ["https://example.test/feed"],
                0.05, 1, feed_fetcher=feed, snapshot_bodies=True,
                body_fetcher=lambda url: "captured body" if url.endswith("/story") else ""))
            store = ObservationStore(path)
            row = store.db.execute("SELECT * FROM observation_body_snapshots").fetchone()
            self.assertEqual(result["status"], "ok")
            self.assertEqual(row["item_url"], "https://example.test/story")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["body"], "captured body")
            self.assertEqual(row["body_sha256"], __import__("hashlib").sha256(b"captured body").hexdigest())
            self.assertEqual(row["bootstrap"], 1)
            self.assertIsNotNone(row["fetch_completed_at"])
            store.close()

    def test_body_queue_records_all_versions_and_does_not_extend_duration(self):
        from datetime import datetime, timezone
        import time
        items = b"".join(
            f'<item><guid>{i}</guid><title>Story {i}</title><link>https://example.test/{i}</link>'
            f'<pubDate>Thu, 01 Jan 2026 00:00:00 GMT</pubDate></item>'.encode()
            for i in range(12))
        payload = b"<rss><channel>" + items + b"</channel></rss>"
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "observe.sqlite")
            def slow_body(_url):
                time.sleep(0.4)
                return "body"
            began = time.monotonic()
            result = asyncio.run(run_observation(path, [], ["https://example.test/feed"], 0.08, 1,
                feed_fetcher=lambda *_: (200, payload, {}, datetime.now(timezone.utc)),
                snapshot_bodies=True, body_fetcher=slow_body))
            elapsed = time.monotonic() - began
            self.assertGreaterEqual(elapsed, 0.3)
            store = ObservationStore(path)
            rows = store.db.execute("SELECT status,fetch_started_at,fetch_completed_at,error FROM observation_body_snapshots").fetchall()
            self.assertEqual(len(rows), 12)
            self.assertTrue(any(row["status"] == "deferred" for row in rows))
            self.assertLessEqual(sum(row["fetch_started_at"] is not None for row in rows), 8)
            self.assertEqual(result["body_snapshots"]["completed"], 8)
            self.assertEqual(result["body_snapshots"]["deferred"], 4)
            self.assertGreaterEqual(result["elapsed_seconds"], 0.3)
            for row in rows:
                if row["status"] == "deferred":
                    self.assertIsNone(row["fetch_completed_at"])
                    self.assertIn("before fetch started", row["error"])
            store.close()

    def test_deferred_body_snapshots_resume_on_unchanged_feed_run(self):
        import time
        payload = b"<rss><channel>" + b"".join(
            f'<item><guid>{i}</guid><title>Story {i}</title><link>https://example.test/{i}</link>'
            f'<pubDate>Thu, 01 Jan 2026 00:00:00 GMT</pubDate></item>'.encode()
            for i in range(10)) + b"</channel></rss>"
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "observe.sqlite")
            feed = lambda *_: (200, payload, {})
            first = asyncio.run(run_observation(path, [], ["https://example.test/feed"], 0.05, 1,
                feed_fetcher=feed, snapshot_bodies=True,
                body_fetcher=lambda _url: (time.sleep(0.25), "first body")[1]))
            self.assertGreater(first["body_snapshots"]["deferred"], 0)
            second = asyncio.run(run_observation(path, [], ["https://example.test/feed"], 0.15, 1,
                feed_fetcher=feed, snapshot_bodies=True, body_fetcher=lambda _url: "resumed body"))
            self.assertEqual(second["body_snapshots"]["completed"], first["body_snapshots"]["deferred"])
            store = ObservationStore(path)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM observation_feed_versions").fetchone()[0], 10)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM observation_body_snapshots").fetchone()[0], 10)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM observation_body_snapshots WHERE status!='ok'").fetchone()[0], 0)
            self.assertEqual(store.db.execute("SELECT COUNT(DISTINCT feed_version_id) FROM observation_body_snapshots").fetchone()[0], 10)
            store.close()

    def test_deferred_snapshots_from_unselected_feeds_are_not_retried(self):
        import time
        payload = b"<rss><channel>" + b"".join(
            f'<item><guid>{i}</guid><title>Story {i}</title><link>https://example.test/{i}</link>'
            f'<pubDate>Thu, 01 Jan 2026 00:00:00 GMT</pubDate></item>'.encode()
            for i in range(10)) + b"</channel></rss>"
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "observe.sqlite")
            first = asyncio.run(run_observation(path, [], ["https://example.test/first"], 0.05, 1,
                feed_fetcher=lambda *_: (200, payload, {}), snapshot_bodies=True,
                body_fetcher=lambda _url: (time.sleep(0.2), "body")[1]))
            deferred_before = first["body_snapshots"]["deferred"]
            self.assertGreater(deferred_before, 0)
            second = asyncio.run(run_observation(path, [], ["https://example.test/second"], 0.08, 1,
                feed_fetcher=lambda *_: (304, b"", {}), snapshot_bodies=True,
                body_fetcher=lambda _url: "unexpected"))
            self.assertEqual(second["body_snapshots"]["completed"], 0)
            store = ObservationStore(path)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM observation_body_snapshots WHERE status='deferred'").fetchone()[0], deferred_before)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM observation_body_snapshots WHERE body='unexpected'").fetchone()[0], 0)
            store.close()

    def test_status_is_read_only_and_handles_database_paths_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "observe db.sqlite")
            store = ObservationStore(path)
            store.close()
            self.assertTrue(observation_status(path)["exists"])
            self.assertFalse(observation_status(str(Path(tmp) / "missing db.sqlite"))["exists"])

    def test_run_observation_bounds_reconnect_backoff_and_records_session(self):
        class Markets:
            def market_by_slug(self, slug):
                return Market(slug, "1", "question", "", "yes-token", "v1",
                              {"active": True, "closed": False, "enableOrderBook": True})
        async def failed_connect(*args, **kwargs):
            raise OSError("offline")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "observe.sqlite")
            started = asyncio.get_event_loop_policy().new_event_loop()
            try:
                began = __import__("time").monotonic()
                result = started.run_until_complete(run_observation(
                    path, ["market"], [], 0.05, polymarket=Markets(), websocket_connect=failed_connect))
                elapsed = __import__("time").monotonic() - began
            finally:
                started.close()
            self.assertLess(elapsed, 0.5)
            self.assertEqual(result["status"], "partial")
            self.assertTrue(result["errors"])
            status = ObservationStore(path)
            session = status.db.execute("SELECT status,error FROM observation_market_sessions").fetchone()
            self.assertEqual(session["status"], "disconnected")
            self.assertIn("offline", session["error"])
            status.close()

    def test_market_lookup_failure_is_visible_in_saved_run(self):
        class Markets:
            def market_by_slug(self, slug):
                raise OSError("market metadata unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "observe.sqlite")
            with self.assertRaisesRegex(OSError, "market metadata unavailable"):
                asyncio.run(run_observation(path, ["market"], [], 0.05,
                                            polymarket=Markets(), websocket_connect=lambda *_: None))
            store = ObservationStore(path)
            run = store.db.execute("SELECT status,error FROM observation_runs").fetchone()
            self.assertEqual(run["status"], "error")
            self.assertIn("market metadata unavailable", run["error"])
            store.close()

    def test_quiet_websocket_does_not_outlive_requested_duration(self):
        class Markets:
            def market_by_slug(self, slug):
                return Market(slug, "1", "question", "", "yes-token", "v1", {"active": True})
        class QuietSocket:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return None
            async def send(self, message): pass
            async def recv(self): await asyncio.Future()
            async def close(self): pass
        async def connect(*args, **kwargs): return QuietSocket()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "quiet.sqlite")
            began = __import__("time").monotonic()
            result = asyncio.run(asyncio.wait_for(run_observation(
                path, ["market"], [], 0.05, polymarket=Markets(), websocket_connect=connect), timeout=0.3))
            self.assertLess(__import__("time").monotonic() - began, 0.3)
            self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
