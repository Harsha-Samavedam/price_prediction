import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_body_sample


class EligibleRow:
    def __init__(self, market_slug, article_url, available_at, headline="Headline"):
        self.market_slug = market_slug
        self.article_url = article_url
        self.available_at = available_at
        self.first_collected_at = "2026-01-03T00:00:00Z"
        self.headline = headline

    @property
    def delta_yes(self):
        raise AssertionError("selection must not inspect reaction outcomes")


class BodySampleTests(unittest.TestCase):
    def test_market_allocation_spans_time_and_fetches_each_url_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "source.sqlite"
            db.write_bytes(b"fixture source database")
            out = root / "sample"
            rows = [
                EligibleRow("b", "https://example.test/shared", "2026-01-01T01:00:00Z"),
                EligibleRow("a", "https://example.test/a2", "2026-01-01T02:00:00Z"),
                EligibleRow("a", "https://example.test/shared", "2026-01-01T01:00:00Z"),
                EligibleRow("a", "https://example.test/a1", "2026-01-01T00:00:00Z"),
                EligibleRow("a", "https://example.test/a1", "2026-01-01T00:00:00Z", "duplicate"),
            ]
            fetched = []

            def fetch(url):
                fetched.append(url)
                return "Fetched story body"

            with patch.object(build_body_sample, "load_forecast_rows", return_value=(rows, {})), \
                    patch.object(build_body_sample, "fetch_article_body", side_effect=fetch):
                manifest = build_body_sample.build(db, out, limit=3, workers=2, retries=0)
            docs = [json.loads(line) for line in (out / "documents.jsonl").read_text().splitlines()]
            self.assertEqual([(row["market_slug"], row["article_url"]) for row in docs], [
                ("a", "https://example.test/a1"),
                ("a", "https://example.test/a2"),
                ("b", "https://example.test/shared"),
            ])
            a_times = [row["provider_seen_at"] for row in docs if row["market_slug"] == "a"]
            self.assertEqual(a_times, ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"])
            self.assertEqual(len(fetched), 3)
            self.assertEqual(len(set(fetched)), 3)
            self.assertEqual(fetched.count("https://example.test/shared"), 1)
            self.assertEqual(manifest["coverage"]["selected_pairs"], 3)
            self.assertTrue(all(doc["body_retrospective"] for doc in docs))
            self.assertTrue(all("delta_yes" not in doc for doc in docs))

    def test_failure_is_recorded_and_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "source.sqlite"
            db.write_bytes(b"db")
            out = root / "sample"
            rows = [EligibleRow("m", "https://example.test/a", "2026-01-01T00:00:00Z")]
            with patch.object(build_body_sample, "load_forecast_rows", return_value=(rows, {})), \
                    patch.object(build_body_sample, "fetch_article_body", side_effect=OSError("offline")):
                build_body_sample.build(db, out, limit=1, workers=1, retries=0)
            row = json.loads((out / "documents.jsonl").read_text().strip())
            self.assertEqual(row["body_fetch_status"], "failed")
            self.assertEqual(row["body_fetch_error"], "OSError: offline")
            with self.assertRaises(FileExistsError):
                build_body_sample.build(db, out, limit=1)
            self.assertEqual(len((out / "documents.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
