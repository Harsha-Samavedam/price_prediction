import csv
import io
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from claim_reaction.__main__ import parser, run
from claim_reaction.core import PricePoint, parse_polymarket_market, parse_time
from claim_reaction.replay import replay_claims
from claim_reaction.sources import gdelt_doc, gdelt_gkg_archive, mediacloud


class MarketParsingTests(unittest.TestCase):
    def test_v1_and_v2_use_the_correct_yes_identifier(self):
        basic = {"slug": "a", "id": "1", "question": "A?",
                 "outcomes": '["No", "Yes"]',
                 "clobTokenIds": '["no-v1", "yes-v1"]',
                 "positionIds": ["no-v2", "yes-v2"]}
        self.assertEqual(parse_polymarket_market({**basic, "version": "v1"}).yes_asset_id,
                         "yes-v1")
        self.assertEqual(parse_polymarket_market({**basic, "version": "v2"}).yes_asset_id,
                         "yes-v2")


class ReplayTests(unittest.TestCase):
    def test_prices_are_point_in_time_and_settlement_is_excluded(self):
        claim = {"claim_id": "c", "market_slug": "m", "claim_text": "new report",
                 "article_url": None, "available_at": "2026-01-01T10:00:00Z"}
        prices = [
            PricePoint("m", parse_time("2026-01-01T09:59:00Z"), .40, 60, "csv"),
            PricePoint("m", parse_time("2026-01-01T10:01:00Z"), .43, 60, "csv"),
            PricePoint("m", parse_time("2026-01-01T10:59:00Z"), .48, 60, "csv"),
            PricePoint("m", parse_time("2026-01-01T11:00:00Z"), 1.0, 0, "csv"),
        ]
        rows, excluded = replay_claims([claim], prices, 60, 2)
        self.assertEqual(excluded, {"no_pre_price": 0, "stale_pre_price": 0,
                                    "no_post_price": 0, "stale_post_price": 0})
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].delta_yes, .08)
        self.assertEqual(rows[0].post_price_time, "2026-01-01T11:00:00Z")

    def test_aggregated_bucket_crossing_claim_is_not_used_as_pre_price(self):
        claim = {"claim_id": "c", "market_slug": "m", "claim_text": "new report",
                 "article_url": None, "available_at": "2026-01-01T10:05:00Z"}
        prices = [
            PricePoint("m", parse_time("2026-01-01T09:30:00Z"), .40, 1800, "v2"),
            PricePoint("m", parse_time("2026-01-01T10:00:00Z"), .70, 1800, "v2"),
            PricePoint("m", parse_time("2026-01-01T10:30:00Z"), .80, 1800, "v2"),
        ]
        rows, _ = replay_claims([claim], prices, 60, 35)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].pre_price, .40)
        self.assertEqual(rows[0].pre_price_time, "2026-01-01T10:00:00Z")
        self.assertEqual(rows[0].post_price, .80)
        self.assertEqual(rows[0].post_price_time, "2026-01-01T11:00:00Z")

    def test_stale_quotes_do_not_create_labels(self):
        claim = {"claim_id": "c", "market_slug": "m", "claim_text": "new report",
                 "article_url": None, "available_at": "2026-01-01T10:00:00Z"}
        prices = [PricePoint("m", parse_time("2026-01-01T09:30:00Z"), .4, None, "csv")]
        rows, excluded = replay_claims([claim], prices, 60, 5)
        self.assertEqual(rows, [])
        self.assertEqual(excluded["stale_pre_price"], 1)


class SourceTests(unittest.TestCase):
    def test_gdelt_doc_uses_provider_seen_time_not_publication(self):
        payload = {"articles": [{"url": "https://example.com/story",
                                 "title": "Report", "seendate": "20260101T120500Z"}]}
        with patch("claim_reaction.sources.get_json", return_value=payload):
            items = list(gdelt_doc("report", parse_time("2026-01-01T12:00:00Z"),
                                   parse_time("2026-01-01T13:00:00Z"), "m"))
        self.assertEqual(items[0].provider_seen_at,
                         parse_time("2026-01-01T12:05:00Z"))
        self.assertIsNone(items[0].published_at)

    def test_gdelt_doc_splits_long_historical_range(self):
        calls = []

        def response(_url, params):
            calls.append(params)
            return {"articles": [{"url": f"https://example.com/{len(calls)}"}]}

        start = parse_time("2025-01-01T00:00:00Z")
        end = parse_time("2025-05-01T00:00:00Z")
        with patch("claim_reaction.sources.get_json", side_effect=response):
            items = list(gdelt_doc("report", start, end, "m"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(items), 2)
        for call in calls:
            window_start = datetime.strptime(call["startdatetime"], "%Y%m%d%H%M%S")
            window_end = datetime.strptime(call["enddatetime"], "%Y%m%d%H%M%S")
            self.assertLessEqual(window_end - window_start, timedelta(days=60, seconds=1))

    def test_gdelt_doc_bisects_capped_window_and_deduplicates_urls(self):
        calls = []

        def response(_url, params):
            calls.append(params)
            if len(calls) == 1:
                return {"articles": [{"url": f"https://example.com/{i}"}
                                     for i in range(250)]}
            return {"articles": [{"url": "https://example.com/shared",
                                  "seendate": "20260101T120500Z"}]}

        with patch("claim_reaction.sources.get_json", side_effect=response):
            items = list(gdelt_doc("report", parse_time("2026-01-01T12:00:00Z"),
                                   parse_time("2026-01-01T13:00:00Z"), "m"))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(items), 1)

    def test_gdelt_doc_includes_article_at_chunk_boundary(self):
        boundary = parse_time("2026-03-02T00:00:00Z")

        def response(_url, params):
            window_start = datetime.strptime(params["startdatetime"], "%Y%m%d%H%M%S")
            window_end = datetime.strptime(params["enddatetime"], "%Y%m%d%H%M%S")
            if window_start < boundary.replace(tzinfo=None) < window_end:
                return {"articles": [{"url": "https://example.com/boundary",
                                      "seendate": "20260302T000000Z"}]}
            return {"articles": []}

        with patch("claim_reaction.sources.get_json", side_effect=response):
            items = list(gdelt_doc("report", parse_time("2026-01-01T00:00:00Z"),
                                   parse_time("2026-04-01T00:00:00Z"), "m"))
        self.assertEqual(len(items), 1)

    def test_gdelt_archive_uses_batch_timestamp_and_metadata_filter(self):
        fields = [""] * 27
        fields[0] = "record-id"
        fields[1] = "20260101120000"
        fields[3] = "example.com"
        fields[4] = "https://example.com/story"
        fields[11] = "jane candidate"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("sample.gkg.csv", "\t".join(fields) + "\n")
        with patch("claim_reaction.sources.get_bytes", return_value=buffer.getvalue()):
            items = list(gdelt_gkg_archive("Jane Candidate",
                                           parse_time("2026-01-01T12:00:00Z"),
                                           parse_time("2026-01-01T12:15:00Z"), "market"))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].provider_seen_at,
                         datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
        self.assertIsNone(items[0].published_at)

    def test_mediacloud_uses_indexed_time_and_pages(self):
        first = {"stories": [{"id": "1", "url": "https://example.com/a",
                              "title": "A", "publish_date": "2026-01-01",
                              "indexed_date": "2026-01-01T12:05:00+00:00"}],
                 "pagination_token": "next"}
        second = {"stories": [], "pagination_token": None}
        with patch("claim_reaction.sources.require_key", return_value="test"), \
             patch("claim_reaction.sources.get_json", side_effect=[first, second]):
            items = list(mediacloud("report", parse_time("2026-01-01T00:00:00Z"),
                                    parse_time("2026-01-02T00:00:00Z"), "m", 123))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].provider_seen_at,
                         parse_time("2026-01-01T12:05:00Z"))
        self.assertIsNone(items[0].published_at)


class CliTests(unittest.TestCase):
    def test_csv_import_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "data.sqlite"
            price_csv = root / "prices.csv"
            claims_csv = root / "claims.csv"
            output = root / "replay.csv"
            price_csv.write_text("timestamp,price\n"
                                 "2026-01-01T09:59:00Z,0.40\n"
                                 "2026-01-01T10:59:00Z,0.48\n", encoding="utf-8")
            claims_csv.write_text("market_slug,claim_text,available_at,article_url\n"
                                  "m,New report,2026-01-01T10:00:00Z,https://example.com\n",
                                  encoding="utf-8")
            p = parser()
            run(p.parse_args(["--db", str(db), "prices-csv", "--slug", "m",
                              "--file", str(price_csv)]))
            run(p.parse_args(["--db", str(db), "claims-csv", "--file", str(claims_csv)]))
            result = run(p.parse_args(["--db", str(db), "replay", "--slug", "m",
                                       "--price-source", "csv", "--horizon-minutes", "60",
                                       "--max-price-age-minutes", "2", "--out", str(output)]))
            self.assertEqual(result["labeled_claims"], 1)
            with output.open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(float(row["delta_yes"]), .08)


if __name__ == "__main__":
    unittest.main()
