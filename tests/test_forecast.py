import json
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from claim_reaction.forecast import (ForecastRow, add_relevance_features,
                                     evaluate_leave_one_market_out,
                                     load_forecast_rows)


class FakeEmbedder:
    def encode(self, texts, role):
        # Deterministic vectors make this test independent of downloaded weights.
        return [[1.0, float(len(texts[i]) % 7 + 1)] for i, _ in enumerate(texts)]


class ForecastTests(unittest.TestCase):
    def _database(self, path):
        db = sqlite3.connect(path)
        db.executescript("""
            CREATE TABLE markets(slug TEXT, question TEXT, raw_json TEXT);
            CREATE TABLE news(source TEXT, source_id TEXT, market_slug TEXT, url TEXT,
              title TEXT, provider_seen_at TEXT, first_collected_at TEXT);
            CREATE TABLE prices(market_slug TEXT, timestamp TEXT, price REAL,
              resolution_seconds INTEGER, source TEXT);
        """)
        db.execute("INSERT INTO markets VALUES (?, ?, ?)",
                    ("m", "Will event happen?", json.dumps({"endDate": "2026-01-02T00:00:00Z"})))
        db.execute("INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("gdelt-doc", "a", "m", "https://example.test/a", "Event report",
                     "2026-01-01T10:00:00Z", "2026-01-03T00:00:00Z"))
        db.execute("INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("gdelt-doc", "b", "m", "https://example.test/b", "Late report",
                     "2026-01-01T23:30:00Z", "2026-01-03T00:00:00Z"))
        # 5-minute aggregate buckets: first target closes at 11:00; second
        # observation has no eligible post horizon before market end.
        for stamp, price in [("2026-01-01T09:55:00Z", .4),
                             ("2026-01-01T10:55:00Z", .5),
                             ("2026-01-01T11:55:00Z", .6)]:
            db.execute("INSERT INTO prices VALUES (?, ?, ?, ?, ?)",
                       ("m", stamp, price, 300, "fixture"))
        db.commit()
        db.close()

    def test_builds_provider_time_labels_and_excludes_after_end(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "forecast.sqlite"
            self._database(db)
            rows, excluded = load_forecast_rows(str(db), horizons_minutes=(60,),
                                                max_price_age_minutes=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].available_at, "2026-01-01T10:00:00Z")
        self.assertEqual(rows[0].first_collected_at, "2026-01-03T00:00:00Z")
        self.assertAlmostEqual(rows[0].delta_yes, .1)
        self.assertEqual(excluded["after_market_end"], 1)

    def test_feature_builder_emits_prior_price_and_relevance(self):
        row = ForecastRow("c", "m", "Question", "Headline", "u", "gdelt-doc",
                          "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z",
                          "2026-01-02T00:00:00Z", 60,
                          .42, .03, "t0", "t1")
        features, names = add_relevance_features([row], FakeEmbedder())
        self.assertEqual(names, ["pre_price", "e5_cosine"])
        self.assertEqual(features.shape, (1, 2))
        self.assertAlmostEqual(features[0, 0], .42)
        self.assertGreater(features[0, 1], 0)

    def test_market_holdout_reports_distinct_baselines(self):
        rows = []
        x = []
        for market in ("a", "b", "c"):
            for i in range(3):
                y = .02 * (i - 1)
                rows.append(ForecastRow(f"{market}{i}", market, "Q", f"H{i}",
                                        f"https://example.test/{market}/{i}", "gdelt-doc",
                                        "t0", "tc", "t1", 60, .2 + i * .2, y, "t0", "t1"))
                x.append([.2 + i * .2, float(i)])
        result = evaluate_leave_one_market_out(rows, np.array(x), alphas=(.1, 1.0))
        self.assertEqual(len(result["folds"]), 3)
        self.assertEqual(result["folds"][0]["horizon_minutes"], 60)
        self.assertIn("price_only_mae", result["folds"][0])
        self.assertIn("zero_change_mae", result["folds"][0])
        self.assertIn("GDELT", result["limitation"])
        self.assertEqual(result["aggregate_mae_by_horizon"][0]["n_test"], 9)
        self.assertEqual(len(result["folds"][0]["predictions"]), 3)

    def test_training_rows_with_test_url_are_removed(self):
        rows = [
            ForecastRow("a1", "a", "Q", "H", "https://same.test/story", "gdelt-doc",
                        "t0", "tc", "t1", 60, .2, .1, "t0", "t1"),
            ForecastRow("a2", "a", "Q", "H2", "https://a.test/other", "gdelt-doc",
                        "t0", "tc", "t1", 60, .4, .0, "t0", "t1"),
            ForecastRow("b1", "b", "Q", "H", "https://same.test/story", "gdelt-doc",
                        "t0", "tc", "t1", 60, .3, .2, "t0", "t1"),
            ForecastRow("b2", "b", "Q", "H3", "https://b.test/other", "gdelt-doc",
                        "t0", "tc", "t1", 60, .5, -.1, "t0", "t1"),
            ForecastRow("c1", "c", "Q", "H4", "https://c.test/other", "gdelt-doc",
                        "t0", "tc", "t1", 60, .6, .1, "t0", "t1"),
        ]
        result = evaluate_leave_one_market_out(rows, np.array([
            [.2, .1], [.4, .2], [.3, .1], [.5, .3], [.6, .4]
        ]), alphas=(.1,))
        fold_a = next(fold for fold in result["folds"] if fold["held_out_market"] == "a")
        # b1 shares a URL with the test market and must not appear in training.
        self.assertEqual(fold_a["n_train"], 2)
        self.assertNotIn("b1", {p["claim_id"] for p in fold_a["predictions"]})

    def test_cli_direct_launch_help(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "forecast_reactions.py"
        result = subprocess.run([sys.executable, str(script), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--embedding-model", result.stdout)


if __name__ == "__main__":
    unittest.main()
