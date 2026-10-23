"""Offline headline reaction forecasts from retrospective GDELT observations.

GDELT's indexed headlines are retrospective observations, not a live event feed.
They are used here as provisional event times at ``provider_seen_at``; results
must not be interpreted as a live-trading backtest.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Iterable

import numpy as np

from .core import PricePoint, iso, parse_time
from .replay import replay_claims

HORIZONS_MINUTES = (60, 1440)
MAX_PRICE_AGE_MINUTES = 30
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class ForecastRow:
    claim_id: str
    market_slug: str
    market_question: str
    headline: str
    article_url: str
    source: str
    available_at: str
    first_collected_at: str
    end_date: str
    horizon_minutes: int
    pre_price: float
    delta_yes: float
    pre_price_time: str
    post_price_time: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_forecast_rows(db_path: str, horizons_minutes: Iterable[int] = HORIZONS_MINUTES,
                       max_price_age_minutes: int = MAX_PRICE_AGE_MINUTES
                       ) -> tuple[list[ForecastRow], dict[str, int]]:
    """Create labeled GDELT headline observations from the collected SQLite DB.

    Only rows with a provider indexed time, URL, title, and a market endDate are
    eligible. The requested horizon must end no later than endDate. Price
    points are interpreted by ``replay_claims`` using their bucket end times.
    """
    horizons = tuple(int(value) for value in horizons_minutes)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must contain positive minute values")
    if max_price_age_minutes < 0:
        raise ValueError("maximum price age must be nonnegative")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        market_rows = db.execute("SELECT slug, question, raw_json FROM markets").fetchall()
        markets: dict[str, tuple[str, Any]] = {}
        for market in market_rows:
            raw = json.loads(market["raw_json"])
            end_date = raw.get("endDate")
            if end_date:
                markets[market["slug"]] = (market["question"], parse_time(end_date))
        news = db.execute(
            "SELECT source, source_id, market_slug, url, title, provider_seen_at, first_collected_at "
            "FROM news WHERE source='gdelt-doc' AND provider_seen_at IS NOT NULL "
            "AND url != '' AND title != '' ORDER BY provider_seen_at, market_slug, url"
        ).fetchall()
        prices_by_market: dict[str, list[PricePoint]] = {}
        for row in db.execute("SELECT * FROM prices ORDER BY market_slug, timestamp"):
            point = PricePoint(row["market_slug"], parse_time(row["timestamp"]),
                               float(row["price"]), row["resolution_seconds"], row["source"])
            prices_by_market.setdefault(point.market_slug, []).append(point)
    finally:
        db.close()

    excluded = {"missing_market_end_date": 0, "after_market_end": 0,
                "no_price_label": 0}
    result: list[ForecastRow] = []
    seen_market_urls: set[tuple[str, str]] = set()
    for item in news:
        slug = item["market_slug"]
        market_url = (slug, item["url"])
        if market_url in seen_market_urls:
            continue
        seen_market_urls.add(market_url)
        market = markets.get(slug)
        if market is None:
            excluded["missing_market_end_date"] += len(horizons)
            continue
        available = parse_time(item["provider_seen_at"])
        question, end_date = market
        eligible_horizons = []
        claims = []
        for horizon in horizons:
            if available + timedelta(minutes=horizon) > end_date:
                excluded["after_market_end"] += 1
                continue
            claim_id = f"{item['source']}:{item['source_id']}:{slug}:{horizon}"
            eligible_horizons.append(horizon)
            claims.append({"claim_id": claim_id, "market_slug": slug,
                           "claim_text": item["title"], "article_url": item["url"],
                           "available_at": iso(available)})
        if not claims:
            continue
        # Replay each horizon independently; this preserves the library's
        # point-in-time price bucket and staleness rules.
        for horizon, claim in zip(eligible_horizons, claims):
            rows, _ = replay_claims([claim], prices_by_market.get(slug, []),
                                    horizon, max_price_age_minutes)
            if not rows:
                excluded["no_price_label"] += 1
                continue
            label = rows[0]
            result.append(ForecastRow(
                claim_id=claim["claim_id"], market_slug=slug,
                market_question=question, headline=item["title"], article_url=item["url"],
                source=item["source"], available_at=iso(available) or "",
                first_collected_at=item["first_collected_at"],
                end_date=iso(end_date) or "", horizon_minutes=horizon,
                pre_price=label.pre_price, delta_yes=label.delta_yes,
                pre_price_time=label.pre_price_time, post_price_time=label.post_price_time,
            ))
    return result, excluded


def add_relevance_features(rows: list[ForecastRow], embedder: Any) -> tuple[np.ndarray, list[str]]:
    """Return fast scalar features: prior YES price and E5 headline/question cosine.

    The given embedder is the existing ``HuggingFaceEmbedder`` configured with
    ``intfloat/multilingual-e5-small``. Encoding is batched over unique texts.
    """
    if not rows:
        return np.empty((0, 2), dtype=float), ["pre_price", "e5_cosine"]
    questions = list(dict.fromkeys(row.market_question for row in rows))
    headlines = list(dict.fromkeys(row.headline for row in rows))
    q_vectors = dict(zip(questions, embedder.encode(questions, "query")))
    h_vectors = dict(zip(headlines, embedder.encode(headlines, "document")))
    features = []
    for row in rows:
        qv, hv = q_vectors[row.market_question], h_vectors[row.headline]
        qnorm = math.sqrt(sum(value * value for value in qv))
        hnorm = math.sqrt(sum(value * value for value in hv))
        cosine = sum(a * b for a, b in zip(qv, hv)) / (qnorm * hnorm) if qnorm and hnorm else 0.0
        features.append([row.pre_price, cosine])
    return np.asarray(features, dtype=float), ["pre_price", "e5_cosine"]


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    return (train - mean) / scale, (other - mean) / scale


def evaluate_leave_one_market_out(rows: list[ForecastRow], features: np.ndarray,
                                  alphas: Iterable[float] = RIDGE_ALPHAS
                                  ) -> dict[str, Any]:
    """Evaluate zero-change, price-only, and relevance-augmented Ridge forecasts.

    Each held-out market is tested separately. Any training row whose article
    URL occurs in that held-out market is removed. Ridge alpha is selected by
    internal leave-one-market-out MAE using training markets only.
    """
    if len(rows) != len(features):
        raise ValueError("rows and feature matrix lengths differ")
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(a <= 0 for a in alphas):
        raise ValueError("alphas must be positive")

    def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        # Match Ridge's unpenalized intercept by centering using training data.
        mean_x, mean_y = x.mean(axis=0), float(y.mean())
        weights = np.linalg.solve((x - mean_x).T @ (x - mean_x) +
                                  alpha * np.eye(x.shape[1]),
                                  (x - mean_x).T @ (y - mean_y))
        return np.append(weights, mean_y - mean_x @ weights)

    def predict_ridge(params: np.ndarray, x: np.ndarray) -> np.ndarray:
        return x @ params[:-1] + params[-1]

    reports = []
    for horizon in sorted({row.horizon_minutes for row in rows}):
        horizon_indices = [i for i, row in enumerate(rows) if row.horizon_minutes == horizon]
        for held_out in sorted({rows[i].market_slug for i in horizon_indices}):
            test_idx = [i for i in horizon_indices if rows[i].market_slug == held_out]
            test_urls = {rows[i].article_url for i in test_idx}
            train_idx = [i for i in horizon_indices if rows[i].market_slug != held_out
                         and rows[i].article_url not in test_urls]
            if not train_idx or not test_idx:
                continue
            train_markets = sorted({rows[i].market_slug for i in train_idx})
            price_features = features[:, :1]

            def tune_alpha(feature_matrix: np.ndarray) -> float | None:
                best_alpha, best_mae = None, float("inf")
                for alpha in alphas:
                    inner_errors = []
                    for inner_market in train_markets:
                        inner_test = [i for i in train_idx if rows[i].market_slug == inner_market]
                        inner_urls = {rows[i].article_url for i in inner_test}
                        inner_train = [i for i in train_idx if rows[i].market_slug != inner_market
                                       and rows[i].article_url not in inner_urls]
                        if not inner_train or not inner_test:
                            continue
                        xt, xv = _standardize(feature_matrix[inner_train], feature_matrix[inner_test])
                        params = fit_ridge(xt, np.array([rows[i].delta_yes for i in inner_train]), alpha)
                        inner_errors.extend(np.abs(predict_ridge(params, xv) -
                                                   [rows[i].delta_yes for i in inner_test]))
                    mae = float(np.mean(inner_errors)) if inner_errors else float("inf")
                    if mae < best_mae:
                        best_alpha, best_mae = alpha, mae
                return best_alpha

            best_alpha = tune_alpha(features)
            price_alpha = tune_alpha(price_features)
            if best_alpha is None or price_alpha is None:
                continue
            xt, xv = _standardize(features[train_idx], features[test_idx])
            yt = np.array([rows[i].delta_yes for i in train_idx])
            yv = np.array([rows[i].delta_yes for i in test_idx])
            params = fit_ridge(xt, yt, best_alpha)
            full_pred = predict_ridge(params, xv)
            # Price-only uses the same train-only Ridge procedure with the
            # relevance column removed; zero-change predicts no movement.
            p_train, p_test = _standardize(price_features[train_idx], price_features[test_idx])
            price_params = fit_ridge(p_train, yt, price_alpha)
            price_pred = predict_ridge(price_params, p_test)
            zero_mae = float(np.mean(np.abs(yv)))
            fold_predictions = []
            for position, row_index in enumerate(test_idx):
                row = rows[row_index]
                fold_predictions.append({
                    "claim_id": row.claim_id, "market_slug": row.market_slug,
                    "article_url": row.article_url, "available_at": row.available_at,
                    "first_collected_at": row.first_collected_at,
                    "horizon_minutes": horizon, "actual_delta_yes": float(yv[position]),
                    "ridge_prediction": float(full_pred[position]),
                    "price_only_prediction": float(price_pred[position]),
                    "zero_change_prediction": 0.0,
                })
            reports.append({"horizon_minutes": horizon, "held_out_market": held_out,
                            "n_test": len(test_idx), "n_train": len(train_idx),
                            "selected_alpha": best_alpha, "price_only_selected_alpha": price_alpha,
                            "ridge_mae": float(np.mean(np.abs(full_pred - yv))),
                            "price_only_mae": float(np.mean(np.abs(price_pred - yv))),
                            "zero_change_mae": zero_mae, "train_markets": train_markets,
                            "predictions": fold_predictions})
    aggregates = []
    for horizon in sorted({fold["horizon_minutes"] for fold in reports}):
        horizon_folds = [fold for fold in reports if fold["horizon_minutes"] == horizon]
        count = sum(fold["n_test"] for fold in horizon_folds)
        aggregates.append({
            "horizon_minutes": horizon, "n_test": count,
            **{key: sum(fold[key] * fold["n_test"] for fold in horizon_folds) / count
               for key in ("ridge_mae", "price_only_mae", "zero_change_mae")},
        })
    return {"limitation": "Retrospective GDELT indexed headlines are provisional event observations, not live-arriving events.",
            "folds": reports, "aggregate_mae_by_horizon": aggregates}
