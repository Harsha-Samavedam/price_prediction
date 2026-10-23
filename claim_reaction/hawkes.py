"""Binned marked Hawkes model for retrospective YES-price moves.

All times are numeric seconds on one shared clock. A price label belongs to
the bucket ending at ``bucket_start + interval_seconds``; it becomes history
only after that endpoint. News marks are available at their supplied times.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

Mode = Literal["constant", "price_history", "timestamp_only", "signed_news"]


@dataclass(frozen=True)
class HawkesModel:
    """Two competing nonnegative per-bin hazard scores."""

    mode: Mode
    decay_seconds: float
    coefficients: NDArray[np.float64]  # shape (2 directions, feature count)
    feature_names: tuple[str, ...]
    interval_seconds: float = 1.0


@dataclass(frozen=True)
class Scores:
    log_loss: float
    brier: float
    probabilities: NDArray[np.float64]


def _validate(bucket_starts: ArrayLike, news_times: ArrayLike, news_marks: ArrayLike,
              labels: ArrayLike, interval_seconds: float, decay_seconds: float
              ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    starts = np.asarray(bucket_starts, dtype=float)
    nt, marks = np.asarray(news_times, dtype=float), np.asarray(news_marks, dtype=float)
    y = np.asarray(labels, dtype=int)
    if starts.ndim != 1 or y.shape != starts.shape or nt.ndim != 1 or marks.shape != nt.shape:
        raise ValueError("timestamps, marks, and labels must be one-dimensional and aligned")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(nt)) or not np.all(np.isfinite(marks)):
        raise ValueError("inputs must be finite")
    if np.any(np.diff(starts) <= 0) or interval_seconds <= 0 or decay_seconds <= 0:
        raise ValueError("bucket starts must increase and intervals/decay must be positive")
    if np.any((y < 0) | (y > 2)):
        raise ValueError("labels must be 0=none, 1=up, or 2=down")
    return starts, nt, marks, y


def history_features(bucket_starts: ArrayLike, news_times: ArrayLike, news_marks: ArrayLike,
                     price_labels: ArrayLike, interval_seconds: float,
                     decay_seconds: float, mode: Mode = "signed_news"
                     ) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Return causal exponential histories, one row per bucket start.

    News is included only when its timestamp is strictly before the bucket
    start. Price labels contribute only when their entire preceding bucket has
    ended by the current start.
    """
    starts, nt, marks, labels = _validate(bucket_starts, news_times, news_marks,
                                           price_labels, interval_seconds, decay_seconds)
    if mode not in ("constant", "price_history", "timestamp_only", "signed_news"):
        raise ValueError(f"unknown mode: {mode}")
    names = ["constant"]
    if mode in ("timestamp_only", "signed_news"):
        names += ["news"] if mode == "timestamp_only" else ["news_up", "news_down"]
    if mode != "constant":
        names += ["price_up", "price_down"]
    x = np.zeros((len(starts), len(names)), dtype=float)
    x[:, 0] = 1.0
    # Maintain decaying sums as a single pass over sorted observation times.
    # This preserves the direct exponential formula while scaling linearly
    # with buckets plus observations instead of their product.
    news_order = np.argsort(nt, kind="stable")
    nt_sorted, marks_sorted = nt[news_order], marks[news_order]
    news_cursor = 0
    news_state = np.zeros(3, dtype=float)  # total, positive-mark, negative-mark
    price_ends = starts + interval_seconds
    price_cursor = 0
    price_state = np.zeros(2, dtype=float)
    previous_start: float | None = None
    for i, start in enumerate(starts):
        if previous_start is not None:
            elapsed = start - previous_start
            news_state *= np.exp(-elapsed / decay_seconds)
            price_state *= np.exp(-elapsed / decay_seconds)
        while news_cursor < len(nt_sorted) and nt_sorted[news_cursor] < start:
            age = start - nt_sorted[news_cursor]
            weight = np.exp(-age / decay_seconds)
            mark = marks_sorted[news_cursor]
            news_state[0] += weight
            if mark > 0:
                news_state[1] += weight * abs(mark)
            elif mark < 0:
                news_state[2] += weight * abs(mark)
            news_cursor += 1
        while price_cursor < len(price_ends) and price_ends[price_cursor] <= start:
            age = start - price_ends[price_cursor]
            event = labels[price_cursor]
            if event in (1, 2):
                price_state[event - 1] += np.exp(-age / decay_seconds)
            price_cursor += 1
        offset = 1
        if mode == "timestamp_only":
            x[i, offset] = news_state[0]; offset += 1
        elif mode == "signed_news":
            x[i, offset] = news_state[1]
            x[i, offset + 1] = news_state[2]
            offset += 2
        if mode != "constant":
            x[i, offset:offset + 2] = price_state
        previous_start = start
    return x, tuple(names)


def predict_probabilities(model: HawkesModel, features: ArrayLike,
                          interval_seconds: float) -> NDArray[np.float64]:
    """Competing-hazard probabilities for none/up/down in each bin."""
    x = np.asarray(features, dtype=float)
    if interval_seconds <= 0 or x.ndim != 2 or x.shape[1] != model.coefficients.shape[1]:
        raise ValueError("invalid features or interval")
    hazards = np.maximum(x @ model.coefficients.T, 0.0) * (interval_seconds / model.interval_seconds)
    total = hazards.sum(axis=1)
    p_move = -np.expm1(-total)
    directional = np.divide(hazards, total[:, None], out=np.full_like(hazards, .5), where=total[:, None] > 0)
    return np.column_stack((1.0 - p_move, p_move[:, None] * directional))


def fit_hawkes(bucket_starts: ArrayLike, news_times: ArrayLike, news_marks: ArrayLike,
               price_labels: ArrayLike, interval_seconds: float, decay_seconds: float,
               mode: Mode = "signed_news", train_indices: Sequence[int] | None = None
               ) -> HawkesModel:
    """Fit nonnegative competing intensities by discrete interval MLE.

    If ``train_indices`` is supplied, only those bins enter optimization.
    Feature histories still use strictly prior observations according to time.
    """
    starts, nt, marks, labels = _validate(bucket_starts, news_times, news_marks,
                                           price_labels, interval_seconds, decay_seconds)
    x, names = history_features(starts, nt, marks, labels, interval_seconds, decay_seconds, mode)
    return fit_from_features(x, labels, interval_seconds, mode, decay_seconds, names, train_indices)


def fit_from_features(features: ArrayLike, labels: ArrayLike, interval_seconds: float,
                      mode: Mode, decay_seconds: float, feature_names: Sequence[str],
                      train_indices: Sequence[int] | None = None) -> HawkesModel:
    """Fit the discrete likelihood on precomputed causal features.

    Use this to pool feature rows from multiple independently constructed
    market histories without allowing one market's events into another's.
    """
    x = np.asarray(features, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    names = tuple(feature_names)
    if (x.ndim != 2 or labels_array.shape != (len(x),) or not len(x)
            or x.shape[1] != len(names) or not np.all(np.isfinite(x))):
        raise ValueError("features and labels must be finite, nonempty, and aligned")
    if np.any(x < 0) or np.any((labels_array < 0) | (labels_array > 2)):
        raise ValueError("features must be nonnegative and labels in {0,1,2}")
    if mode not in ("constant", "price_history", "timestamp_only", "signed_news"):
        raise ValueError(f"unknown mode: {mode}")
    if interval_seconds <= 0 or decay_seconds <= 0 or not np.all(np.isfinite([interval_seconds, decay_seconds])):
        raise ValueError("interval and decay must be positive and finite")
    idx = np.arange(len(x)) if train_indices is None else np.asarray(train_indices, dtype=int)
    if idx.ndim != 1 or len(idx) == 0 or np.any((idx < 0) | (idx >= len(x))):
        raise ValueError("train_indices must select at least one valid bin")
    xt, yt = x[idx], labels_array[idx]
    # Coefficients are per-bin hazard contributions. This avoids tiny
    # per-second parameters at long bucket intervals and improves conditioning.
    # Bounds enforce nonnegative baseline
    # and excitation weights; deterministic L-BFGS-B has no random state.
    init = np.zeros((2, x.shape[1]), dtype=float)
    init[:, 0] = 0.05
    bounds = [(1e-9, None) if feature == 0 else (0.0, None)
              for _ in range(2) for feature in range(x.shape[1])]

    def objective(flat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        hazard = xt @ flat.reshape(2, -1).T
        total = hazard.sum(axis=1)
        none = yt == 0
        moving = ~none
        loss = float(total[none].sum())
        gradient_hazard = np.ones_like(hazard)
        if np.any(moving):
            s = total[moving]
            observed_direction = yt[moving] - 1
            h_observed = hazard[moving, observed_direction]
            loss -= float(np.sum(np.log(-np.expm1(-s)) + np.log(h_observed) - np.log(s)))
            reciprocal_expm1 = np.zeros_like(s)
            small = s < 50.0
            reciprocal_expm1[small] = 1.0 / np.expm1(s[small])
            common = reciprocal_expm1 - 1.0 / s
            gradient_hazard[moving] = -common[:, None]
            moving_rows = np.flatnonzero(moving)
            gradient_hazard[moving_rows, observed_direction] -= 1.0 / h_observed
        gradient = gradient_hazard.T @ xt
        return loss, gradient.ravel()

    result = minimize(objective, init.ravel(), jac=True, method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": 2000, "ftol": 1e-10})
    if not result.success:
        result = minimize(objective, np.maximum(result.x, 0.0), jac=True,
                          method="SLSQP", bounds=bounds,
                          options={"maxiter": 1000, "ftol": 1e-9})
    if not result.success or not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Hawkes optimization failed: {result.message}")
    return HawkesModel(mode, float(decay_seconds), result.x.reshape(2, -1), names,
                       float(interval_seconds))


def score_hawkes(model: HawkesModel, features: ArrayLike, labels: ArrayLike,
                 interval_seconds: float) -> Scores:
    """Compute categorical held-out log loss and multiclass Brier score."""
    y = np.asarray(labels, dtype=int)
    p = predict_probabilities(model, features, interval_seconds)
    if y.shape != (len(p),) or np.any((y < 0) | (y > 2)):
        raise ValueError("labels must align with features and be in {0,1,2}")
    onehot = np.eye(3)[y]
    return Scores(float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()),
                  float(np.square(p - onehot).sum(axis=1).mean()), p)


def first_move_probabilities(model: HawkesModel, initial_features: ArrayLike,
                             horizon_seconds: float, step_seconds: float,
                             interval_seconds: float | None = None) -> NDArray[np.float64]:
    """Estimate first move direction by horizon with no future news arrivals.

    The decaying news state is held at its current contribution and decays
    exponentially. Completed price-event history is propagated only through
    expected event probabilities; no future labels or news are read.
    """
    state = np.asarray(initial_features, dtype=float).reshape(-1).copy()
    if horizon_seconds <= 0 or step_seconds <= 0 or state.size != model.coefficients.shape[1]:
        raise ValueError("invalid horizon, step, or initial features")
    dt = min(step_seconds, interval_seconds or step_seconds)
    survival = 1.0
    directional_mass = np.zeros(2)
    elapsed = 0.0
    while elapsed < horizon_seconds:
        delta = min(dt, horizon_seconds - elapsed)
        probs = predict_probabilities(model, state[None, :], delta)[0]
        directional_mass += survival * probs[1:]
        survival *= probs[0]
        # Along a first-event survival path, there are no future price events.
        if model.mode != "constant":
            pi = [model.feature_names.index("price_up"), model.feature_names.index("price_down")]
            decay = np.exp(-delta / model.decay_seconds)
            state[pi] *= decay
        ni = [i for i, name in enumerate(model.feature_names) if name.startswith("news")]
        state[ni] *= np.exp(-delta / model.decay_seconds)
        elapsed += delta
    return np.r_[survival, directional_mass]
