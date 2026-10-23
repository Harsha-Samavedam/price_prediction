import numpy as np

from claim_reaction.hawkes import (
    fit_hawkes,
    fit_from_features,
    first_move_probabilities,
    history_features,
    predict_probabilities,
    score_hawkes,
)


def test_history_uses_only_news_before_start_and_completed_price_bins():
    starts = np.array([0.0, 10.0, 20.0, 30.0])
    # At exactly 10, news is not yet available to the bucket starting at 10.
    news_times = np.array([0.0, 10.0, 19.0])
    marks = np.array([1.0, -1.0, 1.0])
    labels = np.array([1, 2, 1, 0])
    x, names = history_features(starts, news_times, marks, labels, 10.0, 10.0)
    assert names == ("constant", "news_up", "news_down", "price_up", "price_down")
    assert x[0, 1:3].sum() == 0
    assert np.isclose(x[1, names.index("news_up")], np.exp(-1))
    assert x[1, names.index("news_down")] == 0  # news at bucket start excluded
    assert x[1, names.index("price_up")] > 0  # first price bucket completed
    assert x[1, names.index("price_down")] == 0
    assert x[2, names.index("price_down")] > 0  # second bucket now completed


def test_fit_uses_only_selected_training_labels_and_nonnegative_coefficients():
    starts = np.arange(20, dtype=float) * 60
    news_times = np.arange(1, 20, 2, dtype=float) * 60 - 30
    marks = np.where(np.arange(len(news_times)) % 2, -1.0, 1.0)
    labels = np.where(np.arange(20) % 2, 1, 2)
    train = np.arange(12)
    a = fit_hawkes(starts, news_times, marks, labels, 60, 300, train_indices=train)
    changed = labels.copy()
    changed[12:] = 0 if np.any(changed[12:] != 0) else 1
    b = fit_hawkes(starts, news_times, marks, changed, 60, 300, train_indices=train)
    np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-10)
    assert np.all(a.coefficients >= 0)


def test_discrete_likelihood_prefers_relevant_direction_and_probabilities_sum():
    n = 160
    starts = np.arange(n, dtype=float) * 60
    labels = np.zeros(n, dtype=int)
    # Repeated marks precede up moves; baseline alone cannot encode this.
    news_times = starts[:-1] + 20
    marks = np.ones(n - 1)
    labels[1::3] = 1
    labels[2::7] = 2
    x, _ = history_features(starts, news_times, marks, labels, 60, 300, "signed_news")
    model = fit_hawkes(starts, news_times, marks, labels, 60, 300, "signed_news")
    probs = predict_probabilities(model, x, 60)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))
    assert model.coefficients.shape == (2, x.shape[1])
    signed_loss = score_hawkes(model, x, labels, 60).log_loss
    baseline_x, _ = history_features(starts, news_times, marks, labels, 60, 300, "constant")
    baseline = fit_hawkes(starts, news_times, marks, labels, 60, 300, "constant")
    assert signed_loss < score_hawkes(baseline, baseline_x, labels, 60).log_loss


def test_first_move_distribution_is_valid():
    starts = np.arange(4, dtype=float) * 60
    labels = np.array([0, 1, 2, 0])
    x, names = history_features(starts, [], [], labels, 60, 300, "price_history")
    from claim_reaction.hawkes import HawkesModel
    model = HawkesModel("price_history", 300, np.full((2, len(names)), 0.001), names, 60)
    p = first_move_probabilities(model, x[-1], 600, 60)
    assert p.shape == (3,)
    np.testing.assert_allclose(p.sum(), 1)
    assert np.all(p >= 0)


def test_streaming_features_match_direct_exponential_formula():
    rng = np.random.default_rng(42)
    starts = np.arange(250, dtype=float) * 17.0
    news_times = rng.uniform(-100, starts[-1] + 30, size=400)
    marks = rng.uniform(-1, 1, size=len(news_times))
    labels = rng.integers(0, 3, size=len(starts))
    for mode in ("constant", "price_history", "timestamp_only", "signed_news"):
        actual, names = history_features(starts, news_times, marks, labels, 17, 93, mode)
        expected = np.zeros_like(actual)
        expected[:, 0] = 1
        for i, start in enumerate(starts):
            chosen = news_times < start
            weights = np.exp(-(start - news_times[chosen]) / 93)
            selected_marks = marks[chosen]
            offset = 1
            if mode == "timestamp_only":
                expected[i, offset] = weights.sum(); offset += 1
            elif mode == "signed_news":
                expected[i, offset] = weights[selected_marks > 0] @ np.abs(selected_marks[selected_marks > 0])
                expected[i, offset + 1] = weights[selected_marks < 0] @ np.abs(selected_marks[selected_marks < 0])
                offset += 2
            if mode != "constant":
                prior = starts + 17 <= start
                pw = np.exp(-(start - (starts[prior] + 17)) / 93)
                py = labels[prior]
                expected[i, offset] = pw[py == 1].sum()
                expected[i, offset + 1] = pw[py == 2].sum()
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert len(names) == actual.shape[1]


def test_fit_from_features_pools_independently_built_market_rows():
    starts_a = np.arange(30, dtype=float) * 60
    starts_b = np.arange(30, dtype=float) * 60
    labels_a = np.where(np.arange(30) % 3 == 0, 1, 0)
    labels_b = np.where(np.arange(30) % 4 == 0, 2, 0)
    xa, names = history_features(starts_a, starts_a[::3] + 5, np.ones(10), labels_a, 60, 240)
    xb, names_b = history_features(starts_b, starts_b[::4] + 5, -np.ones(8), labels_b, 60, 240)
    assert names == names_b
    pooled = fit_from_features(np.vstack([xa, xb]), np.r_[labels_a, labels_b], 60,
                               "signed_news", 240, names)
    assert pooled.coefficients.shape == (2, len(names))
    assert np.all(pooled.coefficients >= 0)
    # Existing single-market API is the same feature-fit operation.
    wrapped = fit_hawkes(starts_a, starts_a[::3] + 5, np.ones(10), labels_a, 60, 240)
    direct = fit_from_features(xa, labels_a, 60, "signed_news", 240, names)
    np.testing.assert_allclose(wrapped.coefficients, direct.coefficients, atol=1e-10)


def test_constant_model_reproduces_observed_category_frequencies():
    labels = np.array([0] * 70 + [1] * 20 + [2] * 10)
    features = np.ones((len(labels), 1))
    model = fit_from_features(features, labels, 300, "constant", 1200, ("constant",))
    probabilities = predict_probabilities(model, features[:1], 300)[0]
    np.testing.assert_allclose(probabilities, [0.7, 0.2, 0.1], atol=1e-5)


def test_per_bin_hazard_fit_is_stable_across_bucket_durations():
    labels = np.array([0, 0, 1, 0, 2, 0, 0, 1, 0, 2] * 8)
    features = np.ones((len(labels), 1))
    names = ("constant",)
    models = [fit_from_features(features, labels, interval, "constant", 3600, names)
              for interval in (300, 1800, 10800)]
    for model in models[1:]:
        np.testing.assert_allclose(model.coefficients, models[0].coefficients, atol=1e-8)
        np.testing.assert_allclose(predict_probabilities(model, features[:1], model.interval_seconds),
                                   predict_probabilities(models[0], features[:1], 300), atol=1e-8)
