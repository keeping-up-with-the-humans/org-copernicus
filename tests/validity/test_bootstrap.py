"""Group-clustered bootstrap intervals."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.stats import spearmanr

from copernicus.errors import ConfigurationError, EstimatorError, SettingsError
from copernicus.validity.anatomy import score_errors
from copernicus.validity.bootstrap import BootstrapCI, bootstrap_many, cluster_bootstrap_ci

N_TEXTS = 200
N_CLUSTERS = 40


def _spearman(score: np.ndarray, gold: np.ndarray) -> float:
    return float(spearmanr(score, gold).statistic)


def _pearson(score: np.ndarray, gold: np.ndarray) -> float:
    return float(np.corrcoef(score, gold)[0, 1])


@pytest.fixture
def clustered():
    rng = np.random.default_rng(7)
    gold = rng.normal(size=N_TEXTS)
    score = gold + rng.normal(scale=0.5, size=N_TEXTS)
    clusters = np.repeat(np.arange(N_CLUSTERS), N_TEXTS // N_CLUSTERS)
    return score, gold, clusters


def test_interval_contains_the_point_estimate_and_is_narrow(clustered):
    score, gold, clusters = clustered
    ci = cluster_bootstrap_ci(_spearman, score, gold, clusters)
    assert ci.low <= _spearman(score, gold) <= ci.high
    assert ci.high - ci.low < 0.3
    assert (ci.n_boot, ci.n_undefined) == (1000, 0)


def test_interval_reproduces_the_row_by_row_percentile_procedure(clustered):
    score, gold, clusters = clustered
    unique = np.unique(clusters)
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(300):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([members[cluster] for cluster in chosen])
        estimates.append(_pearson(score[rows], gold[rows]))
    expected = (np.percentile(estimates, 2.5), np.percentile(estimates, 97.5))
    ci = cluster_bootstrap_ci(_pearson, score, gold, clusters, n_boot=300)
    assert (ci.low, ci.high) == pytest.approx(expected, abs=1e-12)


def test_clustering_widens_the_interval_when_texts_within_a_cluster_agree():
    rng = np.random.default_rng(3)
    cluster_effect = rng.normal(size=N_CLUSTERS)
    clusters = np.repeat(np.arange(N_CLUSTERS), 5)
    gold = np.repeat(cluster_effect, 5) + rng.normal(scale=0.2, size=N_TEXTS)
    score = gold + rng.normal(scale=1.0, size=N_TEXTS)
    by_cluster = cluster_bootstrap_ci(_pearson, score, gold, clusters, n_boot=500)
    by_text = cluster_bootstrap_ci(_pearson, score, gold, np.arange(N_TEXTS), n_boot=500)
    assert by_cluster.high - by_cluster.low > by_text.high - by_text.low


def test_a_metric_that_is_nan_on_every_replicate_raises(clustered):
    score, gold, clusters = clustered
    with pytest.raises(EstimatorError, match=r"1000 of 1000"):
        cluster_bootstrap_ci(lambda s, g: float("nan"), score, gold, clusters)


def test_infinite_values_count_as_undefined(clustered):
    score, gold, clusters = clustered
    with pytest.raises(EstimatorError, match="undefined"):
        cluster_bootstrap_ci(lambda s, g: float("inf"), score, gold, clusters, n_boot=50)


def _fails_when_the_first_row_is_low(fraction: float, error: type[Exception]):
    """A metric that fails on about ``fraction`` of the resamples (first drawn cluster low)."""
    threshold = fraction * N_TEXTS

    def metric(score: np.ndarray, gold: np.ndarray) -> float:
        if score[0] < threshold:
            raise error("degenerate resample")
        return _pearson(score, gold)

    return metric


def test_more_than_ten_percent_undefined_raises_and_names_the_counts():
    score = np.arange(float(N_TEXTS))
    gold = score + np.random.default_rng(0).normal(size=N_TEXTS)
    clusters = np.repeat(np.arange(N_CLUSTERS), N_TEXTS // N_CLUSTERS)
    metric = _fails_when_the_first_row_is_low(0.15, EstimatorError)
    with pytest.raises(EstimatorError, match=r"\d+ of 400 .*undefined") as raised:
        cluster_bootstrap_ci(metric, score, gold, clusters, n_boot=400)
    undefined = int(str(raised.value).split(" of 400")[0].split()[-1])
    assert undefined > 40


def test_the_ceiling_error_quotes_and_chains_the_last_tolerated_error():
    score = np.arange(float(N_TEXTS))
    gold = score + np.random.default_rng(0).normal(size=N_TEXTS)
    clusters = np.repeat(np.arange(N_CLUSTERS), N_TEXTS // N_CLUSTERS)
    metric = _fails_when_the_first_row_is_low(0.15, ValueError)
    with pytest.raises(EstimatorError, match="'tone'.*degenerate resample") as raised:
        cluster_bootstrap_ci(metric, score, gold, clusters, n_boot=400, name="tone")
    assert isinstance(raised.value.__cause__, ValueError)
    assert "degenerate resample" in str(raised.value.__cause__)


def test_the_ceiling_error_says_so_when_the_replicates_were_non_finite(clustered):
    score, gold, clusters = clustered
    with pytest.raises(EstimatorError, match="non-finite.*raised nothing") as raised:
        cluster_bootstrap_ci(lambda s, g: float("nan"), score, gold, clusters, n_boot=20)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("error", [EstimatorError, ValueError])
def test_fewer_than_ten_percent_undefined_is_counted_not_dropped(error):
    score = np.arange(float(N_TEXTS))
    gold = score + np.random.default_rng(0).normal(size=N_TEXTS)
    clusters = np.repeat(np.arange(N_CLUSTERS), N_TEXTS // N_CLUSTERS)
    metric = _fails_when_the_first_row_is_low(0.05, error)
    ci = cluster_bootstrap_ci(metric, score, gold, clusters, n_boot=400)
    assert 0 < ci.n_undefined <= 40
    assert ci.n_boot == 400


def test_other_exceptions_propagate(clustered):
    score, gold, clusters = clustered

    def broken(s: np.ndarray, g: np.ndarray) -> float:
        raise RuntimeError("bug in the metric")

    with pytest.raises(RuntimeError, match="bug in the metric"):
        cluster_bootstrap_ci(broken, score, gold, clusters, n_boot=5)


def test_bootstrap_many_returns_reproducible_intervals_for_two_metrics(clustered):
    score, gold, clusters = clustered
    metrics = {"spearman": _spearman, "pearson": _pearson}
    first = bootstrap_many(metrics, score, gold, clusters, n_boot=200)
    second = bootstrap_many(metrics, score, gold, clusters, n_boot=200)
    assert set(first) == {"spearman", "pearson"}
    assert first == second
    assert all(isinstance(value, BootstrapCI) for value in first.values())
    assert first["spearman"] != first["pearson"]


def test_bootstrap_many_with_one_metric_equals_cluster_bootstrap_ci(clustered):
    score, gold, clusters = clustered
    single = cluster_bootstrap_ci(_spearman, score, gold, clusters, n_boot=250, seed=5)
    many = bootstrap_many({"rho": _spearman}, score, gold, clusters, n_boot=250, seed=5)
    assert many["rho"] == single


def test_bootstrap_many_shares_one_row_draw_across_metrics(clustered):
    score, gold, clusters = clustered
    seen: dict[str, list[tuple[float, ...]]] = {"a": [], "b": []}

    def recorder(name: str):
        def metric(s: np.ndarray, g: np.ndarray) -> float:
            seen[name].append(tuple(s[:5]))
            return _pearson(s, g)

        return metric

    bootstrap_many({"a": recorder("a"), "b": recorder("b")}, score, gold, clusters, n_boot=20)
    assert seen["a"] == seen["b"] and len(seen["a"]) == 20


def test_undefined_replicates_are_counted_per_metric_and_the_failing_metric_is_named():
    score = np.arange(float(N_TEXTS))
    gold = score + np.random.default_rng(0).normal(size=N_TEXTS)
    clusters = np.repeat(np.arange(N_CLUSTERS), N_TEXTS // N_CLUSTERS)
    tolerable = _fails_when_the_first_row_is_low(0.05, EstimatorError)
    intolerable = _fails_when_the_first_row_is_low(0.15, EstimatorError)
    result = bootstrap_many(
        {"fine": _pearson, "some": tolerable}, score, gold, clusters, n_boot=400
    )
    assert result["fine"].n_undefined == 0 and result["some"].n_undefined > 0
    with pytest.raises(EstimatorError, match="'too_many'"):
        bootstrap_many(
            {"fine": _pearson, "too_many": intolerable}, score, gold, clusters, n_boot=400
        )


def test_extra_arrays_are_resampled_alongside_score_and_gold(clustered):
    score, gold, clusters = clustered
    covariate = 2.0 * score + 1.0
    checked = []

    def metric(s: np.ndarray, g: np.ndarray, c: np.ndarray) -> float:
        assert c.shape == s.shape == g.shape
        assert np.array_equal(c, 2.0 * s + 1.0)
        checked.append(len(s))
        return _pearson(s, g)

    cluster_bootstrap_ci(metric, score, gold, clusters, covariate, n_boot=30)
    assert len(checked) == 30


def test_covariate_linked_error_interval_covers_the_planted_slope():
    rng = np.random.default_rng(11)
    clusters = np.repeat(np.arange(N_CLUSTERS), N_TEXTS // N_CLUSTERS)
    covariate = rng.normal(size=N_TEXTS)
    gold = rng.uniform(size=N_TEXTS)
    score = gold + 0.1 * covariate + rng.normal(scale=0.05, size=N_TEXTS)

    def raw_slope(s: np.ndarray, g: np.ndarray, c: np.ndarray) -> float:
        return score_errors(s, g, c, covariate_scale="raw").covariate_linked_error

    ci = cluster_bootstrap_ci(raw_slope, score, gold, clusters, covariate, n_boot=300)
    point = raw_slope(score, gold, covariate)
    assert ci.low <= point <= ci.high
    assert ci.high - ci.low < 0.1


def test_string_cluster_ids_are_supported(clustered):
    score, gold, clusters = clustered
    labels = np.array([f"group_{value:02d}" for value in clusters])
    numeric = cluster_bootstrap_ci(_pearson, score, gold, clusters, n_boot=100)
    text = cluster_bootstrap_ci(_pearson, score, gold, labels, n_boot=100)
    assert text == numeric


@pytest.mark.parametrize(
    ("n_boot", "alpha"), [(0, 0.05), (-3, 0.05), (10, 0.0), (10, 1.0), (10, float("nan"))]
)
def test_settings_outside_their_range_raise_settings_error(clustered, n_boot, alpha):
    score, gold, clusters = clustered
    with pytest.raises(SettingsError):
        cluster_bootstrap_ci(_pearson, score, gold, clusters, n_boot=n_boot, alpha=alpha)


def test_an_empty_metric_mapping_is_refused(clustered):
    score, gold, clusters = clustered
    with pytest.raises(SettingsError, match="metric"):
        bootstrap_many({}, score, gold, clusters)


def test_two_dimensional_clusters_are_refused(clustered):
    score, gold, clusters = clustered
    with pytest.raises(ConfigurationError, match="one-dimensional"):
        cluster_bootstrap_ci(_pearson, score, gold, clusters.reshape(-1, 1))


def test_length_mismatches_are_refused(clustered):
    score, gold, clusters = clustered
    with pytest.raises(ConfigurationError, match="length"):
        cluster_bootstrap_ci(_pearson, score, gold, clusters[:-1])
    with pytest.raises(ConfigurationError, match="length"):
        cluster_bootstrap_ci(_pearson, score, gold[:-1], clusters)
    with pytest.raises(ConfigurationError, match="length"):
        cluster_bootstrap_ci(_pearson, score, gold, clusters, np.arange(N_TEXTS - 1.0))


def test_a_single_cluster_is_refused(clustered):
    score, gold, _ = clustered
    with pytest.raises(ConfigurationError, match="two clusters"):
        cluster_bootstrap_ci(_pearson, score, gold, np.zeros(N_TEXTS, dtype=int))


def test_bootstrap_ci_is_frozen():
    ci = BootstrapCI(low=0.1, high=0.2, n_boot=10, n_undefined=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ci.low = 0.0  # type: ignore[misc]
