"""The vector-valued metrics of the cluster bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from copernicus.errors import ConfigurationError, EstimatorError, SettingsError
from copernicus.validity.bootstrap import VectorMetric, bootstrap_many

N = 60
N_BOOT = 100


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    score = rng.normal(size=N)
    gold = score + rng.normal(size=N)
    clusters = np.repeat(np.arange(N // 3), 3)
    return score, gold, clusters


def mean_score(score, gold):
    return float(np.mean(score))


def mean_gold(score, gold):
    return float(np.mean(gold))


def both_means(score, gold):
    return {"mean_score": float(np.mean(score)), "mean_gold": float(np.mean(gold))}


def test_a_vector_metric_gives_the_intervals_of_the_equivalent_scalar_metrics(data):
    score, gold, clusters = data
    scalar = bootstrap_many(
        {"mean_score": mean_score, "mean_gold": mean_gold},
        score,
        gold,
        clusters,
        n_boot=N_BOOT,
        seed=3,
    )
    vector = bootstrap_many(
        {},
        score,
        gold,
        clusters,
        n_boot=N_BOOT,
        seed=3,
        vector_metrics=(VectorMetric(("mean_score", "mean_gold"), both_means),),
    )
    assert vector == scalar


def test_a_vector_metric_is_called_once_per_replicate_and_beside_scalar_metrics(data):
    score, gold, clusters = data
    calls = []

    def counting(score, gold):
        calls.append(1)
        return both_means(score, gold)

    out = bootstrap_many(
        {"spread": lambda score, gold: float(np.std(score))},
        score,
        gold,
        clusters,
        n_boot=N_BOOT,
        vector_metrics=(VectorMetric(("mean_score", "mean_gold"), counting),),
    )
    assert len(calls) == N_BOOT
    assert list(out) == ["spread", "mean_score", "mean_gold"]


def test_a_vector_metric_may_return_more_values_than_it_names(data):
    score, gold, clusters = data
    out = bootstrap_many(
        {},
        score,
        gold,
        clusters,
        n_boot=N_BOOT,
        vector_metrics=(VectorMetric(("mean_gold",), both_means),),
    )
    assert list(out) == ["mean_gold"]


def test_a_non_finite_value_makes_only_its_own_interval_lose_that_replicate(data):
    score, gold, clusters = data
    seen = []

    def sometimes_nan(score, gold):
        seen.append(1)
        return {"first": float(np.mean(score)), "second": np.nan if len(seen) % 20 == 0 else 1.0}

    out = bootstrap_many(
        {},
        score,
        gold,
        clusters,
        n_boot=N_BOOT,
        vector_metrics=(VectorMetric(("first", "second"), sometimes_nan),),
    )
    assert out["first"].n_undefined == 0
    assert out["second"].n_undefined == N_BOOT // 20


def test_a_replicate_on_which_the_vector_metric_raises_is_undefined_for_every_name(data):
    score, gold, clusters = data
    seen = []

    def sometimes_undefined(score, gold):
        seen.append(1)
        if len(seen) % 20 == 0:
            raise EstimatorError("no range")
        return both_means(score, gold)

    out = bootstrap_many(
        {},
        score,
        gold,
        clusters,
        n_boot=N_BOOT,
        vector_metrics=(VectorMetric(("mean_score", "mean_gold"), sometimes_undefined),),
    )
    assert {ci.n_undefined for ci in out.values()} == {N_BOOT // 20}


def test_a_vector_metric_undefined_on_too_many_replicates_names_the_metric(data):
    score, gold, clusters = data

    def never(score, gold):
        raise EstimatorError("no range")

    with pytest.raises(EstimatorError, match=r"'mean_gold'.*no range"):
        bootstrap_many(
            {},
            score,
            gold,
            clusters,
            n_boot=N_BOOT,
            vector_metrics=(VectorMetric(("mean_gold",), never),),
        )


def test_a_vector_metric_that_omits_a_named_value_is_a_configuration_error(data):
    score, gold, clusters = data
    with pytest.raises(ConfigurationError, match="mean_gold"):
        bootstrap_many(
            {},
            score,
            gold,
            clusters,
            n_boot=5,
            vector_metrics=(VectorMetric(("mean_gold",), lambda score, gold: {"other": 1.0}),),
        )


@pytest.mark.parametrize(
    "vector",
    [
        VectorMetric(("mean_score",), both_means),
    ],
)
def test_names_must_be_distinct_from_the_other_metrics(data, vector):
    score, gold, clusters = data
    with pytest.raises(SettingsError, match="name"):
        bootstrap_many(
            {"mean_score": mean_score},
            score,
            gold,
            clusters,
            n_boot=5,
            vector_metrics=(vector,),
        )


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ((), "at least one name"),
        (("a", "a"), "more than once"),
        (["a", "b"], "tuple"),
        ("ab", "tuple"),
        (("a", 1), "strings"),
        (("a", ""), "strings"),
    ],
)
def test_vector_metric_names_are_a_non_empty_tuple_of_distinct_strings(names, message):
    with pytest.raises(SettingsError, match=message):
        VectorMetric(names, both_means)


def test_a_vector_metric_with_good_names_is_accepted():
    assert VectorMetric(("a", "b"), both_means).names == ("a", "b")


def test_without_any_metric_the_call_is_still_refused(data):
    score, gold, clusters = data
    with pytest.raises(SettingsError, match="at least one metric"):
        bootstrap_many({}, score, gold, clusters, n_boot=5)
