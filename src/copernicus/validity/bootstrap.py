"""Group-clustered percentile bootstrap intervals for validity metrics."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

logger = logging.getLogger(__name__)

__all__ = ["BootstrapCI", "VectorMetric", "bootstrap_many", "cluster_bootstrap_ci"]

MAX_UNDEFINED_SHARE = 0.1
MIN_CLUSTERS = 2
SINGLE_METRIC_NAME = "metric"

Metric = Callable[..., float]
VectorFunction = Callable[..., Mapping[str, float]]
Evaluator = tuple[tuple[str, ...], Callable[[Sequence[np.ndarray]], dict[str, float]]]


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile interval of a metric over cluster-bootstrap replicates.

    Attributes:
        low: Lower percentile bound (``100 * alpha / 2``).
        high: Upper percentile bound (``100 * (1 - alpha / 2)``).
        n_boot: Number of replicates drawn.
        n_undefined: Replicates on which the metric was undefined; the bounds use the
            other ``n_boot - n_undefined`` replicates.
    """

    low: float
    high: float
    n_boot: int
    n_undefined: int


@dataclass(frozen=True)
class VectorMetric:
    """A metric that returns several named values from one evaluation per replicate.

    It serves metrics whose values come out of one computation (the four score errors of
    the validity panel), so that the computation runs once per replicate rather than once
    per value. Each name becomes an interval of its own.

    Attributes:
        names: Names of the values that get an interval; distinct, and distinct from the
            names of the other metrics of the call.
        function: Called as ``function(score_b, gold_b, *extra_b)`` on each replicate and
            returning a mapping that holds a value for every name (further keys are
            ignored). A replicate on which it raises ``EstimatorError`` or ``ValueError``
            is undefined for every name; a non-finite value is undefined for its name only.
    """

    names: tuple[str, ...]
    function: VectorFunction

    def __post_init__(self) -> None:
        """Refuse names that are not a non-empty tuple of distinct, non-empty strings.

        Raises:
            SettingsError: If ``names`` is not such a tuple.
        """
        if not isinstance(self.names, tuple):
            raise SettingsError(
                f"a vector metric's names must be a tuple of strings; got {self.names!r}"
            )
        if not self.names:
            raise SettingsError("a vector metric needs at least one name")
        if not all(isinstance(name, str) and name for name in self.names):
            raise SettingsError(
                f"a vector metric's names must be non-empty strings; got {self.names!r}"
            )
        repeated = sorted({name for name in self.names if self.names.count(name) > 1})
        if repeated:
            raise SettingsError(f"vector metric name(s) {repeated} are given more than once")


def _validate_settings(
    metrics: Mapping[str, Metric], vector_metrics: Sequence[VectorMetric], n_boot: int, alpha: float
) -> None:
    if n_boot < 1:
        raise SettingsError(f"n_boot must be at least 1; got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise SettingsError(f"alpha must lie in (0, 1); got {alpha}")
    if not metrics and not vector_metrics:
        raise SettingsError("at least one metric is required")
    names = [*metrics, *(name for vector in vector_metrics for name in vector.names)]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise SettingsError(f"metric name(s) {repeated} are given more than once")


def _validate_arrays(arrays: tuple[np.ndarray, ...], clusters: np.ndarray) -> None:
    if clusters.ndim != 1:
        raise ConfigurationError(
            f"clusters must be one-dimensional; got {clusters.ndim} dimensions"
        )
    shapes = [values.shape for values in arrays]
    if any(len(shape) != 1 or shape[0] != clusters.shape[0] for shape in shapes):
        raise ConfigurationError(
            f"score, gold, the extra arrays and clusters must be one-dimensional and of the "
            f"same length; got shapes {shapes} and {clusters.shape[0]} clusters"
        )
    if np.unique(clusters).size < MIN_CLUSTERS:
        raise ConfigurationError("a cluster bootstrap needs at least two clusters")


def _resampled_rows(clusters: np.ndarray, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Row indices of each replicate: whole clusters drawn with replacement.

    The clusters are the sorted unique ids. Each replicate draws as many clusters as exist
    from ``np.random.default_rng(seed)``, with the same draws as sampling the sorted ids
    themselves, and concatenates their member rows in drawn order.
    """
    unique, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.choice(unique.size, size=unique.size, replace=True)
        yield np.concatenate([members[index] for index in chosen])


def _scalar_evaluator(name: str, metric: Metric) -> Evaluator:
    """Wrap a scalar metric as an evaluator of one named value."""

    def evaluate(resampled: Sequence[np.ndarray]) -> dict[str, float]:
        return {name: float(metric(*resampled))}

    return (name,), evaluate


def _vector_evaluator(vector: VectorMetric) -> Evaluator:
    """Wrap a vector metric as an evaluator of its named values."""

    def evaluate(resampled: Sequence[np.ndarray]) -> dict[str, float]:
        returned = vector.function(*resampled)
        absent = [name for name in vector.names if name not in returned]
        if absent:
            raise ConfigurationError(
                f"the vector metric for {list(vector.names)} returned no value for {absent}"
            )
        return {name: float(returned[name]) for name in vector.names}

    return vector.names, evaluate


def bootstrap_many(
    metrics: Mapping[str, Metric],
    score: np.ndarray,
    gold: np.ndarray,
    clusters: np.ndarray,
    *extra: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    vector_metrics: Sequence[VectorMetric] = (),
) -> dict[str, BootstrapCI]:
    """Percentile bootstrap intervals for several metrics on the same resamples.

    Whole clusters (groups of texts sharing an id) are drawn with replacement, and every
    metric is evaluated on each replicate as ``metric(score_b, gold_b, *extra_b)``. A
    replicate on which a metric returns a non-finite value or raises ``EstimatorError`` or
    ``ValueError`` is undefined for that metric: it is counted in ``n_undefined`` and left
    out of that metric's interval. Any other exception propagates.

    Args:
        metrics: Name to metric function, called with the resampled arrays.
        score: Score values.
        gold: Gold values, of the same length.
        clusters: One cluster id per row (a one-dimensional array of sortable ids).
        *extra: Further arrays of the same length, resampled with the rows and passed to
            every metric after ``gold`` (a covariate, for example).
        n_boot: Number of replicates.
        alpha: Total tail probability; the interval is the ``alpha / 2`` and
            ``1 - alpha / 2`` percentiles.
        seed: Seed of the generator.
        vector_metrics: Metrics that return several named values from one call per
            replicate (see `VectorMetric`); their names get intervals beside those
            of ``metrics``, on the same resamples.

    Returns:
        One interval per metric name, the names of ``metrics`` first and then those of the
        vector metrics.

    Raises:
        SettingsError: If ``n_boot`` is below 1, ``alpha`` is outside (0, 1), no metric
            is given, or a metric name is missing or repeated.
        ConfigurationError: If ``clusters`` is not one-dimensional, the arrays differ in
            length, there are fewer than two clusters, or a vector metric returns no value
            for one of its names.
        EstimatorError: If a metric is undefined on more than 10 percent of the replicates.
    """
    _validate_settings(metrics, vector_metrics, n_boot, alpha)
    arrays = tuple(np.asarray(values) for values in (score, gold, *extra))
    cluster_ids = np.asarray(clusters)
    _validate_arrays(arrays, cluster_ids)
    evaluators = [_scalar_evaluator(name, metric) for name, metric in metrics.items()]
    evaluators += [_vector_evaluator(vector) for vector in vector_metrics]
    estimates: dict[str, list[float]] = {name: [] for names, _ in evaluators for name in names}
    last_errors: dict[str, Exception] = {}
    for rows in _resampled_rows(cluster_ids, n_boot, seed):
        resampled = [values[rows] for values in arrays]
        for names, evaluate in evaluators:
            try:
                values = evaluate(resampled)
            except (EstimatorError, ValueError) as error:
                last_errors.update(dict.fromkeys(names, error))
                continue
            for name, value in values.items():
                if np.isfinite(value):
                    estimates[name].append(value)
    return {
        name: _interval(name, defined, n_boot, alpha, last_errors.get(name))
        for name, defined in estimates.items()
    }


def _interval(
    name: str, defined: list[float], n_boot: int, alpha: float, last_error: Exception | None
) -> BootstrapCI:
    """The percentile interval of the defined replicates of one metric.

    Raises:
        EstimatorError: If more than 10 percent of the replicates are undefined.
    """
    n_undefined = n_boot - len(defined)
    if n_undefined > MAX_UNDEFINED_SHARE * n_boot:
        if last_error is None:
            cause = "the undefined replicates returned non-finite values and raised nothing"
        else:
            cause = f"last error: {last_error}"
        raise EstimatorError(
            f"bootstrap metric {name!r}: {n_undefined} of {n_boot} replicates are "
            f"undefined, above the 10 percent ceiling ({cause})"
        ) from last_error
    if n_undefined > 0:
        logger.info("bootstrap metric %r: %d of %d replicates undefined", name, n_undefined, n_boot)
    values = np.asarray(defined)
    return BootstrapCI(
        low=float(np.percentile(values, 100.0 * alpha / 2.0)),
        high=float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0))),
        n_boot=n_boot,
        n_undefined=n_undefined,
    )


def cluster_bootstrap_ci(
    metric: Metric,
    score: np.ndarray,
    gold: np.ndarray,
    clusters: np.ndarray,
    *extra: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    name: str = SINGLE_METRIC_NAME,
) -> BootstrapCI:
    """Percentile bootstrap interval of one metric, resampling whole clusters.

    The single-metric case of `bootstrap_many`; see there for the resampling, the
    treatment of undefined replicates and the errors.

    Args:
        metric: Function called as ``metric(score_b, gold_b, *extra_b)`` on each replicate.
        score: Score values.
        gold: Gold values, of the same length.
        clusters: One cluster id per row.
        *extra: Further arrays of the same length, resampled with the rows.
        n_boot: Number of replicates.
        alpha: Total tail probability.
        seed: Seed of the generator.
        name: Name of the metric, used in the error raised when too many replicates are
            undefined.

    Returns:
        The interval with the number of replicates and the number undefined.
    """
    result = bootstrap_many(
        {name: metric},
        score,
        gold,
        clusters,
        *extra,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    return result[name]
