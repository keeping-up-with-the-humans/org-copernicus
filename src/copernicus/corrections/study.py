"""The repeated-study check: bias and coverage of the estimators over many human samples.

For a corpus whose gold values are known for every text, ``repeated_study`` draws many human
samples of each size the way ``draw_human_sample`` does (whole groups by simple random sampling,
``pi = m / G``), attaches the known values of the drawn texts with ``attach_draw``, and
estimates the mean and the long-minus-short difference (``long_short_split``) of the construct
with each estimator:

* ``raw``: the first score column plugged in for the construct, with its naive standard error
  (no gold value is used, so the estimate is the same in every draw);
* ``calibrated``: the least-squares line of the gold values on the first score column, fitted on
  the draw's labelled texts and plugged in for every text, with the naive standard error of the
  calibrated values (the consistent least-squares line; the original analysis divided a sample
  covariance by a population variance);
* ``ppi_identity``: ``mean_ppi`` and ``group_difference_ppi`` with the first score column and
  the study's ``cluster`` option: by default the standard error is clustered by group whenever
  a group of the draw holds more than one labelled text, so a draw of a single group, or one
  whose labelled texts on a side of the difference lie in a single group, is undefined;
  ``cluster=False`` treats the texts as independent;
* ``ppi_crossfit``: the cross-fitted pseudo-values (``pseudo_values`` with the ``crossfit``
  rectifier) plugged in, with their naive standard error.

The draw of size ``n`` with index ``d`` takes two integers from the generator
``rng_for(seed, "repeated_study", n, d)``: the first is the seed of the group draw
(``draw_groups``, the helper behind ``draw_human_sample``, so the draw's recorded seed reproduces
it), the second the seed of the draw's cross-fit folds. Sizes must be distinct. Against the
full-gold value of the corpus the result reports the bias, the share of 95 percent intervals
that contain it, and the mean standard error. A draw on which an estimator is undefined (it
raises ``EstimatorError``) is counted in ``n_undefined`` and left out of the summaries, the one
failure the study tolerates. A row (an estimator, size and estimand) undefined on more than 10
percent of its draws reports NaN summaries with the status ``undefined_share_exceeded`` and a
warning naming its last reason; the other rows keep their results (status ``ok``), and the study
stops only when every row is undefined. The cross-fitted rectifier needs about ``k + 1`` drawn
groups (six with its five inner folds), so small sizes often leave ``ppi_crossfit`` undefined.
No finite-population correction is applied, so at large sampling fractions the coverage of the
design-based estimators is conservative: their standard errors include the variance of the
machine term, which a fixed corpus does not have.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from copernicus.baseline.folds import rng_for
from copernicus.corrections._clustering import checked_cluster
from copernicus.corrections._estimate_inputs import complete_column
from copernicus.corrections._proxy import check_count
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.estimates import (
    Z,
    group_difference_ppi,
    long_short_split,
    mean_ppi,
    naive_se,
    plugin_value,
)
from copernicus.corrections.families._fitted import SEED_BOUND
from copernicus.corrections.families.calibration import fit_ols_line
from copernicus.corrections.families.pseudo import pseudo_values
from copernicus.corrections.sampling import attach_draw, draw_from_data
from copernicus.errors import EstimatorError, SettingsError

logger = logging.getLogger(__name__)

__all__ = ["ESTIMANDS", "ESTIMATORS", "RESULT_COLUMNS", "repeated_study"]

ESTIMATORS = ("raw", "calibrated", "ppi_identity", "ppi_crossfit")
ESTIMANDS = ("mean", "long_short_difference")
RESULT_COLUMNS = (
    "estimator",
    "size",
    "estimand",
    "truth",
    "estimate_mean",
    "bias",
    "coverage",
    "se_mean",
    "n_draws",
    "n_undefined",
    "status",
)
UNDEFINED_CEILING = 0.10
STATUS_OK = "ok"
STATUS_UNDEFINED = "undefined_share_exceeded"
SUMMARIES = ("estimate_mean", "bias", "coverage", "se_mean")
MIN_SIZE = 2
MIN_HALF = 2

Key = tuple[str, int, str]


@dataclass(frozen=True, eq=False)
class _Draw:
    """One draw's data and the columns its estimators share, each computed once."""

    data: CorrectionData
    split: np.ndarray
    crossfit_seed: int

    @cached_property
    def raw(self) -> np.ndarray:
        """The first score column."""
        return self.data.frame[self.data.score_columns[0]].to_numpy(dtype=float)

    @cached_property
    def calibrated(self) -> np.ndarray:
        """The first score column through the line fitted on the labelled texts."""
        labelled = self.data.labelled_mask
        line = fit_ols_line(
            self.raw[labelled], self.data.gold[labelled], self.data.score_columns[0]
        )
        return line.predict(self.raw)

    @cached_property
    def ppi_crossfit(self) -> np.ndarray:
        """The cross-fitted pseudo-values."""
        values = pseudo_values(self.data, rectifier="crossfit", seed=self.crossfit_seed)
        return values.column.to_numpy(dtype=float)


def _plugged_column(estimator: str, draw: _Draw) -> np.ndarray:
    """The column that a plug-in estimator (``raw``, ``calibrated``, ``ppi_crossfit``) uses."""
    if estimator == "raw":
        return draw.raw
    if estimator == "calibrated":
        return draw.calibrated
    return draw.ppi_crossfit


def _estimate(
    estimator: str, estimand: str, draw: _Draw, cluster: bool | None
) -> tuple[float, float]:
    """The point and the standard error of one estimator and estimand on one draw."""
    split = None if estimand == "mean" else draw.split
    if estimator == "ppi_identity":
        result = (
            mean_ppi(draw.data, cluster=cluster)
            if split is None
            else group_difference_ppi(draw.data, split, cluster=cluster)
        )
        return result.point, result.se
    column = _plugged_column(estimator, draw)
    return plugin_value(column, split), naive_se(column, split)


def _checked_sizes(sizes: Sequence[int], n_units: int) -> tuple[int, ...]:
    """Distinct sizes, each an integer between ``MIN_SIZE`` and the number of texts."""
    if isinstance(sizes, str | bytes) or not isinstance(sizes, Iterable):
        raise SettingsError(f"sizes must be a sequence of integers; got {sizes!r}")
    values = tuple(sizes)
    if not values:
        raise SettingsError("sizes must name at least one size")
    for size in values:
        if isinstance(size, bool) or not isinstance(size, numbers.Integral):
            raise SettingsError(f"each size must be an integer; got {size!r}")
        if not MIN_SIZE <= size <= n_units:
            raise SettingsError(
                f"each size must lie between {MIN_SIZE} and the {n_units} texts; got {size}"
            )
    if len(set(values)) != len(values):
        raise SettingsError(f"sizes must be distinct; got {values!r}")
    return tuple(int(size) for size in values)


def _checked_estimators(estimators: Sequence[str]) -> tuple[str, ...]:
    """Distinct estimator names among ``ESTIMATORS``."""
    values = () if isinstance(estimators, str) else tuple(estimators)
    if not values or len(set(values)) != len(values) or not set(values) <= set(ESTIMATORS):
        raise SettingsError(
            f"estimators must be distinct names among {', '.join(ESTIMATORS)}; got {estimators!r}"
        )
    return values


def _draw(
    data: CorrectionData,
    truth: pd.Series,
    size: int,
    seed: int,
    split: np.ndarray,
    index: int,
) -> _Draw:
    """The human sample of one size and index, with the known values of its texts."""
    draw_seed, crossfit_seed = (
        int(value)
        for value in rng_for(seed, "repeated_study", size, index).integers(SEED_BOUND, size=2)
    )
    sample = draw_from_data(data, size, seed=draw_seed)
    return _Draw(data=attach_draw(data, truth, sample), split=split, crossfit_seed=crossfit_seed)


@dataclass
class _Tally:
    """The defined estimates and the undefined draws of one estimator, size and estimand."""

    key: Key
    draws: int
    pairs: list[tuple[float, float]] = field(default_factory=list)
    undefined: int = 0
    last_reason: str = ""

    def record(self, draw: _Draw, cluster: bool | None) -> None:
        """Estimate on one draw, or count the draw as undefined and keep its reason."""
        name, _, estimand = self.key
        # The one failure the study tolerates: a draw on which the estimator is undefined is
        # counted; a row with too many of them is reported as undefined by its summary.
        try:
            self.pairs.append(_estimate(name, estimand, draw, cluster))
        except EstimatorError as error:
            self.undefined += 1
            self.last_reason = str(error)

    @property
    def exceeded(self) -> bool:
        """Whether more than 10 percent of the draws are undefined."""
        return self.undefined > UNDEFINED_CEILING * self.draws

    def summary(self, truth: float) -> dict[str, object]:
        """The result row: means over the defined draws and the share of covering intervals.

        A row undefined on more than 10 percent of its draws has NaN summaries.
        """
        estimator, size, estimand = self.key
        row: dict[str, object] = {
            "estimator": estimator,
            "size": size,
            "estimand": estimand,
            "truth": truth,
            **dict.fromkeys(SUMMARIES, float("nan")),
            "n_draws": self.draws,
            "n_undefined": self.undefined,
            "status": STATUS_UNDEFINED if self.exceeded else STATUS_OK,
        }
        if self.exceeded:
            return row
        points = np.array([point for point, _ in self.pairs])
        errors = np.array([se for _, se in self.pairs])
        covered = (points - Z * errors <= truth) & (truth <= points + Z * errors)
        row.update(
            estimate_mean=float(points.mean()),
            bias=float(points.mean()) - truth,
            coverage=float(covered.mean()),
            se_mean=float(errors.mean()),
        )
        return row


def repeated_study(
    data_full: CorrectionData,
    *,
    truth_gold: str,
    sizes: Sequence[int],
    draws: int = 300,
    seed: int = 42,
    estimators: Sequence[str] = ESTIMATORS,
    cluster: bool | None = None,
) -> pd.DataFrame:
    """Bias and coverage of each estimator over repeated human samples of a fully coded corpus.

    Args:
        data_full: The correction data of the corpus; its own gold column is replaced in every
            draw by the values of ``truth_gold`` on the drawn texts.
        truth_gold: A numeric column of the data's frame with a finite value for every text.
        sizes: The numbers of texts to code, each an integer between 2 and the number of texts
            (every text counts as one unit; whole groups are drawn).
        draws: Number of draws per size, at least 1.
        seed: Seed of the draws, an integer of at least 0.
        estimators: Distinct names among ``raw``, ``calibrated``, ``ppi_identity`` and
            ``ppi_crossfit``.
        cluster: The ``cluster`` option of ``mean_ppi`` and ``group_difference_ppi`` in
            ``ppi_identity``: ``None`` clusters the standard error by group when a group of the
            draw holds more than one labelled text, ``True`` and ``False`` (numpy booleans too)
            force the choice. A draw that clustering leaves undefined (one labelled group, or
            one on a side of the difference) is counted in ``n_undefined``.

    Returns:
        A frame with the columns ``estimator``, ``size``, ``estimand`` (``mean`` or
        ``long_short_difference``), ``truth`` (the full-gold value), ``estimate_mean``,
        ``bias``, ``coverage``, ``se_mean`` (over the defined draws; NaN for a row undefined on
        more than 10 percent of its draws), ``n_draws``, ``n_undefined`` and ``status`` (``ok``
        or ``undefined_share_exceeded``), one row per estimator, size and estimand in the
        orders given.

    Raises:
        SettingsError: If a size, ``draws``, ``seed``, ``estimators`` or ``cluster`` is outside
            its range.
        ConfigurationError: If ``truth_gold`` is not a complete numeric column of the data.
        EstimatorError: If the covariate does not split the texts into two halves of at least
            two, or every row is undefined on more than 10 percent of its draws (the message
            gives the last reason).
    """
    size_values = _checked_sizes(sizes, data_full.n)
    draws = check_count("draws", draws, 1)
    seed = check_count("seed", seed, 0)
    names = _checked_estimators(estimators)
    requested = checked_cluster(cluster)
    truth_values = complete_column(data_full, truth_gold, "truth_gold")
    split = long_short_split(data_full)
    if min(int(split.sum()), int((~split).sum())) < MIN_HALF:
        raise EstimatorError(
            "the long-short difference needs at least two texts on each side of the covariate's "
            f"median; the halves hold {int(split.sum())} and {int((~split).sum())}"
        )
    truths = {
        "mean": plugin_value(truth_values),
        "long_short_difference": plugin_value(truth_values, split),
    }
    truth = pd.Series(truth_values, index=data_full.ids)
    tallies = [
        _Tally(key=(name, size, estimand), draws=draws)
        for name in names
        for size in size_values
        for estimand in ESTIMANDS
    ]
    for size in size_values:
        active = [tally for tally in tallies if tally.key[1] == size]
        for index in range(draws):
            draw = _draw(data_full, truth, size, seed, split, index)
            for tally in active:
                tally.record(draw, requested)
        logger.info(
            "repeated study: %d draws of size %d, undefined draws %s",
            draws,
            size,
            {tally.key[0] + "/" + tally.key[2]: tally.undefined for tally in active},
        )
    _report_undefined(tallies)
    rows = [tally.summary(truths[tally.key[2]]) for tally in tallies]
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def _report_undefined(tallies: Sequence[_Tally]) -> None:
    """Warn about every row undefined on too many draws; refuse a study where all of them are.

    Raises:
        EstimatorError: If every row is undefined on more than 10 percent of its draws.
    """
    exceeded = [tally for tally in tallies if tally.exceeded]
    for tally in exceeded:
        name, size, estimand = tally.key
        logger.warning(
            "estimator %r is undefined on %d of the %d draws of size %d for the %s (more than "
            "10 percent), so its row is reported as %s; last reason: %s",
            name,
            tally.undefined,
            tally.draws,
            size,
            estimand,
            STATUS_UNDEFINED,
            tally.last_reason,
        )
    if len(exceeded) == len(tallies):
        raise EstimatorError(
            "every row of the study (estimator, size and estimand) is undefined on more than "
            f"10 percent of its draws; last reason: {tallies[-1].last_reason}"
        )
