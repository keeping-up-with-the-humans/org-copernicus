"""The repeated-study check: bias and coverage of the estimators over many human samples.

For a corpus whose gold values are known for every text, ``repeated_study`` draws many human
samples of each size the way ``draw_human_sample`` does (whole groups by simple random sampling,
``pi = m / G``), attaches the known values of the drawn texts with ``attach_draw``, and
estimates the mean of the construct and the long-short difference, the texts at or above the
covariate's median against the texts below it (``long_short_split``), with each estimator:

* ``raw``: the first score column plugged in for the construct, with its naive standard error
  (no gold value is used, so the estimate is the same in every draw);
* ``calibrated``: the least-squares line of the gold values on the first score column, fitted on
  the draw's labelled texts and plugged in for every text, with the naive standard error of the
  calibrated values (the consistent least-squares line; the original analysis divided a sample
  covariance by a population variance);
* ``ppi_identity``: ``mean_ppi`` and ``group_difference_ppi`` with the first score column and
  the study's ``cluster`` option. A draw with fewer than two labelled texts, or for the
  difference fewer than two on a side, is undefined under every setting. By default the
  standard error is clustered by group whenever a group of the draw holds more than one
  labelled text, so a draw of a single group, or one whose labelled texts on a side of the
  difference lie in a single group, is undefined as well; ``cluster=False`` treats the texts
  as independent;
* ``ppi_crossfit``: the cross-fitted pseudo-values (``pseudo_values`` with the ``crossfit``
  rectifier) plugged in, with the standard error of that column under the same choice as
  ``ppi_identity``: by default the contributions of the texts are summed within each group
  whenever a group of the draw holds more than one labelled text (the formula of
  ``_clustering``), and ``cluster=False`` gives the naive standard error. Before the column
  is cross-fitted, a draw is refused as ``ppi_identity`` refuses it, with its messages
  (``crossfit_clustering``): fewer than two labelled texts, or for the difference fewer than
  two on a side (a side without labelled texts would keep its uncorrected predictions), under
  every setting, and, when the draw clusters, fewer than two groups among the texts or the
  labelled texts. The column covers every text, so a side of the difference whose labelled
  texts lie in one group still adds the variation of its other groups, and such a draw is not
  refused; a side whose texts all lie in one group of the corpus would add nothing, so the
  difference is then refused on every draw that clusters (``require_groups_on_each_side``).

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
No finite-population correction is applied, and the standard errors of the design-based
estimators include the variance of the machine term, the mean of the predictions over all
texts. For ``ppi_identity``, whose score is fixed before the draw, that mean does not vary over
draws from a fixed corpus; ``ppi_crossfit`` learns its prediction on each draw, and that
prediction varies. This can make the intervals conservative at large sampling fractions, but
it does not guarantee coverage. When the groups differ in size, a whole-group draw can miss a
group that holds a large share of the texts in many draws: the mean and the difference of
``ppi_identity`` are then ratio estimators with a bias (see ``mean_ppi``), and the normal
intervals of ``ppi_identity`` and ``ppi_crossfit`` can cover the full-gold value far less often
than 95 percent although every draw is defined, at large sampling fractions too. A size whose
number of drawn texts has a coefficient of variation over its draws (``text_count_variation``)
above ``TEXT_COUNT_VARIATION_THRESHOLD`` gets one warning, as a draw of ``draw_human_sample``
does; its bound covers neither a row's ``bias``, a mean over the defined draws of this run,
nor its coverage (see ``repeated_study``). The status ``ok`` certifies only that the share of
undefined draws is within the limit, not that the intervals cover; read ``coverage``. A row of
``ppi_crossfit`` counts in ``n_rank_deficient_folds`` the inner folds, over its defined draws,
whose design ``[1, score, covariate]`` had a numerical rank below 3 on their labelled training
rows, as ``numpy.linalg.lstsq`` with ``rcond=None`` returns it, where the fold kept the
coefficients that ``lstsq`` returns (see ``pseudo_values``); the other estimators fit no folds
and count 0. Every row counts in ``n_se_exactly_zero`` its defined draws whose standard error
is exactly 0. The warnings that the calls of the draws would log one by one for the cases of
``_open_cases``, such as a fold of that kind or a standard error of exactly 0, are counted
instead: the study logs one warning for each size whose draws met them, with the count of each
case, a cross-fit counted once for a draw that serves both estimands. The work on one draw lives
in ``_study_draws``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd

from copernicus.corrections._clustering import checked_cluster
from copernicus.corrections._estimate_inputs import complete_column
from copernicus.corrections._open_cases import gathered, summary
from copernicus.corrections._proxy import check_count
from copernicus.corrections._stopped_runs import report_stopped
from copernicus.corrections._study_checks import checked_estimators, checked_sizes
from copernicus.corrections._study_draws import (
    StudyDraw,
    crossfit_se,
    draw_estimate,
    plugged_column,
    study_draw,
)
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.estimates import Z, long_short_split, plugin_value
from copernicus.corrections.group_sizes import warn_if_text_counts_vary
from copernicus.corrections.sampling import groups_for_quota
from copernicus.errors import EstimatorError

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
    "n_rank_deficient_folds",
    "n_se_exactly_zero",
)
UNDEFINED_CEILING = 0.10
STATUS_OK = "ok"
STATUS_UNDEFINED = "undefined_share_exceeded"
SUMMARIES = ("estimate_mean", "bias", "coverage", "se_mean")
MIN_SIDE = 2
# The old names in this module of the helpers that _study_draws holds, which code imports from here.
_Draw = StudyDraw
_draw = study_draw
_estimate = draw_estimate
_plugged_column = plugged_column
_crossfit_se = crossfit_se

Key = tuple[str, int, str]


@dataclass
class _Tally:
    """The defined estimates and the undefined draws of one estimator, size and estimand."""

    key: Key
    draws: int
    pairs: list[tuple[float, float]] = field(default_factory=list)
    undefined: int = 0
    last_reason: str = ""
    rank_deficient_folds: int = 0
    zero_se: int = 0

    def record(self, draw: StudyDraw, cluster: bool | None) -> None:
        """Estimate on one draw, or count the draw as undefined and keep its reason."""
        name, _, estimand = self.key
        # The one failure the study tolerates: a draw on which the estimator is undefined is
        # counted; a row with too many of them is reported as undefined by its summary.
        try:
            pair = draw_estimate(name, estimand, draw, cluster)
        except EstimatorError as error:
            self.undefined += 1
            self.last_reason = str(error)
        else:
            self.pairs.append(pair)
            self.rank_deficient_folds += draw.rank_deficient_folds(name)
            self.zero_se += int(pair[1] == 0.0)

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
            "n_rank_deficient_folds": self.rank_deficient_folds,
            "n_se_exactly_zero": self.zero_se,
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

    Before the draws of a size, the study measures with ``text_count_variation`` how much the
    number of drawn texts varies over all draws of its ``m`` of the ``G`` groups: ``c``, the
    coefficient of variation. The point of the mean of ``ppi_identity`` is a mean ``R`` over the
    drawn texts plus a constant, so when no draw is refused its bias is at most ``c`` times its
    standard deviation, both taken over all possible draws: ``R`` times the mean texts per drawn
    group has a fixed expectation, so ``E[R] - R_corpus = -Cov(R, mean texts per drawn group) /
    mean_n``, with ``mean_n`` the mean texts per group, which the Cauchy-Schwarz inequality
    bounds by ``c * sd(R)``. The bound covers neither the ``bias`` of a row, a mean over its
    defined draws of this run, nor a difference, ``calibrated``, ``ppi_crossfit`` or any
    coverage. For each size whose ``c`` exceeds ``TEXT_COUNT_VARIATION_THRESHOLD`` (0.1),
    whatever the number of draws, the study logs one warning, as ``draw_human_sample`` does:
    read the bias and the coverage of that size's rows and the reasons of their undefined
    draws. The mean of the identity pseudo-values, ``pseudo_values(data).mean_rectified``, is
    exactly unbiased over draws. After the draws of a size, one warning gives the count of each
    case of ``_open_cases`` that its calls met, in place of their own warnings; a study that
    stops with an error during a size logs it for the draws it finished.

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
            ``ppi_identity``, and the same choice for the standard error of ``ppi_crossfit``:
            ``None`` clusters the standard error by group when a group of the draw holds more
            than one labelled text, ``True`` and ``False`` (numpy booleans too) force the
            choice. A draw that clustering leaves undefined is counted in ``n_undefined``:
            fewer than two groups among the texts or the labelled texts, and for the difference
            a side whose labelled texts lie in one group (``ppi_identity``) or whose texts lie
            in one group of the corpus (``ppi_crossfit``). Under every setting both count a
            draw with fewer than two labelled texts, or fewer than two on a side of the
            difference, as undefined too.

    Returns:
        A frame with the columns ``estimator``, ``size``, ``estimand`` (``mean`` or
        ``long_short_difference``), ``truth`` (the full-gold value), ``estimate_mean``,
        ``bias``, ``coverage``, ``se_mean`` (over the defined draws; NaN for a row undefined on
        more than 10 percent of its draws), ``n_draws``, ``n_undefined``, ``status`` (``ok``
        or ``undefined_share_exceeded``), ``n_rank_deficient_folds`` (for ``ppi_crossfit``
        the inner folds, over the defined draws, of numerical rank below 3; 0 for the
        other estimators) and ``n_se_exactly_zero`` (the defined draws whose standard error
        is exactly 0, so that their interval has zero width), one row per estimator, size and
        estimand in the orders given.

    Raises:
        SettingsError: If a size, ``draws``, ``seed``, ``estimators`` or ``cluster`` is outside
            its range.
        ConfigurationError: If ``truth_gold`` is not a complete numeric column of the data.
        EstimatorError: If either side of the long-short difference holds fewer than two texts
            (ties at the median make the sides unequal, and a constant covariate leaves the
            short side empty), or every row is undefined on more than 10 percent of its draws
            (the message gives the last reason).
    """
    size_values = checked_sizes(sizes, data_full.n)
    draws = check_count("draws", draws, 1)
    seed = check_count("seed", seed, 0)
    names = checked_estimators(estimators, ESTIMATORS)
    requested = checked_cluster(cluster)
    truth_values = complete_column(data_full, truth_gold, "truth_gold")
    split = long_short_split(data_full)
    if min(int(split.sum()), int((~split).sum())) < MIN_SIDE:
        raise EstimatorError(
            "the long-short difference needs at least two texts on each side of the covariate's "
            f"median; the sides hold {int(split.sum())} and {int((~split).sum())}"
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
        m = groups_for_quota(size, data_full.groups, data_full.ids)
        warn_if_text_counts_vary(data_full.groups, m, log=logger, caller="study", size=size)
        counts: Counter[str] = Counter()
        try:
            with gathered() as counts:
                for index in range(draws):
                    draw = study_draw(data_full, truth, size, seed, split, index)
                    for tally in active:
                        tally.record(draw, requested)
        except BaseException as error:
            report_stopped(error, partial(_warn_open_cases, size, counts, finished=False))
            raise
        _warn_open_cases(size, counts, finished=True)
        logger.info(
            "repeated study: %d draws of size %d, undefined draws %s",
            draws,
            size,
            {tally.key[0] + "/" + tally.key[2]: tally.undefined for tally in active},
        )
    _report_undefined(tallies)
    rows = [tally.summary(truths[tally.key[2]]) for tally in tallies]
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def _warn_open_cases(size: int, counts: Counter[str], *, finished: bool) -> None:
    """One warning for a size whose draws met cases, in place of the warnings of its calls.

    A study that stopped with an error during the size returns no table, which the warning
    then says in place of where the table holds the cases.
    """
    if not counts:
        return
    notes = []
    if not finished:
        notes.append("the study stopped with an error during this size and returns no table")
    elif counts.get("rank_deficient_folds"):
        notes.append("the column n_rank_deficient_folds holds the folds of each row")
    if finished and counts.get("se_zero"):
        notes.append("the column n_se_exactly_zero holds the draws of each row")
    logger.warning(
        "repeated study, size %d: of the warnings that its inner calls log one by one when called "
        "on their own, %s; %s",
        size,
        summary(counts),
        "; ".join(notes),
    )


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
