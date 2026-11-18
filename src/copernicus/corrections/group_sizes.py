"""How unequal group sizes make the number of drawn texts vary, and the bias that this allows.

A whole-group draw, as ``draw_human_sample``, ``repeated_study`` and the battery's diagnostic
make it, takes ``m`` of the ``G`` groups by simple random sampling without replacement and
codes every text of a drawn group, so with groups of unequal size the number of texts drawn
varies from draw to draw. A mean over the drawn texts is then a ratio of two means over the
drawn groups (of the groups' totals and of their numbers of texts) and has a bias.
``text_count_variation`` measures the variation of the number of drawn texts. It bounds the
bias of such a mean of a quantity fixed before the draw over all possible draws, and with it
the point of ``mean_ppi`` in its default form when no draw is refused; it bounds neither a
difference nor a power-tuned or cross-fitted estimate, nor a summary over the accepted draws
only, nor any coverage. ``warn_if_text_counts_vary`` logs the warning of the draw, of the
repeated study and of the battery's diagnostic when the variation exceeds
``TEXT_COUNT_VARIATION_THRESHOLD``: a signal that the number of drawn texts varies much.
"""

from __future__ import annotations

import logging
import math
import numbers
from fractions import Fraction
from typing import Literal

import numpy as np
import pandas as pd

from copernicus.errors import ConfigurationError, SettingsError

logger = logging.getLogger(__name__)

__all__ = ["TEXT_COUNT_VARIATION_THRESHOLD", "text_count_variation", "warn_if_text_counts_vary"]

TEXT_COUNT_VARIATION_THRESHOLD = 0.1
"""The variation above which the draw, the repeated study and the battery's diagnostic warn.

``text_count_variation`` bounds the bias of a mean over the drawn texts of a quantity fixed
before the draw by the variation times the mean's standard deviation, both taken over all
possible draws, so at or below 0.1 that bias is at most a tenth of the standard deviation. The
warning is a signal that the number of drawn texts varies much, not a bound for every
estimator: ``draw_human_sample`` logs one when the variation of its draw exceeds this value,
``repeated_study`` one for each size whose draws exceed it, and ``run_battery`` one for each
such size of its diagnostic. The decision compares the square of the variation, computed
exactly from the counts, with the square of the fraction 1/10 that this float names.
"""

# TEXT_COUNT_VARIATION_THRESHOLD as an exact fraction; a test checks that the float names it.
_THRESHOLD = Fraction(1, 10)

Caller = Literal["draw", "study", "diagnostic"]

_WARNING = (
    "%sthe number of texts in a draw of %d of the %d groups varies with the groups drawn: its "
    "coefficient of variation over all such draws, its standard deviation divided by its mean, "
    "%s, and the largest group holds %.3g percent of the %d texts. A mean over the drawn texts "
    "can then be biased. For the point of mean_ppi without power tuning and with a "
    "prediction fixed before the draw, such as a score, that bias is at most this coefficient "
    "times the point's standard deviation, both taken over all possible draws of %d of the %d "
    "groups, provided mean_ppi refuses none of them (it refuses a draw with fewer than two "
    "labelled texts and, when it clusters, one whose labelled texts lie in one group). The "
    "coefficient does not bound the bias of a difference (group_difference_ppi), whose two "
    "sides each vary on their own, nor that of a power-tuned estimate or of one whose "
    "prediction is fitted on the drawn texts, such as a cross-fitted one, and the intervals of "
    "all these estimates can cover less often than 95 percent. %s"
)
_OPENINGS: dict[Caller, str] = {
    "draw": "",
    "study": "repeated study",
    "diagnostic": "battery diagnostic",
}
_ADVICE: dict[Caller, str] = {
    "draw": (
        "Check the design with repeated_study, or estimate the mean with "
        "pseudo_values(data).mean_rectified, which, with its default identity rectifier, the "
        "recorded pi and every text of the drawn groups coded, is exactly unbiased over draws"
    ),
    "study": (
        "Read the bias and the coverage of the rows of this size and the reasons of their "
        "undefined draws: a row's bias is taken over its defined draws of this run only, which "
        "the bound does not cover, and the bound says nothing about coverage"
    ),
    "diagnostic": (
        "Of the two means that the diagnostic records for this size, mean_rectified, the mean "
        "of the identity pseudo-values over all texts, is exactly unbiased over draws, and "
        "mean_rectified_tuned, a mean over the drawn texts with a weight tuned on them, is a "
        "ratio whose bias the coefficient does not bound"
    ),
}


def _texts_per_group(group_id: pd.Series | np.ndarray) -> list[int]:
    """The number of texts of every group, the ids compared as strings, as the draw compares them.

    Raises:
        ConfigurationError: If the ids are not one-dimensional or hold no text.
    """
    groups = np.asarray(group_id).astype(str)
    if groups.ndim != 1:
        raise ConfigurationError(
            f"group_id must be one-dimensional, one group id per text; it has {groups.ndim} "
            "dimensions"
        )
    if groups.size == 0:
        raise ConfigurationError("group_id holds no text: give the group id of every text")
    return [int(count) for count in np.unique(groups, return_counts=True)[1]]


def _moments(sizes: list[int], m: int) -> tuple[int, int]:
    """The integers ``A`` and ``B`` with ``c^2 = A / B`` for the given numbers of texts per group.

    ``A = (G * sum(n^2) - N^2) * (G - m)`` and ``B = (G - 1) * N^2 * m``, with ``N`` the number
    of texts; ``(0, 1)`` for the full draw (``m = G``), which also covers a single group.

    Raises:
        SettingsError: If ``m`` is not an integer between 1 and the number of groups.
    """
    n_groups = len(sizes)
    if isinstance(m, bool) or not isinstance(m, numbers.Integral) or not 1 <= m <= n_groups:
        raise SettingsError(f"m must be an integer between 1 and the {n_groups} groups; got {m!r}")
    drawn = int(m)
    if drawn == n_groups:
        return 0, 1
    n_texts = sum(sizes)
    spread = n_groups * sum(size * size for size in sizes) - n_texts * n_texts
    return spread * (n_groups - drawn), (n_groups - 1) * n_texts * n_texts * drawn


def _variation(sizes: list[int], m: int) -> float:
    """``c`` of the given numbers of texts per group: ``sqrt(A / B)`` of ``_moments``.

    Python divides the two integers with one correct rounding, and the square root rounds once
    more.

    Raises:
        SettingsError: If ``m`` is not an integer between 1 and the number of groups.
    """
    numerator, denominator = _moments(sizes, m)
    return math.sqrt(numerator / denominator)


def _exceeds_threshold(sizes: list[int], m: int) -> bool:
    """Whether ``c`` exceeds the threshold, decided exactly: ``A / B`` against ``(1 / 10)^2``.

    Raises:
        SettingsError: If ``m`` is not an integer between 1 and the number of groups.
    """
    numerator, denominator = _moments(sizes, m)
    return numerator * _THRESHOLD.denominator**2 > denominator * _THRESHOLD.numerator**2


def text_count_variation(group_id: pd.Series | np.ndarray, m: int) -> float:
    """The variation of the number of texts over the whole-group draws of ``m`` groups.

    The value ``c`` is the coefficient of variation of the number of texts in a draw, its
    standard deviation divided by its mean, over all draws of ``m`` of the ``G`` groups by
    simple random sampling without replacement (the draw of ``draw_human_sample``)::

        c = (sd_n / mean_n) * sqrt((1 - m / G) / m)

    with ``n_g`` the number of texts of group ``g``, ``mean_n`` their mean over all ``G`` groups
    and ``sd_n`` their standard deviation over all ``G`` groups with divisor ``G - 1``. It is
    computed from the integer counts, exactly up to one division and one square root. It is 0
    when every group holds the same number of texts (one text per group included), for the
    full draw (``m = G``), and for a single group, whose one draw is the full draw.

    ``c`` bounds the bias of the mean over the drawn texts of a quantity fixed before the draw,
    ``R``, the drawn groups' total of the quantity divided by their number of texts:
    ``|E[R] - R_corpus| <= c * sd(R)``, with the expectation and the standard deviation taken
    over all possible draws of ``m`` of the ``G`` groups. The reason: ``R`` times the mean
    number of texts of a drawn group is the mean of the drawn groups' totals, whose expectation
    is the corpus total divided by ``G``, so ``E[R] - R_corpus = -Cov(R, mean texts per drawn
    group) / mean_n``; the Cauchy-Schwarz inequality bounds that covariance by ``sd(R)`` times
    the standard deviation of the mean texts per drawn group, which is ``c * mean_n``. At
    ``TEXT_COUNT_VARIATION_THRESHOLD``, 0.1, the bias is at most a tenth of ``sd(R)``. The
    bound is reached when ``R`` is a line in the number of drawn texts: for instance when every
    group but one holds the same number of texts and the quantity differs from zero only on the
    texts of that one group.

    Of the library's estimators the bound covers the point of ``mean_ppi`` in its default form,
    without power tuning and with a prediction fixed before the draw such as a score: such a
    mean, of the gap between the human value and the prediction, plus the mean of the
    prediction over all texts, which does not vary over draws. It covers it only when the
    estimator refuses no draw (it refuses a draw with fewer than two labelled texts and, when it
    clusters, one whose labelled texts lie in one group). It does not bound a summary over the
    accepted draws only, which a refused draw changes, such as the bias that ``repeated_study``
    reports, the mean over the defined draws of a finite run; nor a power-tuned or a
    cross-fitted estimate; nor the coverage of any interval. ``c`` of all texts does not bound
    the bias of ``group_difference_ppi`` either. For that estimator without power tuning and
    with a prediction fixed before the draw, provided it refuses no draw, the bias of the
    difference is at most ``c_A * sd(R_A) + c_B * sd(R_B)``. Here ``R_A`` and ``R_B`` are each
    side's mean, over its drawn texts, of the gap between the human value and the prediction,
    both standard deviations are taken over all possible draws, and ``c_A`` is the formula
    above with ``n_g`` the number of the group's texts on side A, counted over all ``G`` groups
    of the corpus, a group without a text on the side counting 0. This function applied to the
    ids of one side's texts alone leaves out the groups without a text on that side, so keep
    those groups when computing ``c_A``. ``pseudo_values(data).mean_rectified`` divides by the
    number of all texts instead and, with its default ``identity`` rectifier, the recorded
    ``pi = m / G`` and every text of a drawn group coded, is exactly unbiased over draws.

    Args:
        group_id: The group id of every text of the corpus that is drawn from, one entry per
            text; ids are compared as strings, as the draw compares them.
        m: The number of groups drawn, between 1 and ``G``.

    Returns:
        ``c``, at least 0.

    Raises:
        ConfigurationError: If ``group_id`` is not one-dimensional or holds no text.
        SettingsError: If ``m`` is not an integer between 1 and the number of groups.
    """
    return _variation(_texts_per_group(group_id), m)


def warn_if_text_counts_vary(
    group_id: pd.Series | np.ndarray,
    m: int,
    *,
    log: logging.Logger,
    caller: Caller,
    size: int | None,
) -> None:
    """Log one warning when ``text_count_variation`` exceeds ``TEXT_COUNT_VARIATION_THRESHOLD``.

    The decision is exact: ``c^2``, a fraction of two integers of the counts, is compared with
    the square of the fraction 1/10 that the threshold names, so counts whose ``c`` rounds to
    0.1 but lies above it warn. The warning names ``c`` (as "just above 0.1" when three
    significant digits would show the threshold), ``m``, ``G`` and the share of the texts that
    the largest group holds. It says that a mean over the drawn texts is then a ratio with a
    bias; that ``c`` bounds that bias for the point of ``mean_ppi`` in its default form, over all
    possible draws and provided no draw is refused; that ``c`` bounds neither a difference,
    whose two sides each vary on their own, nor a power-tuned or cross-fitted estimate, and
    that the intervals of all of them can cover less often than 95 percent; and the caller's
    advice. At or below the threshold nothing is logged.

    Args:
        group_id: The group id of every text of the corpus that is drawn from.
        m: The number of groups drawn.
        log: The module logger of the caller, which logs the warning.
        caller: ``"draw"`` for ``draw_human_sample``, whose advice is to check the design with
            ``repeated_study`` or to estimate the mean with ``mean_rectified``; ``"study"`` for a
            size of ``repeated_study``, whose advice is to read the bias, the coverage and the
            undefined draws of that size's rows; ``"diagnostic"`` for a size of the battery's
            diagnostic, whose advice says which of the two means it records the bound leaves
            out.
        size: The size of the study or of the diagnostic, which opens the message; ``None`` for
            a draw.

    Raises:
        ConfigurationError: If ``group_id`` is not one-dimensional or holds no text.
        SettingsError: If ``m`` is not an integer between 1 and the number of groups.
    """
    _warn_for_counts(_texts_per_group(group_id), m, log=log, caller=caller, size=size)


def _warn_for_counts(
    sizes: list[int], m: int, *, log: logging.Logger, caller: Caller, size: int | None
) -> None:
    """``warn_if_text_counts_vary`` on the numbers of texts per group.

    Raises:
        SettingsError: If ``m`` is not an integer between 1 and the number of groups.
    """
    if not _exceeds_threshold(sizes, m):
        return
    shown, threshold = f"{_variation(sizes, m):.3g}", f"{TEXT_COUNT_VARIATION_THRESHOLD:g}"
    level = f"is just above {threshold}" if shown == threshold else f"is {shown}, above {threshold}"
    n_texts = sum(sizes)
    log.warning(
        _WARNING,
        "" if size is None else f"{_OPENINGS[caller]}, size {size}: ",
        m,
        len(sizes),
        level,
        100.0 * max(sizes) / n_texts,
        n_texts,
        m,
        len(sizes),
        _ADVICE[caller],
    )
