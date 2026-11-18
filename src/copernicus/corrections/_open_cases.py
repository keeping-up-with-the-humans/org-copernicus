"""The warnings of the cases in which a correction keeps a value that the data leave open.

Seven cases are recorded where they occur (see each record's docstring), and the function that
meets one logs one warning per call through its own module's logger, with the helpers of this
module. ``run_battery`` and ``repeated_study`` call many such functions, so inside them these
warnings are counted instead (``gathered``) and logged as one summary per entry and size of the
battery, or per size of the study, with the count of each case (``summary``). Only the warnings
logged through ``warn`` are counted: no logger's level changes, and no other warning is held
back. The counting belongs to the block that started it: once the block has ended, a warning
is logged again, also one that reaches ``warn`` through a context copied inside the block.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    "CASES",
    "gathered",
    "summary",
    "warn",
    "warn_not_converged",
    "warn_open_orientation",
    "warn_rank_deficient_folds",
    "warn_rank_deficient_resamples",
    "warn_tied_rounds",
    "warn_tied_singular_values",
    "warn_uncorrelated_orientation",
    "warn_without_residual_df",
    "warn_zero_se",
]

CASES: Mapping[str, str] = {
    "rank_deficient_folds": "a cross-fitted prediction has folds whose design has a numerical "
    "rank below 3 (n_rank_deficient_folds)",
    "rank_deficient_resamples": "a fit has bootstrap resamples whose design has a numerical rank "
    "below 3 (n_rank_deficient_resamples)",
    "no_residual_df": "a fit has bootstrap fits with no more rows than coefficients "
    "(n_resamples_without_residual_df)",
    "orientation_open": "the sign of a result is treated as not identified, because the column "
    "that would set it or its reference is constant by is_constant, or the two correlate at "
    "most 1e-12 in absolute value (orientation_identified)",
    "direction_tied": "the direction of a result counts as not unique (direction_unique)",
    "not_converged": "the principal-axis iteration reached its cap of rounds without meeting "
    "its tolerance (factor_converged)",
    "se_zero": "a standard error is exactly 0 (se_exactly_zero)",
}
"""What each case's warning says, in the words of a summary, by the case's name."""


@dataclass
class _Collector:
    """The counts of one ``gathered`` block, and whether the block is still running."""

    counts: Counter[str] = field(default_factory=Counter)
    running: bool = True


_GATHERED: ContextVar[_Collector | None] = ContextVar("open_cases_gathered", default=None)


@contextmanager
def gathered() -> Iterator[Counter[str]]:
    """Count the warnings of the cases that ``warn`` gets within the block instead of logging them.

    The block must log what it counted, also when it ends with an error. After the block a
    warning is logged again, whichever context it comes through.

    Yields:
        The count of each case, filled while the block runs.
    """
    collector = _Collector()
    token = _GATHERED.set(collector)
    try:
        yield collector.counts
    finally:
        collector.running = False
        _GATHERED.reset(token)


def warn(log: logging.Logger, case: str, message: str, *args: object) -> None:
    """Log the warning of a case through ``log``, or count it when it happens within ``gathered``.

    Args:
        log: The module logger of the function that met the case.
        case: The case, a key of ``CASES``.
        message: The warning, with ``%`` placeholders.
        *args: The values of the placeholders.
    """
    collector = _GATHERED.get()
    if collector is None or not collector.running:
        log.warning(message, *args)
    else:
        collector.counts[case] += 1


def summary(counts: Mapping[str, int]) -> str:
    """The counts of the cases in the words of a summary, in the order of ``CASES``.

    Args:
        counts: The count of each case that occurred.

    Returns:
        Such as "6 say that ..., and 1 says that ...".
    """
    parts = [
        f"{counts[case]} {'says' if counts[case] == 1 else 'say'} that {text}"
        for case, text in CASES.items()
        if counts.get(case)
    ]
    return ", and ".join(parts)


def warn_rank_deficient_folds(
    log: logging.Logger, deficient: Sequence[tuple[int, int]], k: int, columns: int
) -> None:
    """The warning of a cross-fitted prediction whose folds have a rank-deficient design.

    Args:
        log: The caller's module logger.
        deficient: The fold and its rank, for each fold whose design has a numerical rank
            below ``columns``.
        k: The number of folds.
        columns: The columns of the design.
    """
    warn(
        log,
        "rank_deficient_folds",
        "cross-fitted prediction: the design of intercept, score and covariate has a numerical "
        "rank below %d, as numpy.linalg.lstsq with rcond=None returns it, on the labelled "
        "training rows of %d of the %d folds (%s), so lstsq there leaves out the directions "
        "below its cutoff and returns the coefficients of smallest norm of what remains. When the "
        "design is rank-deficient in exact arithmetic, many lines fit those rows equally well, "
        "and the prediction of a row whose intercept, score and covariate are not a linear "
        "combination of those of the training rows depends on the one that lstsq returns; "
        "n_rank_deficient_folds counts these folds. Check whether the covariate is constant on a "
        "fold's training rows, or a linear function of the score there",
        columns,
        len(deficient),
        k,
        ", ".join(f"fold {fold}: rank {rank}" for fold, rank in deficient),
    )


def warn_rank_deficient_resamples(
    log: logging.Logger, label: str, counts: tuple[int, int, int], column: str, columns: int
) -> None:
    """The warning of an imputation fit whose bootstrap resamples have a rank-deficient design.

    Args:
        log: The caller's module logger.
        label: The family and variant, ``family/variant``.
        counts: The resamples whose design has a numerical rank below ``columns``, the resamples
            drawn and the training rows.
        column: The score column of the design.
        columns: The columns of the design.
    """
    warn(
        log,
        "rank_deficient_resamples",
        "%s: %d of the %d bootstrap resamples of the %d training rows have a design of "
        "intercept, score %r and covariate of numerical rank below %d, as numpy.linalg.lstsq "
        "with rcond=None returns it, so lstsq on such a resample leaves out the directions below "
        "its cutoff and returns the coefficients of smallest norm of what remains. When the "
        "design is rank-deficient in exact arithmetic, as for a resample of at most two distinct "
        "texts, many lines fit it equally well, and the draws of a text whose intercept, score "
        "and covariate are not a linear combination of those of the resample's rows depend on "
        "the one that lstsq returns; the provenance counts these resamples in "
        "n_rank_deficient_resamples",
        label,
        *counts,
        column,
        columns,
    )


def warn_without_residual_df(
    log: logging.Logger, label: str, counts: tuple[int, int, int], columns: int
) -> None:
    """The warning of an imputation fit whose bootstrap fits have no more rows than coefficients.

    Args:
        log: The caller's module logger.
        label: The family and variant, ``family/variant``.
        counts: The fits with no more rows than coefficients, the fits and the rows of each.
        columns: The coefficients of each fit.
    """
    warn(
        log,
        "no_residual_df",
        "%s: %d of the %d bootstrap fits have no more rows than coefficients (%d rows for %d "
        "coefficients), so the residual variance of each, which within_variance averages and "
        "which is the variance of the noise that model_draws adds to the fitted mean, is its "
        "residual sum of squares divided by 1; the provenance counts these fits in "
        "n_resamples_without_residual_df. Fit on more than %d training texts",
        label,
        *counts,
        columns,
        columns,
    )


def warn_open_orientation(
    log: logging.Logger, method: str, condition: str, signed: str, source: str, consequence: str
) -> None:
    """The warning that a constant input leaves the sign of a result to the decomposition.

    Args:
        log: The caller's module logger.
        method: The method, which opens the warning.
        condition: What is constant, as a clause ("the mean of the members is constant").
        signed: What the sign belongs to, such as "the factor".
        source: What gave the sign that was kept, such as "numpy's eigendecomposition".
        consequence: What the open sign means for the result, as a clause.
    """
    warn(
        log,
        "orientation_open",
        "%s: %s (is_constant), so the sign of %s is not set by a correlation and stays the one "
        "that %s gave it; %s (orientation_identified is False)",
        method,
        condition,
        signed,
        source,
        consequence,
    )


def warn_uncorrelated_orientation(
    log: logging.Logger,
    method: str,
    signed: str,
    reference: str,
    values: tuple[float, float],
    consequence: str,
) -> None:
    """The warning that a correlation at or below the tolerance set the sign of a result.

    Args:
        log: The caller's module logger.
        method: The method, which opens the warning.
        signed: What the sign belongs to, such as "the factor".
        reference: What its correlation is taken with, such as "the mean of the members".
        values: The computed correlation and the tolerance at or below which the library
            treats it as too small to identify the sign.
        consequence: What the open sign means for the result, as a clause.
    """
    warn(
        log,
        "orientation_open",
        "%s: the correlation of %s with %s is %.3g, at most %g in absolute value, which the "
        "library treats as too small to identify the sign of %s; the sign follows the computed "
        "correlation, which at this size can be rounding error; %s (orientation_identified is "
        "False)",
        method,
        signed,
        reference,
        *values,
        signed,
        consequence,
    )


def warn_tied_rounds(
    log: logging.Logger, method: str, tied: int, rounds: int, tolerance: float
) -> None:
    """The warning that some rounds of the principal-axis iteration had a tie.

    Args:
        log: The caller's module logger.
        method: The method, which opens the warning.
        tied: The rounds with two tied largest eigenvalues, at least one.
        rounds: The rounds run.
        tolerance: The relative tolerance of a tie.
    """
    warn(
        log,
        "direction_tied",
        "%s: in %d of the %d rounds of the principal-axis iteration the two largest eigenvalues "
        "of the reduced correlation matrix are equal up to a relative %g (the largest minus the "
        "second at most %g times the largest), so the direction of the eigenvector taken in such "
        "a round counts as not unique: for two equal eigenvalues every unit vector in the plane "
        "of their directions would do, and the loadings can depend on the one that numpy "
        "returned (direction_unique is False)",
        method,
        tied,
        rounds,
        tolerance,
        tolerance,
    )


def warn_tied_singular_values(log: logging.Logger, method: str, tolerance: float) -> None:
    """The warning that the first principal component's direction counts as not unique.

    Args:
        log: The caller's module logger.
        method: The method, which opens the warning.
        tolerance: The relative tolerance of a tie.
    """
    warn(
        log,
        "direction_tied",
        "%s: the two largest singular values of the centred standardised members are equal up "
        "to a relative %g (the largest minus the second at most %g times the largest), so the "
        "direction of the first principal component counts as not unique: for two equal "
        "singular values every unit vector in the plane of their directions would do, and the "
        "combined column can depend on the one that numpy returned (direction_unique is False)",
        method,
        tolerance,
        tolerance,
    )


def warn_not_converged(log: logging.Logger, method: str, rounds: int, tolerance: float) -> None:
    """The warning that the principal-axis iteration reached its cap without converging.

    Args:
        log: The caller's module logger.
        method: The method, which opens the warning.
        rounds: The rounds run, the cap.
        tolerance: The tolerance on the change of the communalities.
    """
    warn(
        log,
        "not_converged",
        "%s: the principal-axis iteration reached its cap of %d rounds without the "
        "communalities changing by less than %g in a round, so the loadings are those of its "
        "last round (factor_converged is False)",
        method,
        rounds,
        tolerance,
    )


ZERO_SE_OF_AN_ESTIMATE = (
    "so the 95 percent interval has zero width (ci_low = ci_high = point), as when the values "
    "that it is computed from do not vary; the details record se_exactly_zero"
)
"""What a standard error of exactly 0 means for an ``Estimate``, and where it is recorded."""


def warn_zero_se(log: logging.Logger, method: str, note: str = ZERO_SE_OF_AN_ESTIMATE) -> None:
    """The warning of a standard error that is exactly 0.

    Args:
        log: The caller's module logger.
        method: The estimator's name.
        note: What the zero means and where it is recorded; by default the words for an
            ``Estimate``.
    """
    warn(log, "se_zero", "%s: the standard error is exactly 0, %s", method, note)
