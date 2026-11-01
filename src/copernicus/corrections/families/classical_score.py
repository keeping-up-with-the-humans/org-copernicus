"""Classical score corrections: best linear predictor shrinkage and Tweedie smoothing.

Both variants work on the single score column with the highest signed Spearman correlation
with the gold values on the training rows, and both use population moments (division by
``n``) for every covariance and variance, so that the ratios are consistent.

* ``fuller_shrinkage``: ``mean(gold) + kappa * (score - mean(score))`` with
  ``kappa = cov(gold, score) / var(score)``, the reliability-ratio line. It is the least
  squares line, so it leaves the ranks of the scores unchanged for a positive ``kappa``.
  It addresses the calibration shortfall (the spread is matched to the signal).
* ``tweedie``: a classical proxy ``X = mean(gold) + (score - mean(score)) / beta`` is built
  from the slope ``beta = cov(score, gold) / var(gold)``; the error variance of the proxy is
  ``sigma2 = max(var(X) - var(gold), 0)``; the posterior mean under normal error is
  ``X + sigma2 * dlog f(X) / dX`` with ``f`` a Gaussian kernel density of the training
  proxy (Silverman bandwidth ``1.06 * sd(X) * n^(-1/5)``). The values are then projected
  with an increasing isotonic regression fitted on the score order, so that the correction
  is monotone in the score and can be applied to any score. A score with negative
  covariance with the gold values is refused: reverse it first. Under a normal marginal it
  reduces to the least-squares line. It addresses the calibration shortfall and noise.

scikit-learn is imported inside the fitting function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._fitted import ScoreFitted, TrainingRows, training_rows
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.errors import EstimatorError, SettingsError

logger = logging.getLogger(__name__)

__all__ = ["ClassicalScore"]

FAMILY = "classical"
VARIANTS = ("fuller_shrinkage", "tweedie")
ERRORS_ADDRESSED = ("calibration_shortfall", "noise")
VARIANT_ERRORS = {
    "fuller_shrinkage": ("calibration_shortfall",),
    "tweedie": ("calibration_shortfall", "noise"),
}
SILVERMAN_FACTOR = 1.06
KDE_BLOCK = 512
ZERO_CORRELATION = 1e-12


def _population_covariance(first: np.ndarray, second: np.ndarray) -> float:
    """Covariance of two columns with the divisor ``n``."""
    return float(np.mean((first - first.mean()) * (second - second.mean())))


def _fit_fuller_shrinkage(rows: TrainingRows) -> ScoreFitted:
    # No constancy guard: the training rows' column was chosen by best_score_column, which
    # skips a constant column and a constant gold, so the variance below is positive.
    mean_gold, mean_score = float(rows.gold.mean()), float(rows.score.mean())
    kappa = _population_covariance(rows.gold, rows.score) / float(np.var(rows.score))
    return ScoreFitted(
        column=rows.column,
        predictor=lambda score, covariate: mean_gold + kappa * (score - mean_score),
        provenance={
            "family": FAMILY,
            "variant": "fuller_shrinkage",
            "score_column": rows.column,
            "kappa": kappa,
            "mean_gold": mean_gold,
            "mean_score": mean_score,
        },
    )


def _kde_log_density_slope(points: np.ndarray, bandwidth: float) -> np.ndarray:
    """``d/dx log f(x)`` at the points for a Gaussian kernel density fitted on the same points.

    The point itself contributes a kernel weight of 1, so the density is never zero.
    """
    slope = np.empty(points.size)
    for start in range(0, points.size, KDE_BLOCK):
        block = points[start : start + KDE_BLOCK]
        u = (block[:, None] - points[None, :]) / bandwidth
        weight = np.exp(-0.5 * u * u)
        slope[start : start + KDE_BLOCK] = (
            (-u * weight).sum(axis=1) / bandwidth / weight.sum(axis=1)
        )
    return slope


def _fit_tweedie(rows: TrainingRows) -> ScoreFitted:
    from sklearn.isotonic import IsotonicRegression

    score, gold = rows.score, rows.gold
    # Neither the score nor the gold values can be constant here (see _fit_fuller_shrinkage).
    covariance = _population_covariance(score, gold)
    variance_gold = float(np.var(gold))
    if abs(covariance) <= ZERO_CORRELATION * float(np.std(score) * np.std(gold)):
        raise EstimatorError(
            f"score column {rows.column!r} has zero covariance with the gold values on the "
            "training rows: the classical proxy is undefined"
        )
    if covariance < 0.0:
        raise EstimatorError(
            f"score column {rows.column!r} has negative covariance with the gold values on the "
            "training rows: reverse the score (for example multiply it by -1) before the "
            "Tweedie correction, because a posterior mean under a reversed proxy is not meaningful"
        )
    beta = covariance / variance_gold
    proxy = gold.mean() + (score - score.mean()) / beta
    sigma2 = max(float(np.var(proxy)) - variance_gold, 0.0)
    bandwidth = SILVERMAN_FACTOR * float(np.std(proxy)) * proxy.size ** (-1 / 5)
    posterior_mean = proxy + sigma2 * _kde_log_density_slope(proxy, bandwidth)
    projection = IsotonicRegression(increasing=True, out_of_bounds="clip")
    projection.fit(score, posterior_mean)
    return ScoreFitted(
        column=rows.column,
        predictor=lambda new_score, covariate: projection.predict(new_score),
        provenance={
            "family": FAMILY,
            "variant": "tweedie",
            "score_column": rows.column,
            "proxy_slope": float(beta),
            "sigma2_delta": sigma2,
            "kde_bandwidth": bandwidth,
        },
    )


_FITTERS: dict[str, Callable[[TrainingRows], ScoreFitted]] = {
    "fuller_shrinkage": _fit_fuller_shrinkage,
    "tweedie": _fit_tweedie,
}


@dataclass(frozen=True)
class ClassicalScore:
    """A classical measurement-error correction of one score column.

    Attributes:
        variant: ``fuller_shrinkage`` or ``tweedie``.
    """

    variant: str
    family: ClassVar[str] = FAMILY
    target: ClassVar[Literal["A", "B"]] = "A"

    def __post_init__(self) -> None:
        """Refuse an unknown variant.

        Raises:
            SettingsError: If the variant is not ``fuller_shrinkage`` or ``tweedie``.
        """
        if self.variant not in VARIANTS:
            raise SettingsError(
                f"unknown variant {self.variant!r} of family {FAMILY!r}; variants: "
                f"{', '.join(VARIANTS)}"
            )

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """The classical corrections apply to any data.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``.
        """
        return True, ""

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> ScoreFitted:
        """Fit the variant on the labelled training rows.

        Args:
            data: The correction data.
            train_ids: Ids of the labelled rows to learn from.
            rng: Unused; the fit is deterministic.

        Returns:
            The fitted correction; its provenance holds ``kappa`` (``fuller_shrinkage``) or
            ``sigma2_delta`` and ``kde_bandwidth`` (``tweedie``) and the score column.

        Raises:
            ConfigurationError: If an id is unknown, repeated or has no gold value.
            EstimatorError: If no score column is usable (a constant score or constant gold
                included), or (for ``tweedie``) the score has zero or negative covariance
                with the gold values.
        """
        del rng
        return _FITTERS[self.variant](training_rows(data, train_ids))


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="A",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors="none",
        factory=ClassicalScore,
        variant_errors=VARIANT_ERRORS,
    )
)
