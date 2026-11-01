"""Marginal matching: the score distribution is mapped onto the gold distribution.

Both variants match marginals, not conditional means. The corrected scores have the mean
and spread (``mean_sd``) or the whole distribution (``quantile``) of the gold values on the
training rows, whatever the relation between the score and the gold value of a single text.
They make scores comparable across settings; they do not remove a bias that depends on a
covariate, and they are tagged ``comparability_only`` in the provenance. The family
addresses level displacement and calibration shortfall in that sense only: the spread is
matched to the human-coded values, not to the signal.

The ``quantile`` variant maps a score through its mid-rank empirical quantile
``(average_rank - 0.5) / n`` among the training scores onto the training gold values. This
deviates from the simplest map, interpolating the sorted training scores against the sorted
gold values, because that map sends every text of a block of tied (discrete) scores to the
gold value at the top edge of the block and so biases the mapped level upwards. On scores
without ties the two maps coincide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from scipy.stats import rankdata

from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._fitted import ScoreFitted, TrainingRows, training_rows
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.corrections.families.calibration import LinePredictor
from copernicus.errors import EstimatorError, SettingsError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["MarginalMatching", "fit_mean_sd_line"]

FAMILY = "marginal_matching"
VARIANTS = ("mean_sd", "quantile")
ERRORS_ADDRESSED = ("level_displacement", "calibration_shortfall")


def fit_mean_sd_line(score: np.ndarray, gold: np.ndarray, column: str) -> LinePredictor:
    """The line that gives the score the mean and population SD of the gold values.

    The slope is ``sd(gold) / sd(score)`` and the intercept ``mean(gold) - slope *
    mean(score)``. A constant score has no spread to rescale: it is refused, because a zero
    slope would return the gold mean for every text and hide that the score carries no
    information.

    Args:
        score: Training scores.
        gold: Training gold values.
        column: Name of the score column, recorded in the line.

    Returns:
        The line.

    Raises:
        EstimatorError: If the score is constant on the training rows.
    """
    if is_constant(score):
        raise EstimatorError(
            f"score column {column!r} is constant on the training rows: "
            "mean and SD matching has no spread to rescale"
        )
    slope = float(np.std(gold) / np.std(score))
    return LinePredictor(
        intercept=float(np.mean(gold) - slope * np.mean(score)), slope=slope, column=column
    )


def _fit_mean_sd(rows: TrainingRows) -> ScoreFitted:
    line = fit_mean_sd_line(rows.score, rows.gold, rows.column)
    return ScoreFitted(
        column=rows.column,
        predictor=lambda score, covariate: line.predict(score),
        provenance={
            "family": FAMILY,
            "variant": "mean_sd",
            "score_column": rows.column,
            "intercept": line.intercept,
            "slope": line.slope,
            "comparability_only": True,
        },
    )


def _fit_quantile(rows: TrainingRows) -> ScoreFitted:
    n_rows = rows.score.size
    values, first = np.unique(rows.score, return_index=True)
    mid_rank_quantile = (rankdata(rows.score, method="average")[first] - 0.5) / n_rows
    gold = np.sort(rows.gold)

    def predictor(score: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        quantile = np.interp(score, values, mid_rank_quantile)
        return np.quantile(gold, quantile, method="hazen")

    return ScoreFitted(
        column=rows.column,
        predictor=predictor,
        provenance={
            "family": FAMILY,
            "variant": "quantile",
            "score_column": rows.column,
            "n_points": int(n_rows),
            "n_distinct_scores": int(values.size),
            "comparability_only": True,
        },
    )


@dataclass(frozen=True)
class MarginalMatching:
    """Marginal matching of one score column to the gold distribution.

    The corrected score has the marginal distribution (mean and SD for ``mean_sd``, all
    quantiles for ``quantile``) of the training gold values. This is a comparability tool:
    it matches marginals, not conditional means, so it must not be read as a correction of
    the error of a single text or of a bias that depends on a covariate.

    Attributes:
        variant: ``mean_sd`` or ``quantile``.
    """

    variant: str
    family: ClassVar[str] = FAMILY
    target: ClassVar[Literal["A", "B"]] = "A"

    def __post_init__(self) -> None:
        """Refuse an unknown variant.

        Raises:
            SettingsError: If the variant is not ``mean_sd`` or ``quantile``.
        """
        if self.variant not in VARIANTS:
            raise SettingsError(
                f"unknown variant {self.variant!r} of family {FAMILY!r}; variants: "
                f"{', '.join(VARIANTS)}"
            )

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """Marginal matching applies to any data.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``.
        """
        return True, ""

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> ScoreFitted:
        """Fit the matching on the labelled training rows.

        Args:
            data: The correction data.
            train_ids: Ids of the labelled rows to learn from.
            rng: Unused; the fit is deterministic.

        Returns:
            The fitted matching; its provenance carries ``comparability_only = True``.

        Raises:
            ConfigurationError: If an id is unknown or has no gold value.
            EstimatorError: If no score column is usable on the training rows.
        """
        del rng
        rows = training_rows(data, train_ids)
        return _fit_mean_sd(rows) if self.variant == "mean_sd" else _fit_quantile(rows)


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="A",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors="none",
        factory=MarginalMatching,
    )
)
