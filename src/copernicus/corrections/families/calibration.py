"""Regression calibration: the score is mapped to the gold scale with a model fitted on gold.

Four variants, all on the single score column that has the highest signed Spearman
correlation with the gold values on the training rows:

* ``linear``: the least-squares line ``gold ~ 1 + score``. A positive slope leaves the
  ranks of the scores unchanged.
* ``covariate_augmented``: ``gold ~ 1 + score + covariate``. The covariate term can move
  the ranks.
* ``platt``: a logistic regression of the larger gold class on the score; the predicted
  probability ``p`` is placed on the gold scale as ``lo + p * (hi - lo)`` with the two gold
  values ``lo < hi`` (the probability itself for 0/1 gold). It applies only to binary gold.
* ``isotonic``: scikit-learn's isotonic regression, a monotone piecewise-linear map: the fit
  is non-decreasing over the training scores, a score between two of them gets the linearly
  interpolated value, and a score outside the training range gets the value at the nearer end.

The family addresses level displacement and calibration shortfall (the spread of the
corrected score is matched to the signal); ``covariate_augmented`` also addresses the error
that is linked to the covariate. scikit-learn is imported inside the fitting functions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._fitted import (
    ScoreFitted,
    TrainingRows,
    labelled_positions,
    training_rows,
)
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.errors import EstimatorError, NotApplicableError, SettingsError
from copernicus.validity.metrics import is_binary, is_constant

logger = logging.getLogger(__name__)

__all__ = ["LinePredictor", "RegressionCalibration", "fit_ols_line"]

FAMILY = "regression_calibration"
VARIANTS = ("linear", "covariate_augmented", "platt", "isotonic")
PLATT_MAX_ITER = 1000
DESIGN_COLUMNS = 3
ERRORS_ADDRESSED = ("level_displacement", "calibration_shortfall", "covariate_linked_error")
VARIANT_ERRORS = {
    "linear": ("level_displacement", "calibration_shortfall"),
    "covariate_augmented": ERRORS_ADDRESSED,
    "platt": ("level_displacement", "calibration_shortfall"),
    "isotonic": ("level_displacement", "calibration_shortfall"),
}


@dataclass(frozen=True)
class LinePredictor:
    """A fitted line ``intercept + slope * score`` on one score column.

    Attributes:
        intercept: The value at a score of zero.
        slope: The change per unit of score.
        column: The score column the line reads.
    """

    intercept: float
    slope: float
    column: str

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Apply the line.

        Args:
            scores: Values of the score column.

        Returns:
            ``intercept + slope * scores`` as floats.
        """
        return self.intercept + self.slope * np.asarray(scores, dtype=float)


def fit_ols_line(score: np.ndarray, gold: np.ndarray, column: str) -> LinePredictor:
    """Least-squares line of the gold values on the score.

    Args:
        score: Training scores.
        gold: Training gold values.
        column: Name of the score column, recorded in the line.

    Returns:
        The fitted line.

    Raises:
        EstimatorError: If the score is constant, so that no slope is defined.
    """
    if is_constant(score):
        raise EstimatorError(f"score column {column!r} is constant on the training rows: no line")
    design = np.column_stack([np.ones_like(score), score])
    beta, *_ = np.linalg.lstsq(design, gold, rcond=None)
    return LinePredictor(intercept=float(beta[0]), slope=float(beta[1]), column=column)


def _provenance(variant: str, rows: TrainingRows, **fitted: object) -> dict[str, object]:
    """The provenance every fit of this family records."""
    return {"family": FAMILY, "variant": variant, "score_column": rows.column, **fitted}


def _fit_linear(rows: TrainingRows) -> ScoreFitted:
    line = fit_ols_line(rows.score, rows.gold, rows.column)
    return ScoreFitted(
        column=rows.column,
        predictor=lambda score, covariate: line.predict(score),
        provenance=_provenance("linear", rows, intercept=line.intercept, slope=line.slope),
    )


def _fit_covariate_augmented(rows: TrainingRows) -> ScoreFitted:
    design = np.column_stack([np.ones_like(rows.score), rows.score, rows.covariate])
    rank = int(np.linalg.matrix_rank(design))
    if rank < DESIGN_COLUMNS:
        raise EstimatorError(
            f"the design of intercept, score {rows.column!r} and covariate has rank {rank} of "
            f"{DESIGN_COLUMNS} on the training rows (a constant or collinear column): "
            "the coefficients are not defined"
        )
    beta, *_ = np.linalg.lstsq(design, rows.gold, rcond=None)
    intercept, slope, coefficient = (float(value) for value in beta)

    def predictor(score: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        return intercept + slope * score + coefficient * covariate

    return ScoreFitted(
        column=rows.column,
        predictor=predictor,
        provenance=_provenance(
            "covariate_augmented",
            rows,
            intercept=intercept,
            slope=slope,
            covariate_coefficient=coefficient,
        ),
    )


def _binary_gold_problem(gold: np.ndarray, rows: str) -> str:
    """Why ``platt`` does not apply to these gold values, or ``""`` when they take two values."""
    if is_binary(gold):
        return ""
    n_classes = int(np.unique(gold[np.isfinite(gold)]).size)
    return (
        f"platt calibration needs binary gold; the {rows} gold values take {n_classes} distinct "
        "values"
    )


def _fit_platt(rows: TrainingRows) -> ScoreFitted:
    """Fit the logistic regression; ``RegressionCalibration.fit`` has checked for two classes."""
    from sklearn.linear_model import LogisticRegression

    classes = np.unique(rows.gold)
    model = LogisticRegression(max_iter=PLATT_MAX_ITER)
    model.fit(rows.score.reshape(-1, 1), (rows.gold == classes[1]).astype(int))

    low, high = float(classes[0]), float(classes[1])

    def predictor(score: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        return low + model.predict_proba(score.reshape(-1, 1))[:, 1] * (high - low)

    return ScoreFitted(
        column=rows.column,
        predictor=predictor,
        provenance=_provenance(
            "platt",
            rows,
            intercept=float(model.intercept_[0]),
            slope=float(model.coef_[0, 0]),
            negative_class=low,
            positive_class=high,
        ),
    )


def _fit_isotonic(rows: TrainingRows) -> ScoreFitted:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(rows.score, rows.gold)

    def predictor(score: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        return model.predict(score)

    return ScoreFitted(
        column=rows.column,
        predictor=predictor,
        provenance=_provenance("isotonic", rows, n_thresholds=int(model.X_thresholds_.size)),
    )


_FITTERS: dict[str, Callable[[TrainingRows], ScoreFitted]] = {
    "linear": _fit_linear,
    "covariate_augmented": _fit_covariate_augmented,
    "platt": _fit_platt,
    "isotonic": _fit_isotonic,
}


@dataclass(frozen=True)
class RegressionCalibration:
    """Regression calibration of one score column against the gold values.

    Attributes:
        variant: ``linear``, ``covariate_augmented``, ``platt`` or ``isotonic``.
    """

    variant: str
    family: ClassVar[str] = FAMILY
    target: ClassVar[Literal["A", "B"]] = "A"

    def __post_init__(self) -> None:
        """Refuse an unknown variant.

        Raises:
            SettingsError: If the variant is not one of the four.
        """
        if self.variant not in VARIANTS:
            raise SettingsError(
                f"unknown variant {self.variant!r} of family {FAMILY!r}; variants: "
                f"{', '.join(VARIANTS)}"
            )

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """Whether the variant can run on the data; only ``platt`` has a condition.

        ``fit`` checks the same condition on its training rows.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``, or ``(False, reason)`` for ``platt`` when the labelled gold
            values do not take exactly two values.
        """
        if self.variant != "platt":
            return True, ""
        problem = _binary_gold_problem(data.gold[data.labelled_mask], "labelled")
        return (False, problem) if problem else (True, "")

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> ScoreFitted:
        """Fit the variant on the labelled training rows.

        Args:
            data: The correction data.
            train_ids: Ids of the labelled rows to learn from.
            rng: Unused; the fit is deterministic.

        Returns:
            The fitted calibration; its provenance names the family, the variant, the
            score column and the coefficients.

        Raises:
            ConfigurationError: If an id is unknown or has no gold value.
            NotApplicableError: If ``platt`` meets fewer or more than two gold classes on the
                training rows; the message gives the reason ``applies`` would give.
            EstimatorError: If no score column is usable or the design is rank deficient.
        """
        del rng
        if self.variant == "platt":
            _, gold = labelled_positions(data, train_ids)
            problem = _binary_gold_problem(gold, "training")
            if problem:
                raise NotApplicableError(
                    f"{FAMILY}/{self.variant} does not apply to the training rows: {problem}"
                )
        return _FITTERS[self.variant](training_rows(data, train_ids))


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="A",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors="none",
        factory=RegressionCalibration,
        variant_errors=VARIANT_ERRORS,
    )
)
