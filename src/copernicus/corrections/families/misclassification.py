"""Misclassification matrix: discretise the score, estimate P(predicted | true), invert by Bayes.

The classes are the distinct gold values of the labelled rows (at most ``max_classes``).
The best score column of the training rows is cut into classes at the quantiles of the
training gold marginal, so that the discretised score has the same class shares as the gold
values. The confusion matrix ``P(predicted class | true class)`` is estimated on the
training rows with a Dirichlet(1) smoothing, and the correction of a text is the posterior
mean ``E[gold | predicted class]`` under the training class priors. With two classes the
family also corrects a predicted class share for misclassification (Rogan-Gladen).

Posterior means are a relabelling of the predicted classes: for two classes the correction
cannot change the order of the texts. The family addresses level displacement, in the sense
of the probability of each class per text, and the bias of class shares for the estimate.
It applies only when the gold values are few and well populated and the score separates
them (minimum informedness, Youden's J). ``applies`` judges the labelled rows and ``fit``
the training rows with one check, so a fit below a floor is refused on either path.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from copernicus.corrections._proxy import check_count
from copernicus.corrections.data import CorrectionData, best_score_column
from copernicus.corrections.families._fitted import (
    labelled_positions,
    require_columns,
    row_positions,
    training_rows,
)
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.errors import (
    ConfigurationError,
    EstimatorError,
    NotApplicableError,
    SettingsError,
)

logger = logging.getLogger(__name__)

__all__ = ["MisclassificationFitted", "MisclassificationMatrix", "make_misclassification"]

FAMILY = "misclassification_matrix"
VARIANT = "posterior"
ERRORS_ADDRESSED = ("level_displacement",)
MIN_CLASSES = 2
SINGULAR_TOLERANCE = 1e-12


def _thresholds(score: np.ndarray, gold: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Quantiles of the score at the cumulative gold class shares (all but the last)."""
    shares = np.array([(gold == value).mean() for value in classes])
    return np.quantile(score, np.cumsum(shares)[:-1])


def _discretize(score: np.ndarray, thresholds: np.ndarray, n_classes: int) -> np.ndarray:
    """Class index of each score: the number of thresholds it reaches, capped at the last class."""
    return np.clip(np.digitize(score, thresholds), 0, n_classes - 1)


def _informedness(predicted: np.ndarray, gold_index: np.ndarray, n_classes: int) -> float:
    """Youden's J for several classes: ``(k * balanced accuracy - 1) / (k - 1)``."""
    recalls = [
        float((predicted[gold_index == index] == index).mean())
        for index in range(n_classes)
        if (gold_index == index).any()
    ]
    return (n_classes * float(np.mean(recalls)) - 1.0) / (n_classes - 1.0)


def _label(value: float) -> str:
    """A class value for a message."""
    return f"{value:g}"


@dataclass(frozen=True, eq=False)
class MisclassificationFitted:
    """A fitted misclassification-matrix correction.

    Attributes:
        column: The score column that is discretised.
        classes: The sorted class values of the training gold.
        thresholds: Cut points of the score between consecutive classes.
        confusion: ``confusion[i, j]`` is the smoothed ``P(predicted i | true j)``.
        priors: Share of each class among the training gold values.
        posterior_means: ``E[gold | predicted class i]`` for each class.
        provenance: What was fitted; stored as a read-only mapping.
    """

    column: str
    classes: np.ndarray
    thresholds: np.ndarray
    confusion: np.ndarray
    priors: np.ndarray
    posterior_means: np.ndarray
    provenance: MappingProxyType[str, object]

    def _predicted_index(self, data: CorrectionData, ids: np.ndarray) -> np.ndarray:
        """Class index of the discretised score of the named rows."""
        require_columns(data, (self.column,))
        positions = row_positions(data, ids)
        score = data.frame[self.column].to_numpy(dtype=float)[positions]
        return _discretize(score, self.thresholds, self.classes.size)

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        """The posterior mean of the discretised score of the named rows.

        Args:
            data: The correction data holding the rows.
            ids: Text ids of the rows.

        Returns:
            Floats on the gold scale indexed by the ids.

        Raises:
            ConfigurationError: If an id is not in the data.
        """
        values = self.posterior_means[self._predicted_index(data, ids)]
        return pd.Series(values, index=pd.Index(np.asarray(ids).astype(str)))

    def class_shares(self, data: CorrectionData, ids: np.ndarray) -> dict[str, object]:
        """Class shares of the named rows, raw and (for two classes) corrected.

        Args:
            data: The correction data holding the rows.
            ids: Text ids of the rows.

        Returns:
            ``raw``, the share of each predicted class keyed by the class value. For exactly
            two classes also ``rogan_gladen`` (the corrected share of the larger class,
            clipped to ``[0, 1]``, or NaN with ``rogan_gladen_note`` when sensitivity minus
            the false-positive rate is 0), ``sensitivity`` and ``specificity`` (the smoothed
            rates of the training rows).

        Raises:
            ConfigurationError: If no id is given or an id is not in the data.
        """
        if np.asarray(ids).size == 0:
            raise ConfigurationError("class shares need at least one text id")
        predicted = self._predicted_index(data, ids)
        raw = {
            float(value): float((predicted == index).mean())
            for index, value in enumerate(self.classes)
        }
        shares: dict[str, object] = {"raw": raw}
        if self.classes.size != MIN_CLASSES:
            return shares
        sensitivity, specificity = float(self.confusion[1, 1]), float(self.confusion[0, 0])
        shares["sensitivity"], shares["specificity"] = sensitivity, specificity
        denominator = sensitivity - (1.0 - specificity)
        if abs(denominator) <= SINGULAR_TOLERANCE:
            shares["rogan_gladen"] = float("nan")
            shares["rogan_gladen_note"] = (
                "sensitivity equals the false-positive rate: the classifier carries no "
                "information and the corrected share is undefined"
            )
        else:
            observed = raw[float(self.classes[1])]
            shares["rogan_gladen"] = float(
                np.clip((observed - (1.0 - specificity)) / denominator, 0.0, 1.0)
            )
        return shares


@dataclass(frozen=True)
class MisclassificationMatrix:
    """Confusion-matrix correction of one discretised score column.

    Attributes:
        youden_floor: Minimum informedness (Youden's J) of the discretised best score
            column on the labelled rows, in ``[0, 1]``.
        per_class_floor: Minimum number of labelled rows in each class, at least 1.
        max_classes: Maximum number of classes, at least 2.
    """

    youden_floor: float = 0.3
    per_class_floor: int = 30
    max_classes: int = 5
    family: ClassVar[str] = FAMILY
    variant: ClassVar[str] = VARIANT
    target: ClassVar[Literal["A", "B"]] = "A"

    def __post_init__(self) -> None:
        """Refuse parameters outside their range; store plain numbers.

        Raises:
            SettingsError: If ``youden_floor`` is not a number in ``[0, 1]``, ``per_class_floor``
                is not an integer of at least 1, or ``max_classes`` is not an integer of at
                least 2 (a boolean is refused everywhere, a float for the two integers).
        """
        floor = self.youden_floor
        if (
            isinstance(floor, bool)
            or not isinstance(floor, numbers.Real)
            or not 0.0 <= float(floor) <= 1.0
        ):
            raise SettingsError(f"youden_floor must be a number between 0 and 1; got {floor!r}")
        object.__setattr__(self, "youden_floor", float(floor))
        per_class = check_count("per_class_floor", self.per_class_floor, 1)
        object.__setattr__(self, "per_class_floor", per_class)
        object.__setattr__(
            self, "max_classes", check_count("max_classes", self.max_classes, MIN_CLASSES)
        )

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """Whether the labelled rows support a misclassification matrix.

        ``fit`` makes the same check on its training rows and refuses rows that fail it.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``, or ``(False, reason)`` naming the number of classes, the class
            below the per-class floor with its count, the informedness below its floor, or
            the absence of a usable score column.
        """
        labelled = data.labelled_mask
        return self._supported(data.gold[labelled], data.scores[labelled].reset_index(drop=True))

    def _supported(self, gold: np.ndarray, scores: pd.DataFrame) -> tuple[bool, str]:
        """Whether labelled rows support the matrix; the one check of ``applies`` and ``fit``.

        The rows need between 2 and ``max_classes`` classes, at least ``per_class_floor`` rows
        in each, a usable score column, and an informedness (Youden's J) of the best column,
        cut at the gold class shares, of at least ``youden_floor``.

        Args:
            gold: The gold values of the rows, without NaN.
            scores: Their score columns, aligned with ``gold``.

        Returns:
            ``(True, "")``, or ``(False, reason)`` naming the number of classes, the class
            below the per-class floor with its count, the informedness below its floor, or
            the absence of a usable score column.
        """
        classes = np.unique(gold)
        if classes.size < MIN_CLASSES or classes.size > self.max_classes:
            return False, (
                f"the labelled gold takes {classes.size} distinct values; the matrix needs "
                f"between {MIN_CLASSES} and {self.max_classes}"
            )
        for value in classes:
            count = int((gold == value).sum())
            if count < self.per_class_floor:
                return False, (
                    f"class {_label(float(value))} has {count} labelled rows; at least "
                    f"{self.per_class_floor} are needed"
                )
        try:  # an unusable column is reported as the reason, as the contract requires
            column = best_score_column(scores, gold)
        except EstimatorError as error:
            return False, f"no informedness can be computed: {error}"
        score = scores[column].to_numpy(dtype=float)
        gold_index = np.searchsorted(classes, gold)
        predicted = _discretize(score, _thresholds(score, gold, classes), classes.size)
        informedness = _informedness(predicted, gold_index, classes.size)
        if informedness < self.youden_floor:
            return False, (
                f"the informedness (Youden's J) of score column {column!r} is "
                f"{informedness:.3f}, below the floor {self.youden_floor:g}"
            )
        return True, ""

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> MisclassificationFitted:
        """Estimate the confusion matrix and the posterior means on the training rows.

        The training rows must pass the check of ``applies`` with the floors of this
        correction, whatever ``applies`` said about other rows: a lower ``per_class_floor``
        or ``youden_floor`` is a setting of the correction, not of the call.

        Args:
            data: The correction data.
            train_ids: Ids of the labelled rows to learn from.
            rng: Unused; the fit is deterministic.

        Returns:
            The fitted correction; its provenance holds ``youden_j``, ``n_classes``,
            ``classes``, ``score_column`` and the confusion matrix as nested lists.

        Raises:
            ConfigurationError: If an id is unknown, repeated or has no gold value.
            NotApplicableError: If the training rows take fewer than 2 or more than
                ``max_classes`` values, a class has fewer than ``per_class_floor`` rows, no
                score column is usable, or the informedness is below ``youden_floor``; the
                message gives the reason ``applies`` would give.
        """
        del rng
        positions, gold = labelled_positions(data, train_ids)
        scores = data.scores.iloc[positions].reset_index(drop=True)
        supported, reason = self._supported(gold, scores)
        if not supported:
            raise NotApplicableError(
                f"{FAMILY}/{VARIANT} does not apply to the training rows: {reason}"
            )
        rows = training_rows(data, train_ids)
        classes = np.unique(rows.gold)
        n_classes = int(classes.size)
        thresholds = _thresholds(rows.score, rows.gold, classes)
        predicted = _discretize(rows.score, thresholds, n_classes)
        gold_index = np.searchsorted(classes, rows.gold)
        counts = np.array([(gold_index == index).sum() for index in range(n_classes)], dtype=float)
        priors = counts / rows.gold.size
        confusion = np.empty((n_classes, n_classes))
        for true in range(n_classes):
            hits = np.bincount(predicted[gold_index == true], minlength=n_classes)
            confusion[:, true] = (hits + 1.0) / (counts[true] + n_classes)
        joint = confusion * priors
        posterior_means = (joint * classes).sum(axis=1) / joint.sum(axis=1)
        informedness = _informedness(predicted, gold_index, n_classes)
        return MisclassificationFitted(
            column=rows.column,
            classes=classes,
            thresholds=thresholds,
            confusion=confusion,
            priors=priors,
            posterior_means=posterior_means,
            provenance=MappingProxyType(
                {
                    "family": FAMILY,
                    "variant": VARIANT,
                    "score_column": rows.column,
                    "youden_j": informedness,
                    "n_classes": n_classes,
                    "classes": [float(value) for value in classes],
                    "thresholds": [float(value) for value in thresholds],
                    "class_priors": [float(value) for value in priors],
                    "confusion_matrix": confusion.tolist(),
                }
            ),
        )


def make_misclassification(
    variant: str,
    *,
    youden_floor: float = 0.3,
    per_class_floor: int = 30,
    max_classes: int = 5,
) -> MisclassificationMatrix:
    """Build the correction named by the registry.

    Args:
        variant: Must be ``posterior``.
        youden_floor: Minimum informedness on the labelled rows.
        per_class_floor: Minimum number of labelled rows per class.
        max_classes: Maximum number of classes.

    Returns:
        The correction.

    Raises:
        SettingsError: If the variant is not ``posterior`` or a parameter is out of range.
    """
    if variant != VARIANT:
        raise SettingsError(f"unknown variant {variant!r} of family {FAMILY!r}; use {VARIANT!r}")
    return MisclassificationMatrix(
        youden_floor=youden_floor, per_class_floor=per_class_floor, max_classes=max_classes
    )


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=(VARIANT,),
        target="A, C",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors="bias of class shares",
        factory=make_misclassification,
    )
)
