"""Design-based pseudo-values: the gold value of a sample, spread over every text.

A human-coded sample drawn with known inclusion probabilities ``pi`` (the group sample of
``draw_human_sample``) supports an unbiased estimate of a population mean even when the
machine score is biased. With a prediction ``ghat`` of the gold value for every text, the
pseudo-value of text ``i`` is::

    rect_i = ghat_i + (D_i / pi_i) * (gold_i - ghat_i)

where ``D_i`` is 1 for a labelled text and 0 otherwise. The mean of ``rect`` over all texts is
the Horvitz-Thompson estimate of the mean of the gold values when ``ghat`` was fixed before the
labels were seen (the ``identity`` rectifier with a named score column that is already on the
gold scale), and the mean of ``rect`` over all possible samples is exactly the mean of the gold
values. The ``calibrated`` and ``crossfit`` rectifiers learn ``ghat`` from the labelled rows,
which leaves a small bias of order one over the sample size in exchange for a smaller variance.

A labelled sample without inclusion probabilities cannot be rectified: pass the sample drawn
by ``draw_human_sample`` through ``attach_draw``, or supply a ``pi`` column.

The prediction-powered tuning coefficient ``lambda_star = cov(gold, ghat) / var(ghat)`` on the
labelled rows uses population moments (divisor ``n``) in both terms, with no floor and no cap.
It is undefined when ``ghat`` is constant there (``is_constant``): ``tuning_coefficient`` then
refuses, and ``pseudo_values`` reports ``lambda_star`` and the tuned mean as NaN, gives the
reason in ``lambda_star_reason`` and ``mean_rectified_tuned_reason``, and logs a warning. The
power-tuned mean is ``mean(gold_lab) + lambda_star * (mean(ghat_all) - mean(ghat_lab))``. It
weights every labelled text alike, so it assumes one ``pi`` on the labelled rows; when ``pi``
differs between them it is not design-consistent and is reported as NaN with that reason
(``mean_rectified`` remains the unbiased mean).

The estimate-level functions for means, differences and coefficients reuse the inclusion
probability check (``inclusion_probabilities``) and the tuning coefficient
(``tuning_coefficient``) of this module.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.data import CorrectionData, best_score_column
from copernicus.corrections.families._crossfit_ghat import crossfit_ghat
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.corrections.families.calibration import fit_ols_line
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["PseudoValues", "inclusion_probabilities", "pseudo_values", "tuning_coefficient"]

FAMILY = "pseudo_values"
VARIANTS = ("identity", "calibrated", "crossfit")
MIN_FOLDS = 2
ESTIMATE_ERRORS = (
    "estimate off and confidence-interval misses, for means and for slopes on any covariate "
    "with the construct as outcome"
)


@dataclass(frozen=True, eq=False)
class PseudoValues:
    """The pseudo-value column of a human-coded sample and the means built from it.

    Attributes:
        column: The pseudo-value of every text, indexed by text id in data order.
        ghat: The prediction of the gold value of every text, same index.
        lambda_star: The tuning coefficient ``cov(gold, ghat) / var(ghat)`` on the labelled
            rows (population moments); NaN when ``ghat`` is constant there (``is_constant``),
            where the coefficient is undefined.
        mean_plugin: The mean of ``ghat`` over all texts.
        mean_rectified: The mean of ``column`` over all texts.
        mean_rectified_tuned: ``mean(gold_lab) + lambda_star * (mean(ghat_all) -
            mean(ghat_lab))``; NaN when ``lambda_star`` is NaN, and when ``pi`` differs between
            labelled rows, because the formula weights every labelled text alike.
        rectifier: ``identity``, ``calibrated`` or ``crossfit``.
        score_column: The score column ``ghat`` reads; for ``crossfit`` the distinct columns
            the folds chose, sorted and joined with ``|``.
        n_labelled: Number of labelled texts.
        lambda_star_reason: ``None`` when ``lambda_star`` is defined; otherwise one sentence
            that says why it is missing.
        mean_rectified_tuned_reason: ``None`` when ``mean_rectified_tuned`` is defined;
            otherwise one sentence that says why it is missing and names every cause.
        n_rank_deficient_folds: For ``crossfit``, the number of inner folds whose design
            ``[1, score, covariate]`` has a numerical rank below 3 on their labelled training
            rows, as ``numpy.linalg.lstsq`` with ``rcond=None`` returns it (a warning names
            them); 0 when there is none, and for ``identity`` and ``calibrated``, which have no
            folds.
    """

    column: pd.Series
    ghat: pd.Series
    lambda_star: float
    mean_plugin: float
    mean_rectified: float
    mean_rectified_tuned: float
    rectifier: str
    score_column: str
    n_labelled: int
    lambda_star_reason: str | None = None
    mean_rectified_tuned_reason: str | None = None
    n_rank_deficient_folds: int = 0


def inclusion_probabilities(
    data: CorrectionData, labelled: np.ndarray, *, purpose: str = "pseudo-values"
) -> np.ndarray:
    """The inclusion probabilities of the data, refusing a labelled row that has none.

    Args:
        data: The correction data.
        labelled: Boolean mask of the labelled rows.
        purpose: What needs the probabilities, for the message (a plural noun phrase).

    Returns:
        The ``pi`` column, NaN where a row is unlabelled and carries none.

    Raises:
        EstimatorError: If the data have no ``pi`` column or a labelled row has a non-finite
            ``pi``; the message names ``pi`` and points to ``draw_human_sample`` and
            ``attach_draw``.
    """
    pi = data.pi
    if pi is None or not np.isfinite(pi[labelled]).all():
        raise EstimatorError(
            f"{purpose} need the inclusion probability pi on every labelled row; the data have "
            + ("no pi column" if pi is None else "a non-finite pi on a labelled row")
            + ": draw the sample with draw_human_sample and attach it with attach_draw, "
            "or supply pi"
        )
    return pi


def _named_column(data: CorrectionData, score_column: str | None) -> str | None:
    """The requested score column, checked against the data's score columns."""
    if score_column is not None and score_column not in data.score_columns:
        raise ConfigurationError(
            f"score column {score_column!r} is missing from the data; score columns: "
            f"{list_first_ten(list(data.score_columns))}"
        )
    return score_column


def _calibrated_ghat(data: CorrectionData, score_column: str | None) -> tuple[np.ndarray, str]:
    """The line of the gold values on the score, fitted on the labelled rows."""
    labelled = data.labelled_mask
    gold = data.gold[labelled]
    column = score_column or best_score_column(data.scores.loc[labelled], gold)
    scores = data.frame[column].to_numpy(dtype=float)
    line = fit_ols_line(scores[labelled], gold, column)
    return line.predict(scores), column


def _predictions(
    data: CorrectionData, rectifier: str, score_column: str | None, k: int, seed: int
) -> tuple[np.ndarray, str, int]:
    """The prediction of every row, the column it reads and its folds of numerical rank below 3."""
    if rectifier == "identity":
        column = score_column or data.score_columns[0]
        return data.frame[column].to_numpy(dtype=float), column, 0
    if rectifier == "calibrated":
        calibrated, column = _calibrated_ghat(data, score_column)
        return calibrated, column, 0
    found = crossfit_ghat(data, k=k, seed=seed)
    return found.ghat, "|".join(sorted(set(found.columns))), len(found.rank_deficient_folds)


def _constant_prediction(ghat_labelled: np.ndarray) -> str:
    """Why the tuning coefficient is undefined on the labelled rows, or ``""`` when it is not.

    The coefficient divides by the variance of the prediction, so it is undefined where the
    prediction is constant by the library's rule (``is_constant``).
    """
    if not is_constant(ghat_labelled):
        return ""
    low, high = float(np.min(ghat_labelled)), float(np.max(ghat_labelled))
    values = (
        f"each is {low!r}" if low == high else f"from {low!r} to {high!r}, equal up to rounding"
    )
    return f"the prediction is constant on the labelled rows ({ghat_labelled.size} rows, {values})"


def _missing_reasons(constant: str, unequal_pi: np.ndarray | None) -> tuple[str | None, str | None]:
    """Why ``lambda_star`` and ``mean_rectified_tuned`` are missing, ``None`` where defined.

    Args:
        constant: What ``_constant_prediction`` found, ``""`` when the weight is defined.
        unequal_pi: The inclusion probabilities of the labelled rows when they differ, else
            ``None``.

    Returns:
        One sentence for the weight and one for the tuned mean, which names every cause.
    """
    causes: list[str] = []
    if constant:
        causes.append(f"its weight is undefined, as {constant}")
    if unequal_pi is not None:
        causes.append(
            "it weights every labelled text alike and so needs one inclusion probability on the "
            f"labelled rows, where pi ranges from {float(unequal_pi.min()):.4g} to "
            f"{float(unequal_pi.max()):.4g}"
        )
    weight_reason = (
        f"The tuning weight cov(gold, ghat) / var(ghat) is undefined because {constant}."
        if constant
        else None
    )
    tuned_reason = (
        "The tuned mean is missing because " + ", and because ".join(causes) + "."
        if causes
        else None
    )
    return weight_reason, tuned_reason


def tuning_coefficient(
    ghat_labelled: np.ndarray, gold_labelled: np.ndarray, *, purpose: str = "power tuning"
) -> float:
    """The power-tuning coefficient ``lambda_star = cov(gold, ghat) / var(ghat)``.

    Both moments are population moments (divisor ``n``) over the labelled rows. The
    coefficient has no floor and no cap: it is negative when gold and prediction covary
    negatively, and it exceeds 1 when their covariance exceeds the variance of the prediction,
    as for a prediction that follows the gold on a compressed scale.

    Args:
        ghat_labelled: The prediction on the labelled rows.
        gold_labelled: The gold values on the same rows.
        purpose: What needs the coefficient, for the message (such as an estimator's name).

    Returns:
        The coefficient.

    Raises:
        EstimatorError: If no labelled row is given, or the prediction is constant on those
            rows (``is_constant``), where the coefficient is undefined; the message names the
            rows and what to do.
    """
    if ghat_labelled.size == 0:
        raise EstimatorError(
            f"{purpose} needs the tuning weight cov(gold, prediction) / var(prediction) over the "
            "labelled rows, and no labelled row was given: pass the prediction and the gold "
            "values of the human-coded texts"
        )
    constant = _constant_prediction(ghat_labelled)
    if constant:
        raise EstimatorError(
            f"{purpose} needs the tuning weight cov(gold, prediction) / var(prediction) over the "
            f"labelled rows, which is undefined because {constant}: estimate without power "
            "tuning (power_tuning=False), or with a prediction that varies on the labelled rows"
        )
    covariance = float(
        np.mean((gold_labelled - gold_labelled.mean()) * (ghat_labelled - ghat_labelled.mean()))
    )
    return covariance / float(np.var(ghat_labelled))


def pseudo_values(
    data: CorrectionData,
    *,
    rectifier: Literal["identity", "calibrated", "crossfit"] = "identity",
    score_column: str | None = None,
    k: int = 5,
    seed: int = 42,
) -> PseudoValues:
    """The pseudo-values of a human-coded sample with known inclusion probabilities.

    ``identity`` uses a score column as the prediction as it stands, so the column must
    already be on the gold scale (a calibrated or rescaled column otherwise); with no
    ``score_column`` it is the first score column, which is never chosen on gold values.
    ``calibrated`` fits ``gold ~ 1 + score`` by least squares on the labelled rows, with the
    column ``score_column`` or else the one with the highest signed Spearman correlation with
    the gold values on those rows. ``crossfit`` predicts every row from a model that did not
    see its group's gold values (``gold ~ 1 + score + covariate`` fitted by grouped inner
    folds, the column chosen on a group-level half of each fold's labelled groups); it ignores
    ``score_column``. A fold whose design has a numerical rank below 3 on its labelled training
    rows, as ``numpy.linalg.lstsq`` with ``rcond=None`` returns it, keeps the coefficients that
    ``lstsq`` returns, the solution of smallest norm of the directions above its cutoff;
    ``n_rank_deficient_folds`` counts such folds, and a warning names them (within
    ``repeated_study`` one warning per size sums such warnings up).

    Where ``ghat`` is constant on the labelled rows, the tuning coefficient is undefined:
    ``lambda_star`` and ``mean_rectified_tuned`` are NaN, ``lambda_star_reason`` and
    ``mean_rectified_tuned_reason`` say why and a warning is logged, while the column and
    ``mean_rectified``, which do not use the coefficient, are computed as always.

    Args:
        data: The correction data with gold on the labelled rows and ``pi`` on every
            labelled row (unlabelled rows may carry NaN).
        rectifier: The way the prediction ``ghat`` is made.
        score_column: The score column ``identity`` reads or ``calibrated`` fits; ``None`` for
            the first column (``identity``) or the best column (``calibrated``).
        k: Number of inner folds of ``crossfit``, at least 2.
        seed: Seed of the folds and of the selection halves of ``crossfit``.

    Returns:
        The pseudo-values.

    Raises:
        SettingsError: If the rectifier is unknown or ``k`` is below 2.
        ConfigurationError: If ``score_column`` is not a score column of the data.
        EstimatorError: If ``pi`` is missing or not finite on a labelled row (draw the sample with
            ``draw_human_sample`` and ``attach_draw``, or supply ``pi``), no text is labelled,
            a line cannot be fitted or a score column chosen, or ``crossfit`` meets
            ``inner_groups_below_k``, ``fold_too_thin`` or
            ``nested_selection_half_degenerate``.
    """
    if rectifier not in VARIANTS:
        raise SettingsError(
            f"unknown rectifier {rectifier!r} of family {FAMILY!r}; use one of "
            f"{', '.join(VARIANTS)}"
        )
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < MIN_FOLDS:
        raise SettingsError(f"k must be an integer of at least {MIN_FOLDS}; got {k!r}")
    labelled = data.labelled_mask
    if not labelled.any():
        raise EstimatorError("pseudo-values need at least one labelled text; none carries gold")
    pi = inclusion_probabilities(data, labelled)
    ghat, column_name, rank_deficient = _predictions(
        data, rectifier, _named_column(data, score_column), k, seed
    )
    gold = data.gold
    correction = np.zeros(data.n)
    correction[labelled] = (gold[labelled] - ghat[labelled]) / pi[labelled]
    rectified = ghat + correction
    ghat_labelled, gold_labelled = ghat[labelled], gold[labelled]
    mean_plugin = float(ghat.mean())
    constant = _constant_prediction(ghat_labelled)
    if constant:
        logger.warning(
            "pseudo-values (%s): the tuning weight cov(gold, ghat) / var(ghat) is undefined "
            "because %s, so lambda_star and mean_rectified_tuned are NaN; the column and "
            "mean_rectified do not use the weight. A tuned mean needs a prediction that varies "
            "on the labelled rows",
            rectifier,
            constant,
        )
        lambda_star = tuned = float("nan")
    else:
        lambda_star = tuning_coefficient(ghat_labelled, gold_labelled)
        tuned = float(gold_labelled.mean() + lambda_star * (mean_plugin - ghat_labelled.mean()))
    unequal = not is_constant(pi[labelled])
    if unequal:
        logger.info(
            "the power-tuned mean weights every labelled text alike and needs one pi on the "
            "labelled rows; pi ranges from %.4g to %.4g, so mean_rectified_tuned is NaN",
            float(pi[labelled].min()),
            float(pi[labelled].max()),
        )
        tuned = float("nan")
    weight_reason, tuned_reason = _missing_reasons(constant, pi[labelled] if unequal else None)
    index = pd.Index(data.ids)
    logger.debug(
        "pseudo-values (%s) on %d labelled of %d texts", rectifier, int(labelled.sum()), data.n
    )
    return PseudoValues(
        column=pd.Series(rectified, index=index),
        ghat=pd.Series(ghat, index=index),
        lambda_star=lambda_star,
        mean_plugin=mean_plugin,
        mean_rectified=float(rectified.mean()),
        mean_rectified_tuned=tuned,
        rectifier=rectifier,
        score_column=column_name,
        n_labelled=int(labelled.sum()),
        lambda_star_reason=weight_reason,
        mean_rectified_tuned_reason=tuned_reason,
        n_rank_deficient_folds=rank_deficient,
    )


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="C",
        level="estimate",
        errors_addressed=(),
        estimate_errors=ESTIMATE_ERRORS,
        factory=None,
    )
)
