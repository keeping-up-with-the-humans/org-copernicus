"""The four score errors: level, calibration, covariate-linked error and noise."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from copernicus.errors import ConfigurationError
from copernicus.validity.metrics import finite_pairs, is_constant, minmax01

logger = logging.getLogger(__name__)

__all__ = [
    "ScoreErrors",
    "constant_score_errors",
    "covariate_slope",
    "replicate_noise",
    "score_errors",
]

CovariateScale = Literal["standardized", "raw"]
COVARIATE_SCALES: tuple[CovariateScale, ...] = ("standardized", "raw")
NOISE_SCALES = ("raw", "observed")


@dataclass(frozen=True)
class ScoreErrors:
    """Errors of a score against gold on the 0-1 scale (both scaled by their observed range).

    Attributes:
        level_displacement: Mean of ``score01 - gold01``.
        calibration_shortfall: ``cov(score01, gold01) / var(gold01) - 1``; negative when the
            score is compressed relative to gold.
        covariate_linked_error: Least-squares slope of ``score01 - gold01`` on the covariate;
            positive means the score rises relative to gold as the covariate rises. NaN when
            the covariate is constant.
        covariate_scale: ``standardized`` (per standard deviation of the covariate) or
            ``raw`` (per unit of the covariate).
        mae_01: Mean absolute value of ``score01 - gold01``.
        signed_gap_01: Mean signed value of ``score01 - gold01``.
        n: Number of rows used.
    """

    level_displacement: float
    calibration_shortfall: float
    covariate_linked_error: float
    covariate_scale: str
    mae_01: float
    signed_gap_01: float
    n: int


def _slope(response: np.ndarray, predictor: np.ndarray) -> float:
    """Least-squares slope in closed form; NaN when the predictor is constant."""
    if is_constant(predictor):
        return float("nan")
    centred = predictor - predictor.mean()
    return float(np.sum(centred * (response - response.mean())) / np.sum(centred**2))


def _check_covariate_scale(covariate_scale: str) -> None:
    if covariate_scale not in COVARIATE_SCALES:
        raise ConfigurationError(
            f"covariate_scale must be one of {COVARIATE_SCALES}; got {covariate_scale!r}"
        )


def covariate_slope(diff: np.ndarray, covariate: np.ndarray, scale: str = "standardized") -> float:
    """The covariate-linked error: the least-squares slope of a signed error on the covariate.

    The slope uses sample moments (ddof=1) over the rows where both arrays are finite. On the
    ``standardized`` scale it is per standard deviation of the covariate (the slope on the
    z-scored covariate, which equals ``cov(diff, z)``); on the ``raw`` scale it is per unit.
    ``score_errors`` reports this slope for ``score01 - gold01``, and the correction battery
    for the train-scaled errors of the held-out texts.

    Args:
        diff: The signed error of every text, such as ``score01 - gold01``.
        covariate: The covariate of every text (for example log characters), of the same
            length.
        scale: ``standardized`` or ``raw``.

    Returns:
        The slope, or NaN when the covariate is constant over the finite rows.

    Raises:
        ConfigurationError: If ``scale`` is unknown, or the arrays are not one-dimensional and
            of equal length.
    """
    _check_covariate_scale(scale)
    response, predictor = finite_pairs(diff, covariate)
    slope = _slope(response, predictor)
    if scale == "standardized" and not np.isnan(slope):
        slope *= float(predictor.std(ddof=1))
    return slope


def score_errors(
    score: np.ndarray,
    gold: np.ndarray,
    covariate: np.ndarray,
    *,
    covariate_scale: Literal["standardized", "raw"] = "standardized",
) -> ScoreErrors:
    """The score errors of ``score`` against ``gold`` and ``covariate``.

    Score and gold are each min-max scaled with their own observed minimum and maximum
    over the rows where score, gold and covariate are all finite, as in the paper's
    analysis. The calibration shortfall is therefore relative to those observed ranges: an
    exact affine compression of gold shows no shortfall, while noise in the score does.
    The covariate-linked error is the sample-variance (ddof=1) least-squares slope of the
    signed error on the covariate; on the ``standardized`` scale the covariate is z-scored
    first, which equals ``cov(diff, z)``.

    Args:
        score: Score values.
        gold: Gold values, of the same length.
        covariate: Covariate values (for example log characters), of the same length.
        covariate_scale: ``standardized`` for the slope per standard deviation of the
            covariate, ``raw`` for the slope per unit.

    Returns:
        The four errors with the companion mean absolute error and signed gap.

    Raises:
        ConfigurationError: If ``covariate_scale`` is unknown or the lengths differ.
        EstimatorError: If score or gold has zero range over the finite rows.
    """
    _check_covariate_scale(covariate_scale)
    pair_score, pair_gold, pair_covariate = finite_pairs(score, gold, covariate)
    score01 = minmax01(pair_score)
    gold01 = minmax01(pair_gold)
    difference = score01 - gold01
    slope = covariate_slope(difference, pair_covariate, covariate_scale)
    gold_variance = float(np.var(gold01, ddof=1))
    shortfall = float(np.cov(score01, gold01, ddof=1)[0, 1]) / gold_variance - 1.0
    return ScoreErrors(
        level_displacement=float(difference.mean()),
        calibration_shortfall=shortfall,
        covariate_linked_error=slope,
        covariate_scale=covariate_scale,
        mae_01=float(np.abs(difference).mean()),
        signed_gap_01=float(difference.mean()),
        n=int(pair_score.size),
    )


def constant_score_errors(
    gold: np.ndarray,
    covariate: np.ndarray,
    *,
    covariate_scale: Literal["standardized", "raw"] = "standardized",
) -> ScoreErrors:
    """The two score errors a constant score keeps, with the other two missing.

    A score that is the same for every text has no 0-1 scale of its own: its zero range
    leaves the level displacement, the mean absolute error and the signed gap undefined
    (NaN). Two errors do not depend on where the constant sits: it does not move with gold,
    so the calibration shortfall is exactly -1, and its signed error ``c - gold01`` moves
    with the covariate only through gold, so the covariate-linked error is minus the slope
    of ``gold01`` on the covariate (per standard deviation of the covariate on the
    ``standardized`` scale, NaN for a constant covariate). Gold is min-max scaled over the
    rows where gold and covariate are finite.

    Args:
        gold: Gold values of the texts with the constant score.
        covariate: Covariate values, of the same length.
        covariate_scale: ``standardized`` or ``raw``, as in `score_errors`.

    Returns:
        The errors, with ``n`` the number of rows used.

    Raises:
        ConfigurationError: If ``covariate_scale`` is unknown or the lengths differ.
        EstimatorError: If gold has zero range over the finite rows.
    """
    _check_covariate_scale(covariate_scale)
    pair_gold, pair_covariate = finite_pairs(gold, covariate)
    gold01 = minmax01(pair_gold)
    nan = float("nan")
    return ScoreErrors(
        level_displacement=nan,
        calibration_shortfall=-1.0,
        covariate_linked_error=-covariate_slope(gold01, pair_covariate, covariate_scale),
        covariate_scale=covariate_scale,
        mae_01=nan,
        signed_gap_01=nan,
        n=int(pair_gold.size),
    )


def _observed_scale(values: np.ndarray) -> np.ndarray:
    """Min-max scale finite values to 0-1; constant values have no spread and become 0."""
    finite = values[np.isfinite(values)]
    if finite.size == 0 or is_constant(finite):
        return np.where(np.isfinite(values), 0.0, np.nan)
    return minmax01(values)


def replicate_noise(
    frame: pd.DataFrame,
    *,
    text_id: str = "text_id",
    run: str = "run_number",
    score: str = "score",
    scale: Literal["raw", "observed"] = "raw",
) -> float:
    """Mean over texts of the standard deviation of the score across replicate runs.

    The standard deviation is the sample one (ddof=1) per text. A text with a single
    run contributes nothing, and the result is NaN when no text has two runs. With
    ``scale="observed"`` the scores are first min-max scaled to 0-1 with the observed
    minimum and maximum over every finite row of the frame (all texts and runs, a text
    with a single run included), which puts the noise on the 0-1 scale of the other score
    errors; scores that are all the same have no run-to-run variation and give 0.

    Args:
        frame: One row per text and run.
        text_id: Column naming the text.
        run: Column naming the replicate run.
        score: Column holding the score.
        scale: ``raw`` keeps the score's own scale; ``observed`` scales it to 0-1 first.

    Returns:
        The mean per-text standard deviation, or NaN.

    Raises:
        ConfigurationError: If one of the three columns is missing from the frame or
            ``scale`` is unknown.
    """
    for column in (text_id, run, score):
        if column not in frame.columns:
            raise ConfigurationError(
                f"replicate_noise needs column {column!r}; the frame has {list(frame.columns)}"
            )
    if scale not in NOISE_SCALES:
        raise ConfigurationError(f"scale must be one of {NOISE_SCALES}; got {scale!r}")
    if scale == "observed":
        values = _observed_scale(frame[score].to_numpy(dtype="float64", na_value=np.nan))
        frame = pd.DataFrame({text_id: frame[text_id].to_numpy(), score: values})
    deviations = frame.groupby(text_id)[score].std(ddof=1).dropna()
    if deviations.empty:
        return float("nan")
    return float(deviations.mean())
