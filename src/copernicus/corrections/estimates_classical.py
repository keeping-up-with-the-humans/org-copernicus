"""Moment subtraction and SIMEX: the construct's coefficient as predictor under classical error.

Both estimate the slope ``beta`` in ``y = a + beta V + e`` of an outcome ``y`` observed on every
text on the construct ``V``, from the classical proxy ``W`` of the reverse calibration on the
labelled rows (see ``_proxy``) and its error variance ``sigma2 = var(W - g)`` there:

* moment subtraction (Fuller): ``beta = cov(W, y) / (var(W) - sigma2)`` over every text, undefined
  when ``var(W) <= sigma2`` (the corrected variance of the construct is not positive).
* SIMEX (Cook and Stefanski): at each multiple ``lambda`` of ``(0, 0.5, 1, 1.5, 2)``, ``reps``
  replicates add ``N(0, lambda * sigma2)`` noise to ``W`` and their slopes
  ``cov(W_noisy, y) / var(W_noisy)`` are averaged; the curve is extrapolated to ``lambda = -1``
  by the rational extrapolant of ``_simex``. The statuses ``pole_le_1`` and ``no_fit`` leave the
  estimate undefined; ``constant`` (no error left to remove) and ``linear_limit`` keep it and are
  reported.

The standard error of both is the standard deviation over a stratified bootstrap that refits
the reverse calibration, ``sigma2`` and (for SIMEX, with ``boot_reps`` noise replicates) the
curve on every resample. The bootstrap resamples texts, not groups, so it treats the texts as
independent. The labelled rows serve only to fit the error model, so they must be a random sample
of the texts; ``pi`` is not used. Moments are sample moments throughout. The score
column defaults to the first one, which is never chosen on gold values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from copernicus.corrections._estimate_inputs import outcome_values, require_labelled, score_values
from copernicus.corrections._proxy import (
    MIN_FINITE_BOOTS,
    MIN_LABELLED,
    ReverseCalibration,
    bootstrap_se,
    check_count,
    reverse_calibrate,
    sample_covariance,
    sample_variance,
    stratified_bootstrap,
)
from copernicus.corrections._simex import LAMBDAS, NO_FIT, POLE_LE_1, rational_extrapolate
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.estimates import Estimate, make_estimate
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.errors import EstimatorError

logger = logging.getLogger(__name__)

__all__ = ["moment_subtraction", "simex"]

ESTIMATE_ERRORS = "attenuation of the construct's coefficient as predictor"


@dataclass(frozen=True, eq=False)
class _MomentFit:
    """The moment-subtraction slope and the moments behind it."""

    slope: float
    naive_slope: float
    sigma2: float
    proxy_variance: float
    calibration: ReverseCalibration


def _moment_fit(
    score: np.ndarray, outcome: np.ndarray, labelled: np.ndarray, gold: np.ndarray
) -> _MomentFit:
    """``cov(W, y) / (var(W) - sigma2)``, refused when the denominator is not positive."""
    calibration = reverse_calibrate(score, labelled, gold)
    proxy = calibration.proxy
    sigma2 = sample_variance(proxy[labelled] - gold[labelled])
    proxy_variance = sample_variance(proxy)
    if proxy_variance - sigma2 <= 0.0:
        raise EstimatorError(
            f"corrected variance not positive: the proxy's variance {proxy_variance:.4g} does "
            f"not exceed its error variance {sigma2:.4g}, so the construct's variance is not "
            "identified"
        )
    covariance = sample_covariance(proxy, outcome)
    return _MomentFit(
        slope=covariance / (proxy_variance - sigma2),
        naive_slope=covariance / proxy_variance,
        sigma2=sigma2,
        proxy_variance=proxy_variance,
        calibration=calibration,
    )


def moment_subtraction(
    data: CorrectionData,
    outcome: str,
    *,
    score_column: str | None = None,
    n_boot: int = 200,
    seed: int = 42,
) -> Estimate:
    """The slope of an outcome on the construct by moment subtraction on the classical proxy.

    Args:
        data: The correction data with gold on a random sample of labelled rows.
        outcome: A numeric column of the data, finite on every text.
        score_column: The score column to put on the construct's scale; ``None`` for the first.
        n_boot: Number of bootstrap resamples, at least ``MIN_FINITE_BOOTS`` (50).
        seed: Seed of the bootstrap, an integer of at least 0.

    Returns:
        The estimate, with method ``moment_subtraction`` and details ``score_column``,
        ``outcome``, ``naive_slope`` (``cov(W, y) / var(W)``), ``sigma2``, ``proxy_variance``,
        ``corrected_variance``, ``calibration_intercept``, ``calibration_slope``, ``n_boot``,
        ``n_boot_undefined`` (resamples on which the estimate was undefined) and
        ``n_boot_defined``.

    Raises:
        SettingsError: If ``n_boot`` or ``seed`` is outside its range.
        ConfigurationError: If the outcome or the score column does not match the data.
        EstimatorError: If fewer than three texts are labelled, the reverse calibration has no
            line, the corrected variance is not positive, or fewer than ``MIN_FINITE_BOOTS``
            resamples have a defined estimate.
    """
    n_boot = check_count("n_boot", n_boot, MIN_FINITE_BOOTS)
    seed = check_count("seed", seed, 0)
    response = outcome_values(data, outcome)
    score, column = score_values(data, score_column)
    require_labelled(data.n_labelled, MIN_LABELLED, "moment subtraction")
    labelled, gold = data.labelled_mask, data.gold
    fit = _moment_fit(score, response, labelled, gold)

    def resample_slope(rows: np.ndarray, sample: np.ndarray) -> float:
        return _moment_fit(score[rows], response[rows], sample, gold[rows]).slope

    draws = stratified_bootstrap(
        resample_slope, labelled, n_boot=n_boot, rng=np.random.default_rng(seed)
    )
    se = bootstrap_se(draws, n_boot)
    logger.info(
        "moment subtraction slope %.4g (SE %.3g); %d of %d resamples undefined",
        fit.slope,
        se,
        draws.n_undefined,
        n_boot,
    )
    return make_estimate(
        fit.slope,
        se,
        n_labelled=int(labelled.sum()),
        n_total=data.n,
        method="moment_subtraction",
        details={
            "score_column": column,
            "outcome": outcome,
            "naive_slope": fit.naive_slope,
            "sigma2": fit.sigma2,
            "proxy_variance": fit.proxy_variance,
            "corrected_variance": fit.proxy_variance - fit.sigma2,
            "calibration_intercept": fit.calibration.intercept,
            "calibration_slope": fit.calibration.slope,
            "n_boot": n_boot,
            "n_boot_undefined": draws.n_undefined,
            "n_boot_defined": n_boot - draws.n_undefined,
        },
    )


@dataclass(frozen=True, eq=False)
class _SimexFit:
    """The SIMEX value with its curve, the extrapolant's pole and status."""

    value: float
    pole: float
    status: str
    curve: np.ndarray
    sigma2: float
    naive_slope: float


def _noisy_slope(
    proxy: np.ndarray, centred_outcome: np.ndarray, noise_scale: float, rng: np.random.Generator
) -> float:
    """The slope of the outcome on the proxy after one draw of added normal noise."""
    noisy = proxy + rng.normal(0.0, noise_scale, size=proxy.size)
    centred = noisy - noisy.mean()
    return float((centred @ centred_outcome) / (centred @ centred))


def _simex_fit(
    score: np.ndarray,
    outcome: np.ndarray,
    labelled: np.ndarray,
    gold: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> _SimexFit:
    """The averaged slope curve over the noise multiples and its extrapolation to -1."""
    proxy = reverse_calibrate(score, labelled, gold).proxy
    sigma2 = sample_variance(proxy[labelled] - gold[labelled])
    centred_outcome = outcome - outcome.mean()
    curve = np.empty(len(LAMBDAS))
    for index, multiple in enumerate(LAMBDAS):
        noise_scale = float(np.sqrt(sigma2 * multiple))
        slopes = [_noisy_slope(proxy, centred_outcome, noise_scale, rng) for _ in range(reps)]
        curve[index] = float(np.mean(slopes))
    extrapolation = rational_extrapolate(np.asarray(LAMBDAS), curve)
    if extrapolation.status == NO_FIT:
        raise EstimatorError(
            "the SIMEX extrapolation is undefined: status no_fit (the slope curve has a "
            "non-finite value)"
        )
    if extrapolation.status == POLE_LE_1:
        raise EstimatorError(
            f"the SIMEX extrapolation is undefined: status pole_le_1 (pole "
            f"{extrapolation.pole:.3g}); a pole at or below 1 means the corrected variance of "
            "the construct is not positive"
        )
    return _SimexFit(
        value=extrapolation.value,
        pole=extrapolation.pole,
        status=extrapolation.status,
        curve=curve,
        sigma2=sigma2,
        naive_slope=sample_covariance(proxy, outcome) / sample_variance(proxy),
    )


def simex(
    data: CorrectionData,
    outcome: str,
    *,
    score_column: str | None = None,
    seed: int = 42,
    reps: int = 25,
    n_boot: int = 100,
    boot_reps: int = 10,
) -> Estimate:
    """The slope of an outcome on the construct by SIMEX with the rational extrapolant.

    Args:
        data: The correction data with gold on a random sample of labelled rows.
        outcome: A numeric column of the data, finite on every text.
        score_column: The score column to put on the construct's scale; ``None`` for the first.
        seed: Seed of the noise and of the bootstrap, an integer of at least 0.
        reps: Noise replicates per multiple for the estimate, at least 1.
        n_boot: Number of bootstrap resamples, at least ``MIN_FINITE_BOOTS`` (50).
        boot_reps: Noise replicates per multiple inside each resample, at least 1.

    Returns:
        The estimate, with method ``simex`` and details ``status`` (``ok``, ``constant`` or
        ``linear_limit``), ``pole``, ``lambdas``, ``curve`` (the averaged slope at each
        multiple, a tuple), ``naive_slope`` (``cov(W, y) / var(W)``), ``sigma2``,
        ``score_column``, ``outcome``, ``reps``, ``n_boot``, ``boot_reps``,
        ``n_boot_undefined`` and ``n_boot_defined``.

    Raises:
        SettingsError: If ``reps``, ``n_boot``, ``boot_reps`` or ``seed`` is outside its range.
        ConfigurationError: If the outcome or the score column does not match the data.
        EstimatorError: If fewer than three texts are labelled, the reverse calibration has no
            line, the extrapolation's status is ``pole_le_1`` or ``no_fit`` (the message names
            it), or fewer than ``MIN_FINITE_BOOTS`` resamples have a defined estimate.
    """
    n_boot = check_count("n_boot", n_boot, MIN_FINITE_BOOTS)
    reps = check_count("reps", reps, 1)
    boot_reps = check_count("boot_reps", boot_reps, 1)
    seed = check_count("seed", seed, 0)
    response = outcome_values(data, outcome)
    score, column = score_values(data, score_column)
    require_labelled(data.n_labelled, MIN_LABELLED, "SIMEX")
    labelled, gold = data.labelled_mask, data.gold
    rng = np.random.default_rng(seed)
    fit = _simex_fit(score, response, labelled, gold, reps, rng)

    def resample_value(rows: np.ndarray, sample: np.ndarray) -> float:
        return _simex_fit(score[rows], response[rows], sample, gold[rows], boot_reps, rng).value

    draws = stratified_bootstrap(resample_value, labelled, n_boot=n_boot, rng=rng)
    se = bootstrap_se(draws, n_boot)
    logger.info("SIMEX slope %.4g (SE %.3g), status %s", fit.value, se, fit.status)
    return make_estimate(
        fit.value,
        se,
        n_labelled=int(labelled.sum()),
        n_total=data.n,
        method="simex",
        details={
            "status": fit.status,
            "pole": fit.pole,
            "lambdas": LAMBDAS,
            "curve": tuple(fit.curve.tolist()),
            "naive_slope": fit.naive_slope,
            "sigma2": fit.sigma2,
            "score_column": column,
            "outcome": outcome,
            "reps": reps,
            "n_boot": n_boot,
            "boot_reps": boot_reps,
            "n_boot_undefined": draws.n_undefined,
            "n_boot_defined": n_boot - draws.n_undefined,
        },
    )


MOMENT_SUBTRACTION_SPEC = register(
    FamilySpec(
        family="moment_subtraction",
        variants=("moment_subtraction",),
        target="D",
        level="estimate",
        errors_addressed=(),
        estimate_errors=ESTIMATE_ERRORS,
        factory=None,
    )
)
SIMEX_SPEC = register(
    FamilySpec(
        family="simex",
        variants=("simex",),
        target="D",
        level="estimate",
        errors_addressed=(),
        estimate_errors=ESTIMATE_ERRORS,
        factory=None,
    )
)
