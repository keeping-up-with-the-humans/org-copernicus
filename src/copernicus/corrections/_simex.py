"""The rational extrapolant of SIMEX: ``a + b / (c + lambda)`` evaluated at ``lambda = -1``.

For the slope of an outcome on a classical proxy ``W`` with error variance ``sigma2``, adding
noise of variance ``lambda * sigma2`` gives the slope curve
``beta * var(V) / (var(W) + lambda * sigma2)``, which is exactly of the rational form with the
pole ``c = var(W) / sigma2``. A pole at or below 1 means the moments imply a corrected variance
of the construct that is not positive, so the extrapolation is undefined there, exactly where
moment subtraction is undefined.

The fit is least squares over the grid of ``lambda`` by variable projection: for a fixed pole
the model is linear in ``(a, b)``; the residual is minimised over the pole on a log grid of 181
points between ``1e-3`` and ``pole_max`` and refined by bounded scalar minimisation between the
neighbours of the best grid point. The curve is divided by its largest absolute value before the
fit, so the fit and its tolerances do not depend on the outcome's units. ``scipy.optimize`` is
imported inside the function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTANT",
    "LAMBDAS",
    "LINEAR_LIMIT",
    "NO_FIT",
    "OK",
    "POLE_LE_1",
    "POLE_MAX",
    "Extrapolation",
    "rational_extrapolate",
]

LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.0)
POLE_MAX = 1e6
POLE_GRID_SIZE = 181
POLE_GRID_LOW_EXPONENT = -3.0
CONSTANT_TOLERANCE = 1e-9
REFINE_TOLERANCE = 1e-12
OK = "ok"
CONSTANT = "constant"
POLE_LE_1 = "pole_le_1"
LINEAR_LIMIT = "linear_limit"
NO_FIT = "no_fit"


@dataclass(frozen=True)
class Extrapolation:
    """The value of the extrapolant at ``lambda = -1``.

    Attributes:
        value: The extrapolated value; NaN when the status is ``pole_le_1`` or ``no_fit``.
        pole: The fitted pole ``c``; NaN for ``constant`` and ``no_fit``.
        status: ``ok`` (an interior pole above 1); ``constant`` (the curve does not vary, its
            mean is returned); ``linear_limit`` (the best pole is the grid's cap, where the
            rational curve is a straight line in ``lambda`` and the value is the linear
            extrapolation); ``pole_le_1`` (undefined); ``no_fit`` (a non-finite curve value,
            undefined).
    """

    value: float
    pole: float
    status: str


def rational_extrapolate(
    lambdas: np.ndarray, values: np.ndarray, *, pole_max: float = POLE_MAX
) -> Extrapolation:
    """Fit ``a + b / (c + lambda)`` to the curve and evaluate it at ``lambda = -1``.

    Args:
        lambdas: The grid of added-noise multiples, starting at 0.
        values: The curve: the mean slope at each multiple.
        pole_max: The upper end of the pole grid.

    Returns:
        The extrapolation with its status.
    """
    from scipy.optimize import minimize_scalar

    grid_lambdas = np.asarray(lambdas, dtype=float)
    curve = np.asarray(values, dtype=float)
    if not np.isfinite(curve).all():
        return Extrapolation(value=float("nan"), pole=float("nan"), status=NO_FIT)
    scale = float(np.max(np.abs(curve)))
    if scale == 0.0 or float(np.ptp(curve)) <= CONSTANT_TOLERANCE * scale:
        return Extrapolation(value=float(curve.mean()), pole=float("nan"), status=CONSTANT)
    normalised = curve / scale

    def fit(pole: float) -> tuple[np.ndarray, float]:
        design = np.column_stack([np.ones_like(grid_lambdas), 1.0 / (pole + grid_lambdas)])
        coefficients, *_ = np.linalg.lstsq(design, normalised, rcond=None)
        return coefficients, float(np.sum((design @ coefficients - normalised) ** 2))

    poles = np.logspace(POLE_GRID_LOW_EXPONENT, np.log10(pole_max), POLE_GRID_SIZE)
    residuals = np.array([fit(float(pole))[1] for pole in poles])
    best = int(np.argmin(residuals))
    low, high = poles[max(best - 1, 0)], poles[min(best + 1, poles.size - 1)]
    refined = minimize_scalar(
        lambda pole: fit(float(pole))[1],
        bounds=(float(low), float(high)),
        method="bounded",
        options={"xatol": REFINE_TOLERANCE},
    )
    pole = float(refined.x) if float(refined.fun) <= residuals[best] else float(poles[best])
    if pole <= 1.0:
        return Extrapolation(value=float("nan"), pole=pole, status=POLE_LE_1)
    (intercept, slope), _ = fit(pole)
    status = LINEAR_LIMIT if best == poles.size - 1 else OK
    logger.debug("rational extrapolation: pole %.4g, status %s", pole, status)
    return Extrapolation(
        value=float(scale * (intercept + slope / (pole - 1.0))), pole=pole, status=status
    )
