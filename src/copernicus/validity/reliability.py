"""Cronbach's alpha, the reliability ceiling on rank validity and the share of the ceiling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from copernicus.errors import ConfigurationError, EstimatorError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["cronbach_alpha", "reliability_ceiling", "share_of_ceiling"]

MIN_ITEMS = 2
MIN_COMPLETE_ROWS = 2


def cronbach_alpha(items: pd.DataFrame | np.ndarray) -> float:
    """Cronbach's alpha of a units-by-items matrix.

    ``alpha = k / (k - 1) * (1 - sum(item variances) / variance(sum of items))`` with
    sample variances (ddof=1). Rows with any missing or non-finite item are dropped, and
    the number dropped is logged at WARNING level. Identical columns give an alpha of 1
    (to floating-point precision).

    Args:
        items: Rows are units, columns are items (coders, runs or endpoints). For the
            reliability ceiling of the validity panel they are the items of the gold
            instrument, such as the ratings of each human coder, so their rows need not
            align with the scored texts.

    Returns:
        Cronbach's alpha over the complete rows; it can be negative.

    Raises:
        ConfigurationError: If the input is not a numeric two-dimensional matrix or has
            fewer than two columns.
        EstimatorError: If fewer than two complete rows remain or the item sum has no
            variance.
    """
    try:
        if isinstance(items, pd.DataFrame):
            matrix = items.to_numpy(dtype=float, na_value=np.nan)
        else:
            matrix = np.asarray(items, dtype=float)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"items must be numeric; got {error}") from error
    if matrix.ndim != 2:
        raise ConfigurationError(
            f"items must be a two-dimensional units-by-items matrix; got {matrix.ndim} dimensions"
        )
    n_items = matrix.shape[1]
    if n_items < MIN_ITEMS:
        raise ConfigurationError(f"alpha needs at least two items (columns); got {n_items}")
    complete = np.isfinite(matrix).all(axis=1)
    n_dropped = int((~complete).sum())
    if n_dropped > 0:
        logger.warning(
            "cronbach_alpha: dropped %d of %d rows with a missing or non-finite item",
            n_dropped,
            matrix.shape[0],
        )
    kept = matrix[complete]
    if kept.shape[0] < MIN_COMPLETE_ROWS:
        raise EstimatorError(
            f"alpha needs at least two complete rows; {kept.shape[0]} remain of {matrix.shape[0]}"
        )
    totals = kept.sum(axis=1)
    if is_constant(totals):
        raise EstimatorError("alpha is undefined: the sum of the items has no variance")
    item_variances = kept.var(axis=0, ddof=1)
    total_variance = float(totals.var(ddof=1))
    return float(n_items / (n_items - 1) * (1.0 - float(item_variances.sum()) / total_variance))


def reliability_ceiling(reliability: float) -> float:
    """Highest rank validity a score of the given reliability can reach: ``sqrt(reliability)``.

    Args:
        reliability: Reliability of the gold value, in (0, 1].

    Returns:
        The square root of the reliability.

    Raises:
        ConfigurationError: If the reliability is not finite or lies outside (0, 1].
    """
    value = float(reliability)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ConfigurationError(f"reliability must lie in (0, 1]; got {reliability!r}")
    return float(np.sqrt(value))


def share_of_ceiling(rank_validity: float, ceiling: float) -> float:
    """Rank validity as a share of the reliability ceiling.

    Args:
        rank_validity: Observed rank validity; NaN gives NaN.
        ceiling: Reliability ceiling, positive and finite.

    Returns:
        ``rank_validity / ceiling``.

    Raises:
        ConfigurationError: If the ceiling is not finite or not positive.
    """
    ceiling_value = float(ceiling)
    if not np.isfinite(ceiling_value) or ceiling_value <= 0.0:
        raise ConfigurationError(f"ceiling must be positive and finite; got {ceiling!r}")
    return float(rank_validity) / ceiling_value
