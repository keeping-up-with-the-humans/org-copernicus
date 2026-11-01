"""Blind transport: a correction fitted on other corpora, applied to a corpus without gold.

The corpora share some score columns (the same coder configurations run on each). Within each
source corpus the human-coded values and every shared column are z-scored on the labelled
rows with the population standard deviation, and the z-scores are pooled across sources. The
shared column with the highest pooled signed Spearman correlation with the pooled z-scored
human-coded values is chosen, and the least-squares line of the pooled z-scored human-coded
values on the pooled z-scored column is applied to the target's z-scored column. The target's
human-coded values are never read: the column and the line come from the sources alone.

The output is in standard-deviation units of the source corpora's human-coded values (the
scales of the corpora differ, so it is not on any one gold scale). Treat it as a ranking and
as a standardised level: it supports rank validity and standardised comparisons, not the raw
level of a particular gold scale. It addresses level displacement and the calibration
shortfall with another corpus's human-coded values. It is a cross-corpus function, not a
per-text correction fitted within one corpus, so it has no factory.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.corrections.families.calibration import fit_ols_line
from copernicus.errors import ConfigurationError, EstimatorError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["TransportResult", "blind_transport"]

FAMILY = "blind_transport"
VARIANT = "transport"
ERRORS_ADDRESSED = ("level_displacement", "calibration_shortfall")


@dataclass(frozen=True, eq=False)
class TransportResult:
    """The transported column and how it was chosen.

    Attributes:
        column: The target's transported values in standard-deviation units of the sources'
            human-coded values, indexed by the target's text ids.
        score_column: The shared column chosen on the sources.
        slope: Slope of the pooled line of z-scored human-coded values on the z-scored column.
        intercept: Intercept of that line.
        n_sources: Number of source corpora.
        selection_rho: The pooled signed Spearman correlation of the chosen column.
        n_shared_columns: Number of score columns the target and every source share.
    """

    column: pd.Series
    score_column: str
    slope: float
    intercept: float
    n_sources: int
    selection_rho: float
    n_shared_columns: int


def _z(values: np.ndarray) -> np.ndarray:
    """Values centred and scaled by the population standard deviation (not constant)."""
    return (values - values.mean()) / values.std()


def _source_gold(sources: Sequence[CorrectionData]) -> list[np.ndarray]:
    """The z-scored labelled gold values of each source, refusing unusable gold."""
    z_gold = []
    for position, source in enumerate(sources):
        gold = source.gold[source.labelled_mask]
        if gold.size == 0:
            raise ConfigurationError(f"source {position} has no labelled gold values")
        if is_constant(gold):
            raise ConfigurationError(
                f"source {position} has constant labelled gold values; they cannot be z-scored"
            )
        z_gold.append(_z(gold))
    return z_gold


def _pooled_column(sources: Sequence[CorrectionData], column: str) -> np.ndarray | None:
    """The z-scored column of every source on its labelled rows, pooled.

    Returns ``None`` when the column is constant on the labelled rows of any source.
    """
    parts = []
    for source in sources:
        values = source.frame[column].to_numpy(dtype=float)[source.labelled_mask]
        if is_constant(values):
            return None
        parts.append(_z(values))
    return np.concatenate(parts)


def blind_transport(sources: Sequence[CorrectionData], target: CorrectionData) -> TransportResult:
    """Transport the correction learned on the source corpora to the target corpus.

    A shared column that is constant on the labelled rows of any source is not a candidate.
    A tie in the pooled correlation goes to the column that comes first in sorted order.
    The column is chosen on the sources alone, so a chosen column that is constant in the
    target is refused even when another shared column would have worked: the selection is
    gold-blind and target-blind by design.

    Args:
        sources: The corpora with human-coded values, at least one.
        target: The corpus to correct; its gold column is never read, so it may be all NaN.

    Returns:
        The transported column, the chosen score column and the pooled line.

    Raises:
        ConfigurationError: If there is no source, or a source has no labelled gold value or
            only a constant one.
        EstimatorError: If the target and the sources share no score column, no shared column
            is usable (constant in a source, or an undefined correlation), or the chosen
            column is constant in the target.
    """
    if len(sources) == 0:
        raise ConfigurationError("blind transport needs at least one source corpus")
    shared = set(target.score_columns)
    for source in sources:
        shared &= set(source.score_columns)
    if not shared:
        raise EstimatorError(
            "the target and the sources share no score column; target columns: "
            f"{list_first_ten(list(target.score_columns))}"
        )
    pooled_gold = np.concatenate(_source_gold(sources))
    best_column: str | None = None
    best_rho = -np.inf
    best_pooled = np.array([])
    for column in sorted(shared):
        pooled = _pooled_column(sources, column)
        if pooled is None:
            continue
        rho = float(spearmanr(pooled, pooled_gold).statistic)
        if np.isfinite(rho) and rho > best_rho:
            best_column, best_rho, best_pooled = column, rho, pooled
    if best_column is None:
        raise EstimatorError(
            "no usable shared score column: each of "
            f"{list_first_ten(sorted(shared))} is constant on the labelled rows of a source "
            "or has an undefined correlation"
        )
    line = fit_ols_line(best_pooled, pooled_gold, best_column)
    here = target.frame[best_column].to_numpy(dtype=float)
    if is_constant(here):
        raise EstimatorError(f"the chosen score column {best_column!r} is constant in the target")
    logger.info(
        "transported %r from %d sources (pooled rho %.3f)", best_column, len(sources), best_rho
    )
    return TransportResult(
        column=pd.Series(line.predict(_z(here)), index=pd.Index(target.ids)),
        score_column=best_column,
        slope=line.slope,
        intercept=line.intercept,
        n_sources=len(sources),
        selection_rho=best_rho,
        n_shared_columns=len(shared),
    )


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=(VARIANT,),
        target="A",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors="none",
        factory=None,
    )
)
