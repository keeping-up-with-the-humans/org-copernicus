"""One draw of the repeated study: its data, the columns its estimators share, their estimates.

The draw of size ``n`` with index ``d`` takes two integers from the generator
``rng_for(seed, "repeated_study", n, d)``: the first is the seed of the group draw
(``draw_from_data``), the second the seed of the draw's cross-fit folds. ``draw_estimate`` gives
the point and the standard error of one estimator and estimand on a draw, as the module
docstring of ``study`` describes them, and raises ``EstimatorError`` where the estimator is
undefined on the draw.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from copernicus.baseline.folds import rng_for
from copernicus.corrections._clustering import Clustering, contributions, mean_covariance
from copernicus.corrections._open_cases import warn_zero_se
from copernicus.corrections._study_checks import crossfit_clustering
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.estimates import (
    group_difference_ppi,
    mean_ppi,
    naive_se,
    plugin_value,
)
from copernicus.corrections.families._fitted import SEED_BOUND
from copernicus.corrections.families.calibration import fit_ols_line
from copernicus.corrections.families.pseudo import PseudoValues, pseudo_values
from copernicus.corrections.sampling import attach_draw, draw_from_data

logger = logging.getLogger(__name__)

__all__ = ["StudyDraw", "crossfit_se", "draw_estimate", "plugged_column", "study_draw"]

CROSSFIT = "ppi_crossfit"
PLUGGED_ZERO_SE = (
    "so the interval of this draw has zero width; the repeated study counts such draws of a "
    "row in n_se_exactly_zero"
)


@dataclass(frozen=True, eq=False)
class StudyDraw:
    """One draw's data and the columns its estimators share, each computed once.

    Attributes:
        data: The draw's correction data, with the known values of its texts as gold.
        split: The long side of the long-short difference, in data order.
        crossfit_seed: The seed of the draw's cross-fit folds.
    """

    data: CorrectionData
    split: np.ndarray
    crossfit_seed: int

    @cached_property
    def raw(self) -> np.ndarray:
        """The first score column."""
        return self.data.frame[self.data.score_columns[0]].to_numpy(dtype=float)

    @cached_property
    def calibrated(self) -> np.ndarray:
        """The first score column through the line fitted on the labelled texts."""
        labelled = self.data.labelled_mask
        line = fit_ols_line(
            self.raw[labelled], self.data.gold[labelled], self.data.score_columns[0]
        )
        return line.predict(self.raw)

    @cached_property
    def crossfit(self) -> PseudoValues:
        """The cross-fitted pseudo-values."""
        return pseudo_values(self.data, rectifier="crossfit", seed=self.crossfit_seed)

    @cached_property
    def ppi_crossfit(self) -> np.ndarray:
        """The column of the cross-fitted pseudo-values."""
        return self.crossfit.column.to_numpy(dtype=float)

    def rank_deficient_folds(self, estimator: str) -> int:
        """The inner folds of numerical rank below 3 behind an estimator defined on the draw.

        Args:
            estimator: An estimator whose estimate on this draw is defined.

        Returns:
            ``n_rank_deficient_folds`` of the cross-fitted pseudo-values for ``ppi_crossfit``;
            0 for the other estimators, which fit no folds.
        """
        return self.crossfit.n_rank_deficient_folds if estimator == CROSSFIT else 0


def plugged_column(estimator: str, draw: StudyDraw) -> np.ndarray:
    """The column that a naive plug-in estimator (``raw`` or ``calibrated``) uses."""
    return draw.raw if estimator == "raw" else draw.calibrated


def crossfit_se(column: np.ndarray, split: np.ndarray | None, clustering: Clustering) -> float:
    """The standard error of the plugged pseudo-values, clustered as ``ppi_identity`` clusters."""
    if clustering.groups is None:
        return naive_se(column, split)
    psi = contributions(column, split)[:, None]
    return float(np.sqrt(mean_covariance(psi, clustering.groups)[0, 0]))


def draw_estimate(
    estimator: str, estimand: str, draw: StudyDraw, cluster: bool | None
) -> tuple[float, float]:
    """The point and the standard error of one estimator and estimand on one draw.

    Args:
        estimator: ``raw``, ``calibrated``, ``ppi_identity`` or ``ppi_crossfit``.
        estimand: ``mean`` or ``long_short_difference``.
        draw: The draw.
        cluster: The study's ``cluster`` option.

    Returns:
        The point and its standard error. A standard error of exactly 0 is returned as it is,
        with one warning (``ppi_identity`` logs it through its estimate), which the study
        counts instead.

    Raises:
        EstimatorError: Where the estimator is undefined on the draw.
    """
    split = None if estimand == "mean" else draw.split
    if estimator == "ppi_identity":
        result = (
            mean_ppi(draw.data, cluster=cluster)
            if split is None
            else group_difference_ppi(draw.data, split, cluster=cluster)
        )
        return result.point, result.se
    if estimator == CROSSFIT:
        clustering = crossfit_clustering(draw.data, split, cluster)
        column = draw.ppi_crossfit
        point, se = plugin_value(column, split), crossfit_se(column, split, clustering)
    else:
        column = plugged_column(estimator, draw)
        point, se = plugin_value(column, split), naive_se(column, split)
    if se == 0.0:
        warn_zero_se(logger, estimator, PLUGGED_ZERO_SE)
    return point, se


def study_draw(
    data: CorrectionData,
    truth: pd.Series,
    size: int,
    seed: int,
    split: np.ndarray,
    index: int,
) -> StudyDraw:
    """The human sample of one size and index, with the known values of its texts."""
    draw_seed, crossfit_seed = (
        int(value)
        for value in rng_for(seed, "repeated_study", size, index).integers(SEED_BOUND, size=2)
    )
    sample = draw_from_data(data, size, seed=draw_seed)
    return StudyDraw(
        data=attach_draw(data, truth, sample), split=split, crossfit_seed=crossfit_seed
    )
