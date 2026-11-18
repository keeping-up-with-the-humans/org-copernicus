"""Bootstrap models and per-text random draws behind multiple imputation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DrawModels", "OlsFit", "fit_ols"]


@dataclass(frozen=True, eq=False)
class OlsFit:
    """A least-squares fit with its rank and the rows minus the columns of its design.

    Attributes:
        coefficients: The coefficients that ``numpy.linalg.lstsq`` with ``rcond=None`` returns:
            it leaves out the directions of the design below its cutoff and returns the solution
            of smallest norm of what remains.
        residual_variance: The residual sum of squares divided by ``residual_df``, or by 1 when
            ``residual_df`` is below 1.
        rank: The numerical rank of the design that ``numpy.linalg.lstsq`` returns, the number
            of its directions above the cutoff; below the columns also for a design of full rank
            in exact arithmetic whose smallest direction lies below the cutoff.
        residual_df: The rows minus the columns of the design; the divisor of the residual
            variance is the larger of it and 1. For a design below full column rank it is
            smaller than the rows minus the rank.
    """

    coefficients: np.ndarray
    residual_variance: float
    rank: int
    residual_df: int


def fit_ols(design: np.ndarray, gold: np.ndarray) -> OlsFit:
    """Least-squares coefficients and the residual variance with the divisor ``max(n - p, 1)``.

    ``n - p`` is the rows minus the columns of the design. With as many rows as columns, or
    fewer, the variance is the residual sum of squares divided by 1.

    Args:
        design: The design, one row per observation.
        gold: The values fitted.

    Returns:
        The coefficients, the residual variance, the rank and the rows minus the columns.
    """
    beta, _, rank, _ = np.linalg.lstsq(design, gold, rcond=None)
    residual = gold - design @ beta
    residual_df = design.shape[0] - design.shape[1]
    return OlsFit(
        coefficients=beta,
        residual_variance=float(residual @ residual) / max(residual_df, 1),
        rank=int(rank),
        residual_df=residual_df,
    )


def _pmm_draw(
    fitted_mean: np.ndarray,
    donor_means: np.ndarray,
    donor_gold: np.ndarray,
    k_donors: int,
    uniform: np.ndarray,
) -> np.ndarray:
    """Observed gold of one of the ``k_donors`` donors nearest in fitted mean.

    With fewer donors than ``k_donors``, every donor can be drawn.
    ``uniform`` holds one number in ``[0, 1)`` per text and picks the donor among the nearest.
    """
    n_donors = donor_means.size
    position = np.searchsorted(donor_means, fitted_mean)
    window = position[:, None] + np.arange(-k_donors, k_donors)[None, :]
    valid = (window >= 0) & (window < n_donors)
    window = np.clip(window, 0, n_donors - 1)
    distance = np.where(valid, np.abs(donor_means[window] - fitted_mean[:, None]), np.inf)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :k_donors]
    available = np.minimum(valid.sum(axis=1), k_donors)
    pick = np.minimum(np.floor(uniform * available).astype(int), available - 1)
    rows = np.arange(fitted_mean.size)
    return donor_gold[window[rows, nearest[rows, pick]]]


@dataclass(frozen=True, eq=False)
class DrawModels:
    """The bootstrap models of all draws and the seed that regenerates their random part.

    Attributes:
        coefficients: Bootstrap coefficients, shape ``[m, 3]`` (intercept, score, covariate).
        residual_variance: Residual variance of each bootstrap fit, shape ``[m]``.
        donors: For ``pmm`` the sorted fitted means of the training rows under each draw and
            the observed gold values in the same order, both of shape ``[m, n_train]``;
            ``None`` for ``model_draws``.
        k_donors: The number of nearest donors (used by ``pmm``).
        child_seed: Seed from which every call regenerates the random part of the draws.
    """

    coefficients: np.ndarray
    residual_variance: np.ndarray
    donors: tuple[np.ndarray, np.ndarray] | None
    k_donors: int
    child_seed: int

    def complete(self, design: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Completed values of the rows of ``design`` under every draw.

        The random numbers of a text come from a generator seeded with the child seed and the
        text's row position in the data, so a text's values do not depend on which other
        texts are completed with it.

        Args:
            design: Columns intercept, score and covariate, one row per text.
            positions: Row position of each text in the correction data.

        Returns:
            Floats of shape ``[m, n]``, identical on every call.
        """
        means = (design @ self.coefficients.T).T
        n_draws = means.shape[0]
        if self.donors is None:
            noise = self._row_numbers(positions, n_draws, normal=True)
            return means + noise * np.sqrt(self.residual_variance)[:, None]
        donor_means, donor_gold = self.donors
        uniform = self._row_numbers(positions, n_draws, normal=False)
        return np.vstack(
            [
                _pmm_draw(
                    means[draw], donor_means[draw], donor_gold[draw], self.k_donors, uniform[draw]
                )
                for draw in range(n_draws)
            ]
        )

    def _row_numbers(self, positions: np.ndarray, n_draws: int, *, normal: bool) -> np.ndarray:
        """Standard normal or uniform numbers, one column per text, keyed to its row position."""
        numbers = np.empty((n_draws, positions.size))
        for column, position in enumerate(positions):
            rng = np.random.default_rng(np.random.SeedSequence([self.child_seed, int(position)]))
            numbers[:, column] = rng.normal(size=n_draws) if normal else rng.random(n_draws)
        return numbers
