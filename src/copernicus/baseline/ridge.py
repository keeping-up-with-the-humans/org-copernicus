"""Ridge with an unpenalised intercept, tuned by exact grouped leave-one-group-out.

The leave-one-group-out (LOGO) residuals of a ridge fit with an unpenalised intercept
follow from the block hat matrix of the full fit,

    H_gg = 1/n * 1 1' + (U_g * d) U_g',   d_k = s_k^2 / (s_k^2 + alpha),

where ``U s V'`` is the thin SVD of the column-centred design and the ``1/n`` term is the
intercept's share. For a linear smoother whose penalty does not depend on ``y``, the
held-out residual of group ``g`` is exactly ``(I - H_gg)^{-1} (y_g - yhat_g)``; singletons
reduce to ``r_i / (1 - h_ii)`` with ``h_ii`` including ``1/n``.

Appending one penalised column ``s`` (a coder score) to the same design changes the hat
matrix by an exact rank-one term (block inverse of the augmented normal equations):

    H'(alpha) = H(alpha) + u u' / k,   u = (I - H) s_c,   k = alpha + s_c' u,

with coefficient ``gamma = u' y_c / k`` on ``s`` and ``beta = beta_E - gamma * beta_s`` on
the embeddings, where ``beta_s`` is the ridge regression of ``s_c`` on the embeddings. Both
variants therefore share one eigendecomposition per fold.

Feature scaling is fitted on the training draw only (``fit_scaler``); the stacked variant
rescales each z-scored encoder block so that every encoder contributes the same total
variance.

The penalty grid is ``logspace(-2, 8, 41)`` (0.25 log10 spacing). The selected penalty
and whether it sits on a grid edge are recorded on every fit.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from copernicus.errors import ConfigurationError, EstimatorError

logger = logging.getLogger(__name__)

__all__ = [
    "ALPHAS",
    "RidgeDesign",
    "RidgeFit",
    "Scaler",
    "apply_scaler",
    "fit_scaler",
    "logo_ridge_fit",
    "predict",
]

ALPHAS: np.ndarray = np.logspace(-2.0, 8.0, 41)
"""The default ridge penalties: 41 values from 1e-2 to 1e8, evenly spaced on the log scale."""
EIGENVALUE_RELATIVE_TOLERANCE = 1e-10
DEGENERATE_DESIGN_MESSAGE = "degenerate training design: no feature variance after centring"


@dataclass(frozen=True)
class Scaler:
    """Per-block column mean and SD and the block weight, fitted on training rows.

    Attributes:
        means: Column means of each block.
        sds: Column standard deviations of each block; constant columns hold 1.
        weights: Multiplier applied to each scaled block.
    """

    means: tuple[np.ndarray, ...]
    sds: tuple[np.ndarray, ...]
    weights: tuple[float, ...]


def fit_scaler(blocks: Sequence[np.ndarray]) -> Scaler:
    """Fit column z-scoring per block; constant columns get SD 1.

    With several blocks (stacked encoders) block ``b`` is multiplied by
    ``sqrt(mean_dim / dim_b)`` so that each block's total variance equals the mean block
    dimension and the concatenation keeps the scale of a plain z-scored design; a single
    block keeps weight 1.

    Args:
        blocks: Training feature blocks, each of shape ``(n_rows, dim_b)``.

    Returns:
        The fitted scaler.

    Raises:
        ConfigurationError: If no block is given, a block is not two-dimensional, the
            blocks differ in row count, or a block holds non-finite values.
    """
    if not blocks:
        raise ConfigurationError("fit_scaler: no feature blocks")
    row_counts = [np.asarray(block).shape[0] for block in blocks]
    if len(set(row_counts)) > 1:
        raise ConfigurationError(
            f"fit_scaler: blocks have different row counts ({', '.join(map(str, row_counts))})"
        )
    means: list[np.ndarray] = []
    sds: list[np.ndarray] = []
    for index, block in enumerate(blocks):
        values = np.asarray(block, dtype=np.float64)
        if values.ndim != 2:
            raise ConfigurationError(
                f"fit_scaler: blocks must be 2-D; got {values.ndim} dimension(s)"
            )
        if not np.isfinite(values).all():
            raise ConfigurationError(f"fit_scaler: block {index} holds non-finite values")
        spread = values.std(axis=0)
        means.append(values.mean(axis=0))
        sds.append(np.where(spread == 0.0, 1.0, spread))
    weights: tuple[float, ...] = (1.0,)
    if len(blocks) > 1:
        dims = np.array([np.asarray(block).shape[1] for block in blocks], dtype=float)
        weights = tuple(float(weight) for weight in np.sqrt(dims.mean() / dims))
    return Scaler(means=tuple(means), sds=tuple(sds), weights=weights)


def apply_scaler(blocks: Sequence[np.ndarray], scaler: Scaler) -> np.ndarray:
    """Scale each block with the fitted scaler and concatenate into one design matrix.

    Args:
        blocks: Feature blocks in the order the scaler was fitted on.
        scaler: A scaler from ``fit_scaler``.

    Returns:
        A float64 matrix of shape ``(n_rows, sum of block dimensions)``.

    Raises:
        ConfigurationError: If the number of blocks differs from the scaler's.
    """
    if len(blocks) != len(scaler.means):
        raise ConfigurationError(
            f"apply_scaler: block count {len(blocks)} differs from the fitted scaler's "
            f"{len(scaler.means)}"
        )
    parts = [
        ((np.asarray(block, dtype=np.float64) - mean) / sd) * weight
        for block, mean, sd, weight in zip(
            blocks, scaler.means, scaler.sds, scaler.weights, strict=True
        )
    ]
    return np.hstack(parts)


@dataclass(frozen=True)
class RidgeFit:
    """A ridge fit at the LOGO-selected penalty with the tuning trace.

    ``coef`` covers the design columns in order, with the appended column (if any) last;
    ``predict`` expects the matching design.

    Attributes:
        alpha: Selected penalty.
        coef: Coefficients on the design columns.
        intercept: Unpenalised intercept.
        alphas: The penalty grid searched.
        logo_mse: Leave-one-group-out mean squared error at each grid penalty.
        n_groups: Number of groups in the training design.
    """

    alpha: float
    coef: np.ndarray
    intercept: float
    alphas: np.ndarray
    logo_mse: np.ndarray
    n_groups: int

    @property
    def alpha_index(self) -> int:
        """Position of the selected penalty in the grid."""
        return int(np.argmin(self.logo_mse))

    @property
    def at_grid_edge(self) -> bool:
        """Whether the selected penalty is the smallest or the largest on the grid."""
        return self.alpha_index in (0, self.alphas.shape[0] - 1)


def predict(fit: RidgeFit, features: np.ndarray) -> np.ndarray:
    """Predict with a fitted ridge on a design that matches ``fit.coef``.

    Args:
        fit: The fitted ridge.
        features: Design matrix with one column per coefficient.

    Returns:
        One prediction per row.
    """
    return fit.intercept + np.asarray(features, dtype=np.float64) @ fit.coef


def _thin_decomposition(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left singular vectors and squared singular values, near-zero directions dropped."""
    n_rows, n_columns = centered.shape
    if n_rows <= n_columns:
        eigenvalues, left = np.linalg.eigh(centered @ centered.T)
    else:
        eigenvalues, right = np.linalg.eigh(centered.T @ centered)
        keep = eigenvalues > EIGENVALUE_RELATIVE_TOLERANCE * max(float(eigenvalues.max()), 1e-300)
        eigenvalues, right = eigenvalues[keep], right[:, keep]
        if eigenvalues.size == 0:
            raise EstimatorError(DEGENERATE_DESIGN_MESSAGE)
        left = (centered @ right) / np.sqrt(eigenvalues)
    keep = eigenvalues > EIGENVALUE_RELATIVE_TOLERANCE * max(float(eigenvalues.max()), 1e-300)
    if not keep.any():
        raise EstimatorError(DEGENERATE_DESIGN_MESSAGE)
    return np.ascontiguousarray(left[:, keep]), eigenvalues[keep]


def _group_positions(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Positions of singleton groups (vector) and of every multi-row group (list)."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    singles = np.flatnonzero(counts[inverse] == 1)
    order = np.argsort(inverse, kind="stable")
    multis: list[np.ndarray] = []
    start = 0
    for index, end in enumerate(np.cumsum(counts)):
        if counts[index] > 1:
            multis.append(order[start:end])
        start = int(end)
    return singles, multis


class RidgeDesign:
    """One training design with its decomposition, reusable across penalties and variants.

    Args:
        features: Design matrix of shape ``(n, p)``; scale it first with ``apply_scaler``.
        y: Target of length ``n``.
        groups: Group label of each row; leave-one-group-out holds out a whole group.

    Raises:
        EstimatorError: If the lengths differ or fewer than two groups exist.
    """

    def __init__(self, features: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
        design = np.asarray(features, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        labels = np.asarray(groups)
        self.n = design.shape[0]
        if target.shape[0] != self.n or labels.shape[0] != self.n:
            raise EstimatorError(
                f"RidgeDesign: features, y, groups length mismatch "
                f"({self.n}, {target.shape[0]}, {labels.shape[0]})"
            )
        self.n_groups = int(np.unique(labels).shape[0])
        if self.n_groups < 2:
            raise EstimatorError(
                f"RidgeDesign: needs at least two groups; got {self.n_groups} in {self.n} rows"
            )
        self.column_means = design.mean(axis=0)
        self.target_mean = float(target.mean())
        self.centered = design - self.column_means
        self.centered_target = target - self.target_mean
        self.left, self.squared_singular = _thin_decomposition(self.centered)
        self.left_squared = self.left**2
        self.left_target = self.left.T @ self.centered_target
        self.singles, self.multis = _group_positions(labels)

    def _logo_mse(
        self,
        residual: np.ndarray,
        leverage: np.ndarray,
        shrunk_left: np.ndarray,
        update: np.ndarray | None,
        denominator: float,
    ) -> float:
        held_out = np.empty(self.n)
        if self.singles.size:
            held_out[self.singles] = residual[self.singles] / (1.0 - leverage[self.singles])
        for positions in self.multis:
            block = 1.0 / self.n + shrunk_left[positions] @ self.left[positions].T
            if update is not None:
                block = block + np.outer(update[positions], update[positions]) / denominator
            held_out[positions] = np.linalg.solve(
                np.eye(positions.shape[0]) - block, residual[positions]
            )
        return float(np.mean(held_out**2))

    def fit(self, alphas: np.ndarray = ALPHAS, extra: np.ndarray | None = None) -> RidgeFit:
        """Tune the penalty by exact LOGO error over ``alphas`` and refit on all rows.

        Args:
            alphas: Penalty grid.
            extra: Optional column appended to the design and penalised like the others;
                its coefficient is the last entry of ``coef``.

        Returns:
            The fit at the penalty with the smallest LOGO error.

        Raises:
            EstimatorError: If ``extra`` has the wrong length or a LOGO error is not finite.
        """
        centered_extra: np.ndarray | None = None
        extra_mean = 0.0
        left_extra: npt.NDArray[np.float64] = np.empty(0)
        if extra is not None:
            extra_values = np.asarray(extra, dtype=np.float64)
            if extra_values.shape[0] != self.n:
                raise EstimatorError(
                    f"RidgeDesign.fit: extra column length mismatch "
                    f"({extra_values.shape[0]} versus {self.n} rows)"
                )
            extra_mean = float(extra_values.mean())
            centered_extra = extra_values - extra_mean
            left_extra = self.left.T @ centered_extra
        mse = np.empty(alphas.shape[0])
        for index, alpha in enumerate(alphas):
            shrink = self.squared_singular / (self.squared_singular + alpha)
            shrunk_left = self.left * shrink
            residual = self.centered_target - shrunk_left @ self.left_target
            leverage = 1.0 / self.n + self.left_squared @ shrink
            update: np.ndarray | None = None
            denominator = 1.0
            if centered_extra is not None:
                update = centered_extra - shrunk_left @ left_extra
                denominator = float(alpha + centered_extra @ update)
                gamma = float(update @ self.centered_target) / denominator
                residual = residual - gamma * update
                leverage = leverage + update**2 / denominator
            mse[index] = self._logo_mse(residual, leverage, shrunk_left, update, denominator)
        if not np.isfinite(mse).all():
            bad = np.flatnonzero(~np.isfinite(mse))
            raise EstimatorError(
                f"RidgeDesign.fit: non-finite leave-one-group-out error at "
                f"{bad.size} of {mse.size} penalties"
            )
        alpha = float(alphas[int(np.argmin(mse))])
        inverse = 1.0 / (self.squared_singular + alpha)
        coef = self.centered.T @ (self.left @ (self.left_target * inverse))
        intercept = self.target_mean - float(self.column_means @ coef)
        if centered_extra is not None:
            shrink = self.squared_singular / (self.squared_singular + alpha)
            update = centered_extra - (self.left * shrink) @ left_extra
            denominator = float(alpha + centered_extra @ update)
            gamma = float(update @ self.centered_target) / denominator
            extra_on_columns = self.centered.T @ (self.left @ (left_extra * inverse))
            coef = coef - gamma * extra_on_columns
            intercept = self.target_mean - float(self.column_means @ coef) - gamma * extra_mean
            coef = np.append(coef, gamma)
        logger.debug(
            "ridge alpha %.4g, grid edge %s", alpha, int(np.argmin(mse)) in (0, mse.size - 1)
        )
        return RidgeFit(
            alpha=alpha,
            coef=coef,
            intercept=intercept,
            alphas=np.asarray(alphas),
            logo_mse=mse,
            n_groups=self.n_groups,
        )


def logo_ridge_fit(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: np.ndarray = ALPHAS,
    extra: np.ndarray | None = None,
) -> RidgeFit:
    """Build the design and fit once.

    Args:
        features: Design matrix.
        y: Target.
        groups: Group label of each row.
        alphas: Penalty grid.
        extra: Optional appended penalised column.

    Returns:
        The LOGO-tuned fit.
    """
    return RidgeDesign(features, y, groups).fit(alphas, extra)
