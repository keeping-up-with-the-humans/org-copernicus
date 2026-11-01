"""Ridge predictions along a penalty grid, and a penalty chosen on grouped deletions.

The embedding-augmented family chooses its ridge penalty by deleting whole groups of the
training rows and refitting the complete pipeline on the rows that remain: the score column
is selected again and the calibration line refitted before the error of the deleted rows is
scored. The design or the target of the ridge then changes from one deletion to the next,
so the single-decomposition leave-one-group-out shortcut of ``RidgeDesign`` does not apply
and each deletion is refitted explicitly.

* ``ridge_path_predictions``: ridge regression with an unpenalised intercept along a whole
  penalty grid from one thin singular value decomposition ``U s V'`` of the centred fitting
  design: ``beta(alpha) = V diag(s / (s^2 + alpha)) U' (y - mean(y))`` and
  ``prediction = mean(y) + (x - mean(x)) beta``. Singular values at or below ``1e-12``
  times the largest are dropped. The estimator is scikit-learn's
  ``Ridge(fit_intercept=True, solver="svd")``.
* ``ridge_coefficients``: the same estimator at one penalty, as an intercept and weights,
  for a fit that is applied to new rows later.
* ``choose_penalty``: the penalty with the smallest mean squared error pooled over all rows
  the deletions hold out, with the error curve of each deletion, so that the share of
  deletions whose own minimiser lies on an edge of the grid can be reported.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = [
    "Deletion",
    "GroupDeletion",
    "PenaltyChoice",
    "RidgeCoefficients",
    "choose_penalty",
    "ridge_coefficients",
    "ridge_path_predictions",
    "squared_error",
]

SINGULAR_RELATIVE_TOLERANCE = 1e-12
SINGULAR_FLOOR = 1e-300


class Deletion(Protocol):
    """One inner deletion: the pipeline is refitted on ``retained`` and scored on ``held``."""

    @property
    def retained(self) -> np.ndarray:
        """Positions of the rows the pipeline is refitted on."""
        ...

    @property
    def held(self) -> np.ndarray:
        """Positions of the rows whose error the deletion contributes."""
        ...


DeletionT = TypeVar("DeletionT", bound=Deletion)


@dataclass(frozen=True, eq=False)
class GroupDeletion:
    """A deletion of whole groups from a set of training rows.

    Attributes:
        retained: Positions of the rows the pipeline is refitted on.
        held: Positions of the rows whose error the deletion contributes.
    """

    retained: np.ndarray
    held: np.ndarray


@dataclass(frozen=True, eq=False)
class RidgeCoefficients:
    """A ridge fit at one penalty.

    Attributes:
        alpha: The penalty.
        intercept: The unpenalised intercept.
        weights: The coefficient of each design column.
    """

    alpha: float
    intercept: float
    weights: np.ndarray

    def predict(self, design: np.ndarray) -> np.ndarray:
        """Apply the fit.

        Args:
            design: Rows with the columns of the fitting design.

        Returns:
            ``intercept + design @ weights``.
        """
        return self.intercept + np.asarray(design, dtype=np.float64) @ self.weights


@dataclass(frozen=True, eq=False)
class _Decomposition:
    """The centring and the kept singular directions of a fitting design."""

    column_means: np.ndarray
    target_mean: float
    singular: np.ndarray
    right: np.ndarray
    left_target: np.ndarray

    def shrunk_target(self, alpha: float) -> np.ndarray:
        """``diag(s / (s^2 + alpha)) U' (y - mean(y))`` on the kept directions."""
        return self.singular / (self.singular**2 + alpha) * self.left_target


def _fitting_arrays(x_fit: np.ndarray, y_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fitting design and target as floats, refused when they cannot be fitted."""
    design = np.asarray(x_fit, dtype=np.float64)
    target = np.asarray(y_fit, dtype=np.float64)
    if design.ndim != 2:
        raise ConfigurationError(
            f"ridge: the fitting design must be two-dimensional; got shape {design.shape}"
        )
    if target.ndim != 1 or target.shape[0] != design.shape[0]:
        raise ConfigurationError(
            f"ridge: {design.shape[0]} fitting rows and {target.shape[0]} targets "
            f"(target shape {target.shape})"
        )
    if design.shape[0] == 0:
        raise ConfigurationError("ridge: an empty fitting set")
    if not (np.isfinite(design).all() and np.isfinite(target).all()):
        raise ConfigurationError("ridge: non-finite input in the fitting design or target")
    return design, target


def _decompose(design: np.ndarray, target: np.ndarray) -> _Decomposition:
    """Centre the design and the target and keep the non-negligible singular directions."""
    column_means = design.mean(axis=0)
    target_mean = float(target.mean())
    left, singular, right = np.linalg.svd(design - column_means, full_matrices=False)
    largest = float(singular.max()) if singular.size else 0.0
    keep = singular > SINGULAR_RELATIVE_TOLERANCE * max(largest, SINGULAR_FLOOR)
    return _Decomposition(
        column_means=column_means,
        target_mean=target_mean,
        singular=singular[keep],
        right=right[keep],
        left_target=left[:, keep].T @ (target - target_mean),
    )


def _checked_grid(alphas: np.ndarray) -> np.ndarray:
    """The penalty grid as floats, refused unless one-dimensional, non-empty and non-negative."""
    grid = np.asarray(alphas, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all() or (grid < 0.0).any():
        raise ConfigurationError(
            "the penalty grid must be a non-empty one-dimensional array of finite, "
            f"non-negative penalties; got {grid.tolist()!r}"
        )
    return grid


def ridge_path_predictions(
    x_fit: np.ndarray, y_fit: np.ndarray, x_pred: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    """Ridge predictions with an unpenalised intercept for every penalty of a grid.

    One thin singular value decomposition of the centred fitting design serves the grid.

    Args:
        x_fit: The rows the ridge is fitted on.
        y_fit: Their target.
        x_pred: The rows to predict, with the columns of ``x_fit``.
        alphas: The penalty grid.

    Returns:
        Predictions of shape ``(len(alphas), len(x_pred))``.

    Raises:
        ConfigurationError: If a design is not two-dimensional, the rows and the targets or
            the columns of the two designs differ in number, a set is empty, an input is not
            finite, or the grid is empty, not one-dimensional, negative or not finite.
    """
    design, target = _fitting_arrays(x_fit, y_fit)
    rows = np.asarray(x_pred, dtype=np.float64)
    if rows.ndim != 2:
        raise ConfigurationError(
            f"ridge: the prediction design must be two-dimensional; got shape {rows.shape}"
        )
    if design.shape[1] != rows.shape[1]:
        raise ConfigurationError(
            f"ridge: {design.shape[1]} fitting columns and {rows.shape[1]} prediction columns"
        )
    if rows.shape[0] == 0:
        raise ConfigurationError("ridge: an empty prediction set")
    if not np.isfinite(rows).all():
        raise ConfigurationError("ridge: non-finite input in the prediction design")
    grid = _checked_grid(alphas)
    decomposition = _decompose(design, target)
    projected = (rows - decomposition.column_means) @ decomposition.right.T
    predictions = np.empty((grid.shape[0], rows.shape[0]), dtype=np.float64)
    for index, alpha in enumerate(grid):
        predictions[index] = decomposition.target_mean + projected @ (
            decomposition.shrunk_target(float(alpha))
        )
    return predictions


def ridge_coefficients(x_fit: np.ndarray, y_fit: np.ndarray, alpha: float) -> RidgeCoefficients:
    """The ridge fit with an unpenalised intercept at one penalty.

    The weights come from the same decomposition as ``ridge_path_predictions``, so the fit
    predicts what the path predicts at this penalty.

    Args:
        x_fit: The rows the ridge is fitted on.
        y_fit: Their target.
        alpha: The penalty.

    Returns:
        The intercept and the weights.

    Raises:
        ConfigurationError: If the design is not two-dimensional, rows and targets differ in
            number, the design is empty, an input is not finite, or the penalty is negative
            or not finite.
    """
    design, target = _fitting_arrays(x_fit, y_fit)
    if not (np.isfinite(alpha) and alpha >= 0.0):
        raise ConfigurationError(f"ridge: the penalty must be finite and non-negative; got {alpha}")
    decomposition = _decompose(design, target)
    weights = decomposition.right.T @ decomposition.shrunk_target(float(alpha))
    intercept = decomposition.target_mean - float(decomposition.column_means @ weights)
    return RidgeCoefficients(alpha=float(alpha), intercept=intercept, weights=weights)


@dataclass(frozen=True, eq=False)
class PenaltyChoice:
    """The penalty the grouped deletions selected, with its error trace.

    Attributes:
        alpha: The selected penalty, the first minimiser of ``mse`` on the grid.
        policy: The recorded name of the selection policy.
        n_deletions: The number of deletions.
        n_groups: The number of groups in the training rows.
        alphas: The penalty grid.
        mse: The mean squared error per penalty, pooled over every row the deletions hold out.
        per_deletion_mse: The mean squared error per penalty over the rows of each deletion,
            one row per deletion in deletion order.
    """

    alpha: float
    policy: str
    n_deletions: int
    n_groups: int
    alphas: np.ndarray
    mse: np.ndarray
    per_deletion_mse: np.ndarray

    @property
    def alpha_index(self) -> int:
        """Position of the selected penalty in the grid."""
        return int(np.argmin(self.mse))

    @property
    def at_lower_edge(self) -> bool:
        """Whether the selected penalty is the first of the grid."""
        return self.alpha_index == 0

    @property
    def at_upper_edge(self) -> bool:
        """Whether the selected penalty is the last of the grid."""
        return self.alpha_index == self.alphas.shape[0] - 1

    @property
    def at_grid_edge(self) -> bool:
        """Whether the selected penalty is the first or the last of the grid."""
        return self.at_lower_edge or self.at_upper_edge

    def deletion_edge_share(self, upper: bool) -> float:
        """Share of the deletions whose own minimiser is the last (or first) grid penalty.

        Args:
            upper: ``True`` for the last penalty of the grid, ``False`` for the first.

        Returns:
            A share between 0 and 1. It describes how stable the choice is; it does not
            change the selection.
        """
        edge = self.alphas.shape[0] - 1 if upper else 0
        return float(np.mean(np.argmin(self.per_deletion_mse, axis=1) == edge))


def choose_penalty(
    deletions: Sequence[DeletionT],
    error_fn: Callable[[DeletionT], np.ndarray],
    alphas: np.ndarray,
    policy: str,
    n_groups: int,
) -> PenaltyChoice:
    """The penalty minimising the squared error pooled over the deletions.

    Args:
        deletions: The inner group deletions; each holds out at least one row.
        error_fn: The summed squared error of the held-out rows of one deletion, per penalty.
        alphas: The penalty grid.
        policy: The recorded name of the selection policy.
        n_groups: The number of groups in the training rows.

    Returns:
        The selected penalty with the pooled and the per-deletion mean squared errors.

    Raises:
        ConfigurationError: If there is no deletion, a deletion holds out no row, an error
            does not have one entry per penalty, the pooled error is not finite, or the grid
            is not valid.
    """
    grid = _checked_grid(alphas)
    if not deletions:
        raise ConfigurationError("choose_penalty: no inner deletions")
    summed = np.empty((len(deletions), grid.shape[0]), dtype=np.float64)
    held_rows = np.empty(len(deletions), dtype=np.float64)
    for index, deletion in enumerate(deletions):
        held_rows[index] = np.asarray(deletion.held).shape[0]
        if held_rows[index] == 0:
            raise ConfigurationError(
                f"choose_penalty: deletion {index} holds no row, so it has no error to score"
            )
        errors = np.asarray(error_fn(deletion), dtype=np.float64)
        if errors.shape != grid.shape:
            raise ConfigurationError(
                f"choose_penalty: the error of deletion {index} has shape {errors.shape}; "
                f"expected {grid.shape}"
            )
        summed[index] = errors
    total = summed.sum(axis=0)
    if not np.isfinite(total).all():
        raise ConfigurationError("choose_penalty: non-finite grouped error")
    mse = total / held_rows.sum()
    alpha = float(grid[int(np.argmin(mse))])
    logger.debug("penalty %.4g chosen by %s over %d deletions", alpha, policy, len(deletions))
    return PenaltyChoice(
        alpha=alpha,
        policy=policy,
        n_deletions=len(deletions),
        n_groups=n_groups,
        alphas=grid,
        mse=mse,
        per_deletion_mse=summed / held_rows[:, None],
    )


def squared_error(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Summed squared error of each row of a block of predictions.

    Args:
        predictions: Predictions of shape ``(penalties, rows)``.
        target: The true values of the rows.

    Returns:
        One summed squared error per penalty.

    Raises:
        ConfigurationError: If the block is not two-dimensional or its rows do not match the
            target.
    """
    values = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or truth.ndim != 1 or values.shape[1] != truth.shape[0]:
        raise ConfigurationError(
            f"squared_error: predictions of shape {values.shape} cannot be scored against "
            f"{truth.shape[0]} rows"
        )
    return np.sum((values - truth) ** 2, axis=1)
