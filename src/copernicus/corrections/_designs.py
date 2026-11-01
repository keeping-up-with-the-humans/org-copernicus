"""Design matrices of the OLS estimators: a formula over the data's columns or a matrix.

A formula is a right-hand side in the syntax of ``formulaic`` (``"1 + log_chars + C(kind)"``)
evaluated on the data's frame; categorical columns are dummy-coded against their first level
and the intercept, when the formula keeps it, comes first and is named ``Intercept``. Every
column the formula reads must be present and observed on every text, so no row is dropped.
``formulaic`` is imported inside the functions, so importing the correction module stays light.
"""

from __future__ import annotations

import logging
import numbers

import numpy as np
import pandas as pd

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.data import CorrectionData
from copernicus.errors import ConfigurationError, SettingsError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = [
    "INTERCEPT",
    "coefficient_index",
    "covariate_design",
    "formula_design",
    "outcome_design",
]

INTERCEPT = "Intercept"
NUMERIC_KINDS = "biuf"


def formula_design(
    frame: pd.DataFrame, formula: str, *, role: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The design matrix of a right-hand-side formula over the frame's columns.

    Args:
        frame: The data's frame.
        formula: A right-hand side such as ``"1 + log_chars"``.
        role: How the formula is named in the messages.

    Returns:
        The design as a float matrix with one row per row of the frame, and its column names.

    Raises:
        SettingsError: If the formula is empty or has a left-hand side (a ``~``).
        ConfigurationError: If the formula reads a column the frame lacks or a column with a
            missing value (also inside a stateful transform such as ``scale()``), or its design
            is not finite.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise SettingsError(f"{role} must be a non-empty formula such as '1 + log_chars'")
    if "~" in formula:
        raise SettingsError(
            f"{role} must be a right-hand side such as '1 + log_chars', without '~'; "
            f"got {formula!r}"
        )
    from formulaic import Formula, model_matrix
    from formulaic.errors import FactorEvaluationError

    required = sorted(str(name) for name in Formula(formula).required_variables)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ConfigurationError(
            f"{role} {formula!r} names column(s) {list_first_ten(missing)} that the data lack; "
            f"columns: {list_first_ten([str(column) for column in frame.columns])}"
        )
    incomplete = [name for name in required if bool(frame[name].isna().any())]
    if incomplete:
        raise ConfigurationError(
            f"{role} {formula!r} reads column(s) {list_first_ten(incomplete)} with missing "
            "values; the design must be observed on every text"
        )
    # Stateful transforms such as scale(), center(), poly() or bs() report no required columns,
    # so a column they read can still be absent (FactorEvaluationError) or incomplete (the
    # ValueError of na_action="raise"); both are inconsistencies between the formula and the data.
    try:
        matrix = model_matrix(formula, frame, na_action="raise")
    except (FactorEvaluationError, ValueError) as error:
        raise ConfigurationError(
            f"{role} {formula!r} cannot be evaluated on the data, which lack a column it reads "
            f"or have a missing value in one: {error}"
        ) from error
    values = np.asarray(matrix, dtype=float)
    names = tuple(str(name) for name in matrix.columns)
    if values.shape[0] != len(frame) or not np.isfinite(values).all():
        raise ConfigurationError(
            f"{role} {formula!r} does not give a finite value in every column for every text"
        )
    logger.debug("design of %s %r: columns %s", role, formula, names)
    return values, names


def _matrix_design(data: CorrectionData, x: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """A numeric matrix with one finite row per text; its columns are named ``x0``, ``x1``, ..."""
    values = np.asarray(x)
    if values.ndim != 2 or values.shape[0] != data.n or values.shape[1] == 0:
        raise ConfigurationError(
            f"X must be a matrix with one row per text ({data.n} rows) and at least one column; "
            f"got shape {values.shape}"
        )
    if values.dtype.kind not in NUMERIC_KINDS:
        raise ConfigurationError(f"X must be numeric, got dtype {values.dtype}")
    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ConfigurationError("X must be finite in every entry")
    return values, tuple(f"x{index}" for index in range(values.shape[1]))


def outcome_design(
    data: CorrectionData, formula: str | None, x: np.ndarray | None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The design of a regression of the construct: from exactly one of a formula and a matrix.

    Args:
        data: The correction data.
        formula: A right-hand side over the data's columns, or ``None``.
        x: A matrix with one row per text in data order, or ``None``.

    Returns:
        The design and its column names (``x0``, ``x1``, ... for a matrix).

    Raises:
        SettingsError: If both or neither of ``formula`` and ``x`` are given, or the formula is
            malformed.
        ConfigurationError: If the design does not match the data.
    """
    if (formula is None) == (x is None):
        raise SettingsError("give exactly one of formula and X")
    if formula is not None:
        values, names = formula_design(data.frame, formula, role="formula")
    else:
        values, names = _matrix_design(data, np.asarray(x))
    _check_unique(names)
    return values, names


def covariate_design(
    data: CorrectionData, covariates: str | None, x: np.ndarray | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The covariate columns ``W`` of the design ``[1, construct, W]``, without an intercept.

    The formula is evaluated with its intercept, so categorical columns are coded against their
    first level, and the intercept column is then dropped (the design adds its own). A matrix
    is taken as it stands; a constant column would duplicate the intercept and is refused.

    Args:
        data: The correction data.
        covariates: A right-hand side over the data's columns, or ``None``.
        x: A matrix with one row per text in data order, or ``None``; at most one of
            ``covariates`` and ``x`` is given, and neither means no covariates.

    Returns:
        The covariate matrix (zero columns for neither) and its column names (``x0``, ``x1``,
        ... for a matrix).

    Raises:
        SettingsError: If both are given, or the formula is malformed or removes the intercept
            (``0 +`` or ``- 1``), which would code a categorical column against no level.
        ConfigurationError: If the formula or the matrix does not match the data, or a column of
            the matrix is constant.
    """
    if covariates is not None and x is not None:
        raise SettingsError("give at most one of covariates and X")
    if x is not None:
        values, names = _matrix_design(data, np.asarray(x))
        constant = [name for index, name in enumerate(names) if is_constant(values[:, index])]
        if constant:
            raise ConfigurationError(
                f"X columns {list_first_ten(constant)} are constant: the design [1, construct, X] "
                "holds its own intercept"
            )
        return values, names
    if covariates is None:
        return np.empty((data.n, 0)), ()
    values, names = formula_design(data.frame, covariates, role="covariates")
    if INTERCEPT not in names:
        raise SettingsError(
            "covariates must keep the intercept, which the design [1, construct, covariates] "
            f"always holds; drop '0 +' or '- 1' from {covariates!r}"
        )
    keep = [index for index, name in enumerate(names) if name != INTERCEPT]
    return values[:, keep], tuple(names[index] for index in keep)


def _check_unique(names: tuple[str, ...]) -> None:
    """Refuse a design whose column names repeat, so every coefficient can be named."""
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigurationError(f"design column names repeat: {list_first_ten(repeated)}")


def coefficient_index(columns: tuple[str, ...], coefficient: str | int) -> int:
    """The position of the reported coefficient in the design.

    Args:
        columns: The design's column names.
        coefficient: A column name, or a position counted from 0 (the intercept, when the
            design has one).

    Returns:
        The position.

    Raises:
        ConfigurationError: If the name is not a column of the design or the position is not
            one of its columns.
    """
    _check_unique(columns)
    if isinstance(coefficient, str):
        if coefficient not in columns:
            raise ConfigurationError(
                f"coefficient {coefficient!r} is not a column of the design; columns: "
                f"{list_first_ten(list(columns))}"
            )
        return columns.index(coefficient)
    if isinstance(coefficient, bool) or not isinstance(coefficient, numbers.Integral):
        raise ConfigurationError(
            f"coefficient must be a column position or a column name; got {coefficient!r}"
        )
    if not 0 <= coefficient < len(columns):
        raise ConfigurationError(
            f"coefficient position {coefficient} is outside the design's {len(columns)} "
            f"columns: {list_first_ten(list(columns))}"
        )
    return int(coefficient)
