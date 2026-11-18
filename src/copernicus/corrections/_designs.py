"""Design matrices of the OLS estimators: a formula over the data's columns or a matrix.

A formula is a right-hand side in the syntax of ``formulaic`` (``"1 + log_chars + C(kind)"``)
evaluated on the data's frame; categorical columns are dummy-coded against their first level
and the intercept, when the formula keeps it, comes first and is named ``Intercept``. No row is
dropped: a missing value that reaches the design is refused. A formula is judged by its design
and not by the columns it reads, so a formula that fills a missing value itself, as
``x.fillna(0)`` does, is evaluated as written, inside a transform or outside. A formula that
turns a missing value into a value without a fill is evaluated too: ``I(x > 0)`` is false where
a column of floats holds NaN, while in a column of a nullable type of pandas the comparison
stays missing and the design is refused. A numpy masked array is read by its data: its mask
is no missing value of the design. Whoever needs every input observed checks the columns
before the call.
When a formula is evaluated it sees the columns of the data, the names that
``formulaic`` provides (its transforms and ``np``) and Python's builtins, and no variable of
this library. Before that, a name that ``formulaic`` reports as read and that is no column of
the data is refused; with version 1.2.2 that includes a builtin outside a stateful transform,
so ``abs(x)`` is refused where ``center(abs(x))`` is evaluated.

These errors are the library's own, each with the error of ``formulaic``, Python or numpy as
its cause where there is one. A formula that cannot be parsed, or that has several parts
separated by ``|``, is a setting that is refused. A formula does not fit the data when its
evaluation fails with one of the errors named under ``formula_design``, or when a column of
its design holds a value that the library does not count as a real number. It counts the
real numbers of Python and of numpy (``numbers.Real``, without the durations of numpy),
fractions and decimals, and turns them into floats. It refuses every other value: a text,
which numpy would read as a number where it can; a complex number, of which numpy would keep
the real part; a duration; a list; and a number of another package that is not registered as
a real number of Python, such as a symbolic expression. A number that cannot become a float,
because it is too large or because it is a decimal that signals an invalid operation, does
not fit the data either. Another error that ``formulaic`` raises while it builds a design is
passed on as it is.
``formulaic`` is imported inside the functions, so importing the correction module stays light.
"""

from __future__ import annotations

import decimal
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
REAL_NUMBERS = (numbers.Real, decimal.Decimal)


def _first_line(error: Exception) -> str:
    """The first line of an error's message, or the name of its class when it has none.

    ``formulaic`` follows the first line of a parsing error with the formula in terminal
    colours, which do not belong in a message of this library; the error itself is kept as
    the cause.
    """
    lines = str(error).splitlines()
    return lines[0] if lines else type(error).__name__


def _other_than_real(column: pd.Series) -> str | None:
    """The type of the first value of a design column that does not count as a real number.

    ``None`` when every value counts. A column of booleans, integers or floats holds real
    numbers only. In a column of another type (objects, the string type of pandas, complex
    numbers, durations) every value is looked at: the real numbers of Python and of numpy,
    fractions and decimals count, and no other value does. A duration of numpy does not
    count, although numpy files it under its integers.
    """
    if column.dtype.kind in NUMERIC_KINDS:
        return None
    for value in column:
        if isinstance(value, np.timedelta64) or not isinstance(value, REAL_NUMBERS):
            return type(value).__name__
    return None


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
        SettingsError: If the formula is empty, has a left-hand side (a ``~``) or several parts
            separated by ``|``, or cannot be parsed: ``formulaic`` raises its
            ``FormulaParsingError``, or Python a ``SyntaxError`` or a ``ValueError`` for a
            factor that is no Python expression, when it is asked which columns the formula
            reads.
        ConfigurationError: If ``formulaic`` reports that the formula reads a column the frame
            lacks or a name that pandas does not resolve to one column of the frame; if the
            evaluation fails with its ``FactorEvaluationError``, a ``ValueError``, a
            ``TypeError``, an ``IndexError`` or an ``ArithmeticError``, as when the formula
            reads an absent column that the report did not name, a missing value reaches its
            design, a column has a type that gives no numbers, contrasts have the wrong form,
            or a decimal signals an invalid operation; if a number of its design is too large
            for a float; if a column of its design holds a value that does not count as a real
            number as this module's docstring defines it (a text, a complex number, a
            duration, a list, a number of another package that is not registered as a real
            number of Python); or if its design is not finite.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise SettingsError(f"{role} must be a non-empty formula such as '1 + log_chars'")
    if "~" in formula:
        raise SettingsError(
            f"{role} must be a right-hand side such as '1 + log_chars', without '~'; "
            f"got {formula!r}"
        )
    from formulaic import Formula, StructuredFormula, model_matrix
    from formulaic.errors import FactorEvaluationError, FormulaParsingError

    # Only the text of the formula enters these two statements, not the data.
    try:
        parsed = Formula(formula)
        required = sorted(str(name) for name in parsed.required_variables)
    except (FormulaParsingError, SyntaxError, ValueError) as error:
        raise SettingsError(
            f"{role} {formula!r} cannot be parsed as a formula: {_first_line(error)}"
        ) from error
    if isinstance(parsed, StructuredFormula):
        raise SettingsError(
            f"{role} must be one right-hand side such as '1 + log_chars'; {formula!r} has "
            "several parts"
        )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ConfigurationError(
            f"{role} {formula!r} names column(s) {list_first_ten(missing)} that the data lack; "
            f"columns: {list_first_ten([str(column) for column in frame.columns])}"
        )
    # pandas gives a frame, not a column, for a name that more than one column has and for a
    # name that is one part of labels of several levels.
    repeated = [name for name in required if isinstance(frame[name], pd.DataFrame)]
    if repeated:
        raise ConfigurationError(
            f"{role} {formula!r} reads {list_first_ten(repeated)}, which pandas does not "
            "resolve to one column of the data: more than one column has the name, or the "
            "labels of the columns have several levels"
        )
    # The evaluation can fail in several ways that are inconsistencies between the formula and
    # the data: a column that the report did not name is absent (FactorEvaluationError); a
    # missing value reaches the design (the ValueError of na_action="raise"); a column of dates
    # or of decimal objects fails inside numpy (TypeError); contrasts of the wrong form or an
    # array without a dimension fail inside formulaic (IndexError); a number too large for a
    # float or a signalling decimal fails in its arithmetic (ArithmeticError). The empty context
    # keeps the variables of this function and of this module out of the formula's reach:
    # formulaic would otherwise hand them to the formula.
    try:
        matrix = model_matrix(formula, frame, na_action="raise", context={})
    except (FactorEvaluationError, ValueError, TypeError, IndexError, ArithmeticError) as error:
        raise ConfigurationError(
            f"{role} {formula!r} cannot be evaluated on the data; the usual causes are a column "
            "that the data lack, a missing value in a column that it reads, and a column of a "
            f"type that gives no numbers: {error}"
        ) from error
    # formulaic 1.2.2 passes on a text column of pandas' string type, the bytes of an Arrow
    # column and a text constant as they are, and complex numbers, durations and lists too.
    # numpy would read a text such as '1.5' as a number and keep only the real part of a
    # complex number, so a design must hold real numbers before it is turned into floats.
    found = [(name, column.dtype, _other_than_real(column)) for name, column in matrix.items()]
    unusable = [f"{name} ({kind} in {dtype})" for name, dtype, kind in found if kind is not None]
    if unusable:
        raise ConfigurationError(
            f"{role} {formula!r} gives design column(s) {list_first_ten(unusable)} with values "
            "that are not real numbers of Python or numpy, fractions or decimals; turn such a "
            "column into floats before the call, or code a text column by its levels with C()"
        )
    try:
        values = np.asarray(matrix, dtype=float)
    except ArithmeticError as error:
        raise ConfigurationError(
            f"{role} {formula!r} gives a design with a number that cannot be turned into floats, "
            f"such as a number too large for a float: {error!r}"
        ) from error
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
            malformed (empty, with a ``~`` or several parts, or one that cannot be parsed).
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
        SettingsError: If both are given, or the formula is malformed (as for
            ``outcome_design``) or removes the intercept (``0 +`` or ``- 1``), which would code
            a categorical column against no level.
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
