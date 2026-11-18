"""The limit on the size of scores, gold values and covariates.

`construct_scores` and the validity panel refuse a finite value above `MAX_ABSOLUTE_VALUE` in
absolute value, so that the sums of their averages stay inside the range of 64-bit floats. The
check reads a column as 64-bit floats. A number that is finite where it stands and has no
finite float, such as a decimal of ``1e400`` or an integer of 400 digits in an object column,
is above the limit and is refused with the others.
"""

from __future__ import annotations

import decimal
import logging
import math

import numpy as np
import pandas as pd

from copernicus.coders.store_cells import listed_ids
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["MAX_ABSOLUTE_VALUE", "check_value_limit", "floats_within_limit"]

MAX_ABSOLUTE_VALUE = 1e150
"""The limit on the absolute value of the inputs of construct scoring and of the validity panel.

`construct_scores` refuses a finite attribute score, and `copernicus.validity.validity_panel` a
finite score, gold value or covariate, whose absolute value as a 64-bit float is above it, with
``ConfigurationError``; other functions do not apply it. Every value up to this limit can be
squared in 64-bit floats (the square
of ``1e150`` is ``1e300``, and the largest float is about ``1.8e308``), and a sum of such values
stays finite for any number of rows that a table can hold, so a mean of accepted values does not
overflow. A missing value and an infinity are left to the rules of each function.
"""


def check_value_limit(
    what: str,
    values: np.ndarray,
    ids: np.ndarray,
    prefix: str = "",
    beyond: np.ndarray | None = None,
) -> None:
    """Refuse finite values whose absolute value is above `MAX_ABSOLUTE_VALUE`.

    A missing value and an infinity are not checked: each function keeps its own rule for them.

    Args:
        what: The kind of value, as the message names it: ``"score"``, or a gold value or a
            covariate value with its column, such as ``"gold value in column 'rating'"``.
        values: The values as 64-bit floats.
        ids: The text id of each value, in the order of ``values``.
        prefix: Start of the message, for example the target that the values belong to.
        beyond: True where a value is finite in its column and has no finite 64-bit float;
            such a value is above the limit although its float is an infinity.

    Raises:
        ConfigurationError: If a finite value is above the limit, naming the texts that hold
            such a value (the first ten), the largest absolute value among them as a float (or
            that it is beyond the range of 64-bit floats), the limit, and the remedy, to divide
            the column by one constant before the call.
    """
    magnitudes = np.abs(values)
    above = np.isfinite(magnitudes) & (magnitudes > MAX_ABSOLUTE_VALUE)
    largest = repr(float(np.max(magnitudes[above]))) if above.any() else ""
    if beyond is not None and beyond.any():
        above = above | beyond
        largest = "beyond the range of 64-bit floats"
    if not above.any():
        return
    named = sorted({str(one) for one in ids[above]})
    raise ConfigurationError(
        f"{prefix}text(s) {listed_ids(named)} have a {what} above {MAX_ABSOLUTE_VALUE!r} in "
        f"absolute value (the largest is {largest}); the library refuses a score, a gold value "
        "or a covariate above that limit, because sums and squares of larger values can overflow "
        "the range of 64-bit floats: divide the column by one constant before the call"
    )


def _is_infinity(value: object) -> bool:
    """Whether a cell holds an infinity where it stands, and not a number too large for a float.

    The cell is compared with the two infinities as it is, without being turned into a float
    first, so that a large integer, fraction or decimal does not count and an infinity of
    another package's number type does.
    """
    if isinstance(value, decimal.Decimal):
        return value.is_infinite()
    try:
        return bool(value == math.inf or value == -math.inf)
    except (TypeError, ArithmeticError):
        return False


def _one_float(value: object) -> float:
    """The float of one cell: NaN for a missing one, an infinity for a number without a float."""
    if pd.isna(value):  # type: ignore[call-overload]
        return math.nan
    try:
        return float(value)  # type: ignore[arg-type]
    except OverflowError:
        return math.inf


def floats_within_limit(
    what: str, column: pd.Series, ids: np.ndarray, prefix: str = ""
) -> np.ndarray:
    """Return a numeric column as 64-bit floats after the check of `check_value_limit`.

    A column of 64-bit floats, of integers or of booleans is converted as a whole. In another
    column (an object column, a column of decimals) a number can be finite and have no finite
    float; it is refused as a value above the limit, where the plain conversion would make it
    an infinity or fail with ``OverflowError``.

    Args:
        what: The kind of value, as for `check_value_limit`.
        column: The numeric column; missing values become NaN.
        ids: The id of each value, in the order of the column.
        prefix: Start of the message.

    Returns:
        The values as 64-bit floats, missing ones as NaN.

    Raises:
        ConfigurationError: As `check_value_limit`.
    """
    # A number without a finite float is reported by the check below, so numpy's own warning of
    # an overflow in the conversion is turned off.
    with np.errstate(over="ignore"):
        try:
            values = column.to_numpy(dtype="float64", na_value=np.nan)
        except OverflowError:
            values = np.array([_one_float(cell) for cell in column], dtype="float64")
    beyond = None
    if getattr(column.dtype, "kind", "O") not in "fiub" and np.isinf(values).any():
        cells = column.to_numpy(dtype=object)
        beyond = np.array(
            [
                bool(np.isinf(v)) and not _is_infinity(cell)
                for v, cell in zip(values, cells, strict=True)
            ],
            dtype=bool,
        )
    check_value_limit(what, values, ids, prefix, beyond)
    return values
