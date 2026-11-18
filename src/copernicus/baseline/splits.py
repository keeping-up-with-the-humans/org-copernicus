"""Folds and training draws of the NLP-ML baseline as tables.

``BaselineSplits`` holds two tables. ``folds`` gives the test fold of every analysis text for
every seed; ``draws`` gives the texts of the training draw of each fold at each size, or is
``None``, and then the baseline draws them itself from its keyed generators. On construction the
splits check their own form: the columns, their types, no missing value and no duplicate row.
Whether they fit the analysis texts and the settings of a run is checked by
``run_ridge_baseline``, on a checked copy (``checked_splits``) that a table changed after
construction cannot bypass. The splits file is in ``splits_file``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus.baseline.folds import DrawReceipt
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = [
    "DRAW_COLUMNS",
    "FOLD_COLUMNS",
    "INTEGER_COLUMNS",
    "BaselineSplits",
    "Draws",
    "Folds",
    "checked_splits",
    "integer_array",
    "listed",
]

Folds = list[tuple[np.ndarray, np.ndarray]]
"""The training and test row positions of each fold."""
Draws = list[tuple[np.ndarray, DrawReceipt]]
"""The drawn row positions of each fold, with the receipt of the draw."""

FOLD_COLUMNS = ("seed", "text_id", "fold")
"""The columns of the folds table, in their order."""
DRAW_COLUMNS = ("seed", "size", "fold", "text_id")
"""The columns of the draws table, in their order."""
INTEGER_COLUMNS = ("seed", "fold")
"""The columns that hold integers."""
INT64 = np.iinfo(np.int64)
SIZE_COLUMN = "size"
SIZE_LABEL = re.compile(r"full|[1-9][0-9]*")
MAX_IN_MESSAGE = 10


def listed(values: Sequence[object]) -> str:
    """Render up to ten values for a message, with the count of the rest.

    Args:
        values: The values, such as the offending text ids.

    Returns:
        The first ten values joined by commas, followed by ``... (N more)`` when there are more.
    """
    shown = ", ".join(str(value) for value in values[:MAX_IN_MESSAGE])
    rest = len(values) - MAX_IN_MESSAGE
    return f"{shown}, ... ({rest} more)" if rest > 0 else shown


def _is_integer(value: object) -> bool:
    """Tell whether a value is an integer, a numpy integer included, and not a boolean."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)


def _is_label(value: object) -> bool:
    """Tell whether a value can stand for a text id: a string or an integer, not a boolean."""
    return isinstance(value, str) or _is_integer(value)


def _how_many(count: int) -> str:
    """``1 value does not`` or ``<n> values do not``, for a refusal."""
    return "1 value does not" if count == 1 else f"{count} values do not"


def integer_array(values: Sequence[int]) -> np.ndarray:
    """Integers as int64 when every one fits in 64 signed bits, else as Python integers.

    Args:
        values: Python integers of any size.

    Returns:
        An int64 array, or an object array that keeps every value exactly.
    """
    if all(INT64.min <= value <= INT64.max for value in values):
        return np.asarray(values, dtype=np.int64)
    exact = np.empty(len(values), dtype=object)
    exact[:] = list(values)
    return exact


def _check_columns(frame: pd.DataFrame, table: str, expected: tuple[str, ...]) -> None:
    """Refuse a table whose columns are not exactly the expected ones."""
    names = [str(column) for column in frame.columns]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigurationError(f"the {table} table repeats the column(s) {listed(repeated)}")
    absent = [column for column in expected if column not in names]
    if absent:
        raise ConfigurationError(
            f"the {table} table lacks the column(s) {listed(absent)}; it has {listed(names)}"
        )
    extra = [name for name in names if name not in expected]
    if extra:
        raise ConfigurationError(
            f"the {table} table holds columns that splits do not have: {listed(extra)}; its "
            f"columns are {', '.join(expected)}"
        )


def _check_missing(frame: pd.DataFrame, table: str, columns: tuple[str, ...]) -> None:
    """Refuse a table with a missing value, naming the column and the row positions."""
    for column in columns:
        rows = np.flatnonzero(frame[column].isna().to_numpy())
        if rows.size:
            raise ConfigurationError(
                f"the {table} table has missing values in column {column}, in the rows "
                f"{listed(rows.tolist())} (counted from 0)"
            )


def _integers(values: pd.Series, table: str, column: str) -> np.ndarray:
    """An integer column, exact at any size (see ``integer_array``); other values are refused.

    Signed integer columns are int64 already; unsigned ones and columns of Python objects, as
    pandas holds integers beyond 64 bits, are read value by value. A boolean is refused.
    """
    refusal = f"the {table} table's column {column} must hold integers"
    if pd.api.types.is_signed_integer_dtype(values):
        return values.to_numpy(dtype=np.int64)
    if not (pd.api.types.is_unsigned_integer_dtype(values) or pd.api.types.is_object_dtype(values)):
        raise ConfigurationError(f"{refusal}; got dtype {values.dtype}")
    items = values.tolist()
    refused = [item for item in items if not _is_integer(item)]
    if refused:
        raise ConfigurationError(
            f"{refusal}; {_how_many(len(refused))}, for example {refused[0]!r}"
        )
    return integer_array([int(item) for item in items])


def _text_ids(values: pd.Series, table: str) -> np.ndarray:
    """Text ids as strings: strings stay, integers of any size become their decimal strings."""
    refusal = f"the {table} table's column text_id must hold strings or integers"
    if pd.api.types.is_integer_dtype(values):
        return np.asarray([str(item) for item in values.tolist()], dtype=str)
    if not (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)):
        raise ConfigurationError(f"{refusal}; got dtype {values.dtype}")
    items = values.to_numpy(dtype=object).tolist()
    refused = [item for item in items if not _is_label(item)]
    if refused:
        raise ConfigurationError(
            f"{refusal}; {_how_many(len(refused))}, for example {refused[0]!r}"
        )
    return np.asarray([str(item) for item in items], dtype=str)


def _sizes(values: pd.Series) -> np.ndarray:
    """Size labels as the results table writes them: ``"full"`` or the quota in digits."""
    refusal = (
        "the draws table's column size must hold labels as the results table writes them, "
        "such as '50' or 'full'"
    )
    if not (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)):
        raise ConfigurationError(f"{refusal}; got dtype {values.dtype}")
    items = values.to_numpy(dtype=object).tolist()
    not_text = [item for item in items if not isinstance(item, str)]
    if not_text:
        raise ConfigurationError(f"{refusal}; got {not_text[0]!r}")
    unknown = sorted({item for item in items if SIZE_LABEL.fullmatch(item) is None})
    if unknown:
        raise ConfigurationError(
            f"the draws table's column size holds labels that the results table never writes: "
            f"{listed([repr(label) for label in unknown])}; write a size as its quota in digits, "
            f"such as '50', or as 'full'"
        )
    return np.asarray(items, dtype=str)


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    """A table without rows whose columns have their types: int64 integers, strings."""
    return pd.DataFrame(
        {
            column: np.empty(0, dtype=np.int64 if column in INTEGER_COLUMNS else str)
            for column in columns
        }
    )


def _normalised(frame: pd.DataFrame, table: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """A checked copy of a table: the columns in order, typed, with a fresh index.

    A table without rows takes the types of its columns whatever types it came with, as a
    JSON list without values, or ``pd.DataFrame(columns=...)``, gives no type to keep.
    """
    _check_columns(frame, table, columns)
    if frame.empty:
        return _empty(columns)
    _check_missing(frame, table, columns)
    data: dict[str, np.ndarray] = {}
    for column in columns:
        if column in INTEGER_COLUMNS:
            data[column] = _integers(frame[column], table, column)
        elif column == SIZE_COLUMN:
            data[column] = _sizes(frame[column])
        else:
            data[column] = _text_ids(frame[column], table)
    normalised = pd.DataFrame(data, columns=list(columns))
    duplicated = normalised.duplicated(keep="first").to_numpy()
    if duplicated.any():
        rows = normalised.loc[duplicated].drop_duplicates()
        shown = [f"({', '.join(str(value) for value in row)})" for row in rows.itertuples(False)]
        raise ConfigurationError(f"the {table} table holds duplicate rows: {listed(shown)}")
    return normalised


@dataclass(frozen=True, eq=False)
class BaselineSplits:
    """Folds and training draws that a baseline run takes in place of its keyed generators.

    The splits keep checked copies of the tables they are given, with the columns in the order
    below, ``seed`` and ``fold`` as integers (int64 where every value fits in 64 signed bits,
    otherwise Python integers, so that any seed of the settings is kept exactly), ``text_id``
    and ``size`` as strings (integer text ids of any size become their decimal strings, as the
    baseline compares ids) and a fresh index; the rows keep their order, and a table without
    rows takes these types. The dataclass is frozen, but the contents of its tables can still
    be changed; ``run_ridge_baseline`` and ``write_splits`` therefore check a copy first. Whether
    the tables fit the analysis texts and the settings of a run is checked by
    ``run_ridge_baseline``, which names what does not fit.

    Attributes:
        folds: Columns ``seed``, ``text_id`` and ``fold``: the test fold (0 to
            ``k_folds - 1``) of every analysis text, for every seed of the settings.
        draws: ``None``, so that the baseline draws the training texts itself from its keyed
            generators, or columns ``seed``, ``size``, ``fold`` and ``text_id``: the texts of
            the training draw of that fold at that size. ``size`` is written as the results
            table writes it (``"50"``, ``"full"``). A draw of size ``"full"`` may be left out:
            it is the fold's whole training pool.

    Raises:
        ConfigurationError: If a table is not a pandas DataFrame (``draws`` may be ``None``),
            lacks one of its columns, repeats one or holds another, has a missing value, has a
            column of another type (``seed`` and ``fold`` integers, ``text_id`` strings or
            integers, ``size`` the label ``"full"`` or a positive quota in digits), or holds
            a row twice; the message names the table, the column and the rows.
    """

    folds: pd.DataFrame
    draws: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Check the form of both tables and keep their checked copies."""
        if not isinstance(self.folds, pd.DataFrame):
            raise ConfigurationError(
                f"folds must be a pandas DataFrame with the columns {', '.join(FOLD_COLUMNS)}; "
                f"got {type(self.folds).__name__}"
            )
        object.__setattr__(self, "folds", _normalised(self.folds, "folds", FOLD_COLUMNS))
        if self.draws is None:
            return
        if not isinstance(self.draws, pd.DataFrame):
            raise ConfigurationError(
                f"draws must be None or a pandas DataFrame with the columns "
                f"{', '.join(DRAW_COLUMNS)}; got {type(self.draws).__name__}"
            )
        object.__setattr__(self, "draws", _normalised(self.draws, "draws", DRAW_COLUMNS))


def checked_splits(splits: object) -> BaselineSplits:
    """A checked copy of splits, so that a table changed after construction meets every check.

    Args:
        splits: The splits as given.

    Returns:
        New splits built from the tables as they are now.

    Raises:
        ConfigurationError: If ``splits`` is not a ``BaselineSplits``, or its tables are now
            refused by ``BaselineSplits``.
    """
    if not isinstance(splits, BaselineSplits):
        raise ConfigurationError(f"splits must be a BaselineSplits; got {type(splits).__name__}")
    return BaselineSplits(folds=splits.folds, draws=splits.draws)
