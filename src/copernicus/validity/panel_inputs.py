"""Turn the gold frame, the table and the covariate given to the validity panel into lookups.

The score frame is resolved into per-target rows by `panel_scores`; `CONSTRUCT_KEYS` and
`resolve_target_scores` are importable from here as well.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from copernicus.coders.limits import floats_within_limit
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.validity._frames import listed, require_columns
from copernicus.validity.panel_scores import CONSTRUCT_KEYS, resolve_target_scores

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTRUCT_KEYS",
    "LOG_CHARS_NAME",
    "check_gold_overlap",
    "check_group_ids",
    "check_scored_in_table",
    "indexed_gold",
    "numeric_column",
    "require_groups",
    "resolve_covariate",
    "resolve_groups",
    "resolve_target_scores",
]

LOG_CHARS_NAME = "log_n_chars"
EXAMPLE_IDS = 3


def _by_string_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Index a frame by its text ids converted to strings."""
    return frame.assign(text_id=frame["text_id"].astype(str)).set_index("text_id")


def numeric_column(frame: pd.DataFrame, column: str, what: str) -> pd.Series:
    """Return a numeric column as floats, with missing values as NaN.

    Every row of the column is checked, whether its text was scored or not: a finite value
    above `copernicus.coders.MAX_ABSOLUTE_VALUE` (1e150) in absolute value is refused, so that
    a sum of the values stays inside the range of 64-bit floats. A missing value and
    an infinity are returned as they are.

    Args:
        frame: Frame holding the column, indexed by text id.
        column: Column name.
        what: Role of the column in messages (``gold`` or ``covariate``).

    Returns:
        The values as a float series with the frame's index.

    Raises:
        ConfigurationError: If the column is absent or not numeric (boolean counts as
            numeric); numeric values are required, so encode categories first. Also if a
            finite value is above the limit in absolute value, naming the column, its texts
            (the first ten), the largest absolute value, the limit and the remedy.
    """
    require_columns(frame, [column], what)
    series = frame[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise ConfigurationError(
            f"{what} column {column!r} must be numeric (dtype {series.dtype}); "
            f"encode string labels as numbers, for example 1 for low and 2 for high"
        )
    ids = frame.index.to_numpy(dtype=object)
    values = floats_within_limit(f"{what} value in column {column!r}", series, ids)
    return pd.Series(values, index=frame.index, name=column)


def indexed_gold(gold: pd.DataFrame) -> pd.DataFrame:
    """Index the gold frame by string text id, refusing duplicated ids."""
    require_columns(gold, ["text_id"], "the gold frame")
    indexed = _by_string_id(gold)
    duplicated = indexed.index[indexed.index.duplicated()].unique()
    if len(duplicated) > 0:
        raise ConfigurationError(f"the gold frame holds duplicate text ids: {listed(duplicated)}")
    return indexed


def check_gold_overlap(scored_ids: Sequence[str], gold_ids: pd.Index) -> None:
    """Raise when no scored text has a row in the gold frame, which points to id formats.

    Raises:
        ConfigurationError: Showing a few ids of each side, if no scored id is a gold id.
    """
    if not gold_ids.isin(scored_ids).any():
        scored = ", ".join(scored_ids[:EXAMPLE_IDS])
        gold = ", ".join(str(text) for text in gold_ids[:EXAMPLE_IDS])
        raise ConfigurationError(
            f"none of the {len(scored_ids)} scored texts has a row in the gold frame; text ids "
            f"are compared as strings, so write them alike on both sides (zero padding, "
            f"prefixes, numbers read as numbers): scored ids such as {scored}; gold ids such "
            f"as {gold}"
        )


def check_scored_in_table(scored_ids: Iterable[str], table: TextTable) -> None:
    """Raise when a scored text is missing from the table."""
    known = set(table.frame["text_id"].astype(str))
    unknown = sorted(set(scored_ids) - known)
    if unknown:
        raise ConfigurationError(
            f"{len(unknown)} scored text(s) are absent from the table: {listed(unknown)}"
        )


def resolve_groups(ids: Sequence[str], table: TextTable | None) -> pd.Series:
    """Return the cluster of each text id: the table's ``group_id``, else the id itself."""
    if table is None:
        return pd.Series(list(ids), index=list(ids), dtype=object)
    return _by_string_id(table.frame)["group_id"].astype(str)


def _ids_described(pairs: pd.DataFrame, keys: np.ndarray) -> str:
    """Describe the raw ids of each set of rows that share a key, as ``1 (int) and '1' (str)``."""
    raw = pairs["raw"].to_numpy()
    sets = [
        " and ".join(f"{value!r} ({type(value).__name__})" for value in raw[keys == key])
        for key in pd.unique(keys)
    ]
    return listed(sets)


def check_group_ids(table: TextTable) -> None:
    """Refuse group ids that would merge or split when the panel writes them as text.

    `resolve_groups` writes every group id as text, so two different ids that read alike
    (``1`` and ``"1"``) would become one group, and one id written two ways (``1`` and
    ``1.0``, equal values) would become two. Compared once per group, either would change the
    units without notice.

    Args:
        table: The text table whose ``group_id`` column is checked.

    Raises:
        ConfigurationError: Naming the ids with their types, if two different ids read alike
            as text, or two equal ids read differently.
    """
    raw = table.frame["group_id"]
    pairs = pd.DataFrame(
        {"raw": raw.to_numpy(dtype=object), "text": raw.astype(str).to_numpy(dtype=object)}
    ).drop_duplicates()
    texts = pairs["text"].to_numpy()
    merged = pairs["text"].duplicated(keep=False).to_numpy()
    if merged.any():
        raise ConfigurationError(
            "unit 'group' identifies a group by its id written as text, but the table's group "
            f"ids {_ids_described(pairs[merged], texts[merged])} read alike as text; rename "
            "one of them"
        )
    codes, _ = pd.factorize(pairs["raw"])
    split = pd.Series(codes).duplicated(keep=False).to_numpy()
    if split.any():
        raise ConfigurationError(
            "unit 'group' identifies a group by its id written as text, but the table's group "
            f"ids {_ids_described(pairs[split], codes[split])} are equal values that read "
            "differently as text; write each group's id the same way on all its texts"
        )


def require_groups(table: TextTable | None) -> None:
    """Refuse ``unit="group"`` unless every text has a group whose id keeps it apart.

    Raises:
        ConfigurationError: If there is no table, the table was read without a group column
            (``group_defaulted``), so that every text would be its own group, or two of its
            group ids merge or split when written as text (see `check_group_ids`).
    """
    if table is None:
        raise ConfigurationError(
            "unit 'group' needs a group column, and no table was given; pass the text table "
            "with its group column, or compare once per text with unit 'text'"
        )
    if table.group_defaulted:
        raise ConfigurationError(
            "unit 'group' needs a group column, but the table was read without one, so every "
            "text is its own group; name the column that holds the groups (group_id= in "
            "Python, --group-column on the command line), or compare once per text with unit "
            "'text'"
        )
    check_group_ids(table)


def resolve_covariate(
    name: str | None, gold: pd.DataFrame, table: TextTable | None
) -> tuple[str, pd.Series]:
    """Return the covariate name and its values, indexed by string text id.

    With no name the covariate is ``log(n_chars)`` from the table. A name is looked up in
    the indexed gold frame first and then in the table's frame.

    Raises:
        ConfigurationError: If there is no table and no name, the name is in neither
            place, or the column is not numeric or holds a finite value above
            `copernicus.coders.MAX_ABSOLUTE_VALUE` in absolute value (see `numeric_column`).
    """
    if name is None:
        if table is None:
            raise ConfigurationError(
                "without a table the covariate cannot default to log characters; pass a "
                "table or a covariate column of the gold frame"
            )
        by_id = _by_string_id(table.frame)
        return LOG_CHARS_NAME, np.log(by_id["n_chars"].astype(float))
    if name in gold.columns:
        return name, numeric_column(gold, name, "covariate")
    if table is not None and name in table.frame.columns:
        return name, numeric_column(_by_string_id(table.frame), name, "covariate")
    where = "the gold frame" if table is None else "the gold frame or the table"
    raise ConfigurationError(f"covariate {name!r} is not a column of {where}")
