"""Turn the frames given to the validity panel into per-target score frames and lookups."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from copernicus.coders.results import construct_scores
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.rubric.identifiers import ROLLUP_SUFFIX, TARGET_CONSTRUCT, target_name_problem

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTRUCT_KEYS",
    "LOG_CHARS_NAME",
    "check_gold_overlap",
    "check_scored_in_table",
    "indexed_gold",
    "numeric_column",
    "resolve_covariate",
    "resolve_groups",
    "resolve_target_scores",
]

CONSTRUCT_KEYS = (TARGET_CONSTRUCT, f"{TARGET_CONSTRUCT}:{TARGET_CONSTRUCT}")
LOG_CHARS_NAME = "log_n_chars"
MAX_LISTED = 10
EXAMPLE_IDS = 3
RUN_COLUMNS = ["text_id", "run_number", "score"]


def _listed(values: Iterable[object]) -> str:
    """Render up to ``MAX_LISTED`` values, saying how many were left out."""
    items = [str(value) for value in values]
    shown = ", ".join(items[:MAX_LISTED])
    rest = len(items) - MAX_LISTED
    return f"{shown}, ... ({rest} more)" if rest > 0 else shown


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    absent = [name for name in columns if name not in frame.columns]
    if absent:
        raise ConfigurationError(f"{what} needs column(s) {absent}; it has {list(frame.columns)}")


def _by_string_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Index a frame by its text ids converted to strings."""
    return frame.assign(text_id=frame["text_id"].astype(str)).set_index("text_id")


def numeric_column(frame: pd.DataFrame, column: str, what: str) -> pd.Series:
    """Return a numeric column as floats, with missing values as NaN.

    Args:
        frame: Frame holding the column.
        column: Column name.
        what: Role of the column in messages (``gold`` or ``covariate``).

    Returns:
        The values as a float series with the frame's index.

    Raises:
        ConfigurationError: If the column is absent or not numeric (boolean counts as
            numeric); numeric values are required, so encode categories first.
    """
    _require_columns(frame, [column], what)
    series = frame[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise ConfigurationError(
            f"{what} column {column!r} must be numeric (dtype {series.dtype}); "
            f"encode string labels as numbers, for example 1 for low and 2 for high"
        )
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    return pd.Series(values, index=frame.index, name=column)


def _score_values(frame: pd.DataFrame) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(frame["score"]):
        raise ConfigurationError(
            f"the score column must be numeric; got dtype {frame['score'].dtype}"
        )
    values: np.ndarray = frame["score"].to_numpy(dtype="float64", na_value=np.nan)
    return values


def _check_unique_runs(runs: pd.DataFrame, key: str, has_run_column: bool) -> None:
    if runs.duplicated(["text_id", "run_number"]).any():
        hint = (
            "filter the frame to one setting first (for example one encoder, reduction and "
            "writer pooling)"
        )
        if not has_run_column:
            hint = (
                "the frame has no run_number column, so every row counts as run 1: add "
                f"run_number when the rows are replicate runs, or {hint}"
            )
        raise ConfigurationError(f"target {key!r} has more than one score per text and run; {hint}")


def _attribute_runs(scores: pd.DataFrame, keys: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Per-target (text, run, score) frames from the long attribute table."""
    _require_columns(scores, ["text_id", "run_number", "attribute_id", "score"], "the scores")
    long = pd.DataFrame(
        {
            "text_id": scores["text_id"].astype(str).to_numpy(),
            "run_number": scores["run_number"].to_numpy(),
            "attribute_id": scores["attribute_id"].astype(str).to_numpy(),
            "score": _score_values(scores),
        }
    )
    if long.duplicated(["text_id", "run_number", "attribute_id"]).any():
        raise ConfigurationError(
            "the scores hold duplicate rows for a text, run and attribute; keep one"
        )
    attributes = sorted(long["attribute_id"].unique())
    clashes = [clash for name in attributes if (clash := target_name_problem(name)) is not None]
    if clashes:
        raise ConfigurationError(
            "the scores hold attribute ids that the naming rule of rubrics refuses, because "
            "they name a target the panel generates (the key construct is the mean of the "
            "attributes): " + "; ".join(clashes) + "; rename them"
        )
    resolved: dict[str, pd.DataFrame] = {}
    for key in keys:
        if key in CONSTRUCT_KEYS:
            constructed = construct_scores(long.rename(columns={"score": "value"}))
            runs = constructed.rename(columns={"construct_score": "score"})[RUN_COLUMNS]
        elif key in attributes:
            runs = long.loc[long["attribute_id"] == key, RUN_COLUMNS]
        else:
            raise ConfigurationError(
                f"target {key!r} matches no scores; available: "
                f"{_listed([TARGET_CONSTRUCT, *attributes])}"
            )
        resolved[key] = runs.reset_index(drop=True)
    return resolved


def _level_and_id(key: str, levels: set[str]) -> tuple[str | None, str]:
    """Split a mapping key into a target level (or None) and a target id."""
    prefix, separator, rest = key.partition(":")
    if separator and prefix in levels:
        return prefix, rest
    if key == TARGET_CONSTRUCT:
        return TARGET_CONSTRUCT, TARGET_CONSTRUCT
    if key.endswith(ROLLUP_SUFFIX):
        return "attribute", key
    return None, key


def _target_runs(scores: pd.DataFrame, keys: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Per-target (text, run, score) frames from a frame that carries ``target_id``."""
    _require_columns(scores, ["text_id", "score", "target_id"], "the scores")
    has_level = "target_level" in scores.columns
    has_run = "run_number" in scores.columns
    long = pd.DataFrame(
        {
            "text_id": scores["text_id"].astype(str).to_numpy(),
            "run_number": scores["run_number"].to_numpy() if has_run else 1,
            "target_id": scores["target_id"].astype(str).to_numpy(),
            "score": _score_values(scores),
        }
    )
    if has_level:
        long["target_level"] = scores["target_level"].astype(str).to_numpy()
    levels = set(long["target_level"]) if has_level else set()
    resolved: dict[str, pd.DataFrame] = {}
    for key in keys:
        level, target_id = _level_and_id(key, levels) if has_level else (None, key)
        mask = long["target_id"] == target_id
        if level is not None:
            mask &= long["target_level"] == level
        selected = long[mask]
        if selected.empty:
            available = (
                (long["target_level"] + ":" + long["target_id"]) if has_level else long["target_id"]
            )
            raise ConfigurationError(
                f"target {key!r} matches no scores; available: "
                f"{_listed(sorted(available.unique()))}"
            )
        if has_level and level is None and selected["target_level"].nunique() > 1:
            raise ConfigurationError(
                f"target {key!r} exists at several target levels "
                f"{sorted(selected['target_level'].unique())}; write the key as '<level>:<id>'"
            )
        runs = selected[RUN_COLUMNS].reset_index(drop=True)
        _check_unique_runs(runs, key, has_run)
        resolved[key] = runs
    return resolved


def resolve_target_scores(scores: pd.DataFrame, keys: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Return one (text, run, score) frame per target key.

    Two shapes are accepted. A long attribute table (``text_id``, ``run_number``,
    ``attribute_id``, ``score``) is keyed by attribute id, and the key ``construct`` is
    the mean of the attribute scores per text and run, ignoring missing ones. A frame with
    a ``target_id`` column (similarity or hybrid output; ``run_number`` is taken as 1 when
    absent) is keyed by target id, or by ``target_level`` and target id when that column
    exists, where a key may be written ``construct``, ``<rubric_id>__rollup``,
    ``<level>:<id>``, or as a plain id that exists at one level only.

    Args:
        scores: The score frame.
        keys: Target keys to extract.

    Returns:
        Key to a frame with columns ``text_id`` (string), ``run_number`` and ``score``
        (float, NaN when missing).

    Raises:
        ConfigurationError: If a column is missing, a key matches nothing or several
            levels, a text has more than one score for a target and run, or an attribute id
            of a long attribute table names a generated target (``construct``, or an id
            ending in ``__rollup``, case ignored), which the naming rule of rubrics refuses.
    """
    if "target_id" in scores.columns:
        return _target_runs(scores, keys)
    return _attribute_runs(scores, keys)


def indexed_gold(gold: pd.DataFrame) -> pd.DataFrame:
    """Index the gold frame by string text id, refusing duplicated ids."""
    _require_columns(gold, ["text_id"], "the gold frame")
    indexed = _by_string_id(gold)
    duplicated = indexed.index[indexed.index.duplicated()].unique()
    if len(duplicated) > 0:
        raise ConfigurationError(f"the gold frame holds duplicate text ids: {_listed(duplicated)}")
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
            f"{len(unknown)} scored text(s) are absent from the table: {_listed(unknown)}"
        )


def resolve_groups(ids: Sequence[str], table: TextTable | None) -> pd.Series:
    """Return the cluster of each text id: the table's ``group_id``, else the id itself."""
    if table is None:
        return pd.Series(list(ids), index=list(ids), dtype=object)
    return _by_string_id(table.frame)["group_id"].astype(str)


def resolve_covariate(
    name: str | None, gold: pd.DataFrame, table: TextTable | None
) -> tuple[str, pd.Series]:
    """Return the covariate name and its values, indexed by string text id.

    With no name the covariate is ``log(n_chars)`` from the table. A name is looked up in
    the indexed gold frame first and then in the table's frame.

    Raises:
        ConfigurationError: If there is no table and no name, the name is in neither
            place, or the column is not numeric.
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
