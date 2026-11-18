"""Turn the score frame given to the validity panel into one (text, run, score) frame per target.

Two shapes are accepted (see `resolve_target_scores`). Run numbers keep the values the frame
gives them: a column of numpy integers, floats or objects is read as it is, and a column of a
pandas extension type (such as the nullable ``Int64``) that holds a missing value is read as
objects, its numbers as Python integers and its missing values as ``None``. Read through numpy,
such a column would become 64-bit floats, in which two run numbers above ``2**53`` can become
one.

For the panel, `resolve_targets` also records per target the texts whose input rows hold an
infinite score: the target's own rows, or for the construct generated from a long attribute
table every attribute row of the text. The construct is averaged from a copy of the rows in
which an infinite score is missing, so it averages the finite attribute scores only and no
average meets an infinity (the mean of ``+inf`` and ``-inf`` would be a missing value, and numpy
warns of it); the record is read from the raw rows, which keep the infinite values. The panel
refuses a text of the record that has a finite gold value before it averages the target's runs
per text, and drops and counts a text without gold.

`resolve_targets` refuses, before the construct is averaged, a score above
`copernicus.coders.MAX_ABSOLUTE_VALUE` (1e150) in absolute value among the rows a target is built
from, whether or not its text has a gold value, so that no average of the panel overflows.
`resolve_target_scores` refuses such a score only through `construct_scores`, for the generated
construct of a long attribute table under either spelling of its key.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus.coders.results import check_value_limit, construct_scores
from copernicus.errors import ConfigurationError
from copernicus.rubric.identifiers import ROLLUP_SUFFIX, TARGET_CONSTRUCT, target_name_problem
from copernicus.validity._frames import listed, require_columns

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTRUCT_KEYS",
    "RUN_COLUMNS",
    "TargetRuns",
    "resolve_target_scores",
    "resolve_targets",
    "run_numbers",
]

CONSTRUCT_KEYS = (TARGET_CONSTRUCT, f"{TARGET_CONSTRUCT}:{TARGET_CONSTRUCT}")
RUN_COLUMNS = ["text_id", "run_number", "score"]


@dataclass(frozen=True, eq=False)
class TargetRuns:
    """The rows of one target and the texts whose input rows hold an infinite score.

    Attributes:
        runs: Frame with ``text_id`` (string), ``run_number`` and ``score`` (float, NaN when
            missing).
        infinite: Ids of the texts with an infinite score among the rows the target is built
            from, including those whose infinite values the construct averages away.
    """

    runs: pd.DataFrame
    infinite: frozenset[str]


def run_numbers(column: pd.Series) -> np.ndarray:
    """The run numbers of a score frame, each with the value the frame gives it.

    Args:
        column: The ``run_number`` column.

    Returns:
        The values: as numpy reads them, or, for a column of a pandas extension type that holds
        a missing value, as objects with ``None`` for the missing ones.
    """
    if isinstance(column.dtype, pd.api.extensions.ExtensionDtype) and column.hasnans:
        values: np.ndarray = column.to_numpy(dtype=object, na_value=None)
        return values
    return column.to_numpy()


def _score_values(frame: pd.DataFrame) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(frame["score"]):
        raise ConfigurationError(
            f"the score column must be numeric; got dtype {frame['score'].dtype}"
        )
    values: np.ndarray = frame["score"].to_numpy(dtype="float64", na_value=np.nan)
    return values


def _texts_with_infinite(rows: pd.DataFrame) -> frozenset[str]:
    """The text ids of the rows whose score is infinite."""
    return frozenset(rows.loc[np.isinf(rows["score"].to_numpy()), "text_id"])


def _check_limit(key: str, rows: pd.DataFrame) -> None:
    """Refuse a score above `MAX_ABSOLUTE_VALUE` among the rows a target is built from."""
    values = rows["score"].to_numpy(dtype="float64")
    check_value_limit("score", values, rows["text_id"].to_numpy(dtype=object), f"target {key!r}: ")


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


def _construct_runs(long: pd.DataFrame, keep_unnumbered: bool) -> pd.DataFrame:
    """The construct per text and run: the mean of the attribute scores that exist.

    `construct_scores` refuses a score row without a run number, so the construct is averaged
    from the rows that have one. With ``keep_unnumbered`` (as the panel resolves its targets)
    the construct of a text's rows without a run number comes back as one row with a missing
    run number, so that the panel's run-number check refuses a scored text with a finite gold
    value instead of losing the row or the whole text, and counts a text without gold that it
    drops. The run numbers are joined to the missing one as objects, which keeps each as it
    was: joined as numbers, integers would become 64-bit floats, and two run numbers above
    2**53 could become one.
    """
    numeric = long.rename(columns={"score": "value"})
    missing = pd.isna(numeric["run_number"]).to_numpy()
    numbered = construct_scores(numeric[~missing])
    runs = numbered.rename(columns={"construct_score": "score"})[RUN_COLUMNS]
    if not keep_unnumbered:
        return runs
    unnumbered = numeric[missing]
    if unnumbered.empty:
        return runs
    lost = construct_scores(unnumbered.assign(run_number=0)).assign(run_number=np.nan)
    kept = lost.rename(columns={"construct_score": "score"})[RUN_COLUMNS]
    as_objects = {"run_number": object}
    return pd.concat([runs.astype(as_objects), kept.astype(as_objects)], ignore_index=True)


def _attribute_rows(long: pd.DataFrame, attributes: Sequence[str], key: str) -> pd.DataFrame:
    """The rows of the long attribute table that a target key is built from.

    Raises:
        ConfigurationError: Naming the keys that exist, if the key is neither the construct
            nor an attribute of the table.
    """
    if key in CONSTRUCT_KEYS:
        return long
    if key in attributes:
        return long[long["attribute_id"] == key]
    raise ConfigurationError(
        f"target {key!r} matches no scores; available: {listed([TARGET_CONSTRUCT, *attributes])}"
    )


def _attribute_runs(
    scores: pd.DataFrame,
    keys: Sequence[str],
    keep_unnumbered: bool,
    finite_construct: bool,
    bounded: bool,
) -> dict[str, TargetRuns]:
    """Per-target (text, run, score) frames from the long attribute table.

    With ``finite_construct`` the construct is averaged from a copy in which an infinite score is
    missing, so it averages the finite attribute scores only; the texts with an infinite score
    are recorded from the raw rows, which keep the infinite values. With ``bounded`` a score
    above `MAX_ABSOLUTE_VALUE` in the rows a target is built from is refused before the construct
    is averaged.
    """
    require_columns(scores, ["text_id", "run_number", "attribute_id", "score"], "the scores")
    long = pd.DataFrame(
        {
            "text_id": scores["text_id"].astype(str).to_numpy(),
            "run_number": run_numbers(scores["run_number"]),
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
    values = long["score"].to_numpy()
    averaged = long.assign(score=np.where(np.isinf(values), np.nan, values))
    resolved: dict[str, TargetRuns] = {}
    for key in keys:
        rows = _attribute_rows(long, attributes, key)
        if bounded:
            _check_limit(key, rows)
        if key in CONSTRUCT_KEYS:
            runs = _construct_runs(averaged if finite_construct else long, keep_unnumbered)
        else:
            runs = rows[RUN_COLUMNS]
        resolved[key] = TargetRuns(runs.reset_index(drop=True), _texts_with_infinite(rows))
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


def _target_runs(scores: pd.DataFrame, keys: Sequence[str], bounded: bool) -> dict[str, TargetRuns]:
    """Per-target (text, run, score) frames from a frame that carries ``target_id``.

    With ``bounded`` a score above `MAX_ABSOLUTE_VALUE` in the rows of a target is refused.
    """
    require_columns(scores, ["text_id", "score", "target_id"], "the scores")
    has_level = "target_level" in scores.columns
    has_run = "run_number" in scores.columns
    long = pd.DataFrame(
        {
            "text_id": scores["text_id"].astype(str).to_numpy(),
            "run_number": run_numbers(scores["run_number"]) if has_run else 1,
            "target_id": scores["target_id"].astype(str).to_numpy(),
            "score": _score_values(scores),
        }
    )
    if has_level:
        long["target_level"] = scores["target_level"].astype(str).to_numpy()
    levels = set(long["target_level"]) if has_level else set()
    resolved: dict[str, TargetRuns] = {}
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
                f"target {key!r} matches no scores; available: {listed(sorted(available.unique()))}"
            )
        if has_level and level is None and selected["target_level"].nunique() > 1:
            raise ConfigurationError(
                f"target {key!r} exists at several target levels "
                f"{sorted(selected['target_level'].unique())}; write the key as '<level>:<id>'"
            )
        runs = selected[RUN_COLUMNS].reset_index(drop=True)
        _check_unique_runs(runs, key, has_run)
        if bounded:
            _check_limit(key, selected)
        resolved[key] = TargetRuns(runs, _texts_with_infinite(selected))
    return resolved


def _resolve(
    scores: pd.DataFrame,
    keys: Sequence[str],
    keep_unnumbered: bool,
    finite_construct: bool,
    bounded: bool,
) -> dict[str, TargetRuns]:
    """The rows of every key, from either shape of the score frame."""
    if "target_id" in scores.columns:
        return _target_runs(scores, keys, bounded)
    return _attribute_runs(scores, keys, keep_unnumbered, finite_construct, bounded)


def resolve_targets(scores: pd.DataFrame, keys: Sequence[str]) -> dict[str, TargetRuns]:
    """The rows of every target key, as the validity panel reads them, with their infinite texts.

    As `resolve_target_scores`, except that the construct of a long attribute table keeps a
    text's score rows without a run number, so that the panel refuses them for a scored text
    with a finite gold value, under either unit, and counts a text without gold that it drops.
    The texts with an infinite score are recorded from the raw values, and the construct
    averages the finite attribute scores only. A finite score above
    `copernicus.coders.MAX_ABSOLUTE_VALUE` in absolute value in the rows a target is built from
    (every attribute row for the construct) is refused before the construct is averaged.

    Args:
        scores: The score frame.
        keys: Target keys to extract.

    Returns:
        Key to its rows and to the texts whose input rows hold an infinite score.

    Raises:
        ConfigurationError: As `resolve_target_scores`, and if a score of a target's rows is
            above the limit in absolute value, naming the target, its texts (the first ten), the
            largest absolute value, the limit and the remedy.
    """
    return _resolve(scores, keys, keep_unnumbered=True, finite_construct=True, bounded=True)


def resolve_target_scores(scores: pd.DataFrame, keys: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Return one (text, run, score) frame per target key.

    Two shapes are accepted. A long attribute table (``text_id``, ``run_number``,
    ``attribute_id``, ``score``) is keyed by attribute id, and the key ``construct`` is
    the mean of the attribute scores per text and run, ignoring missing ones, over the score
    rows that have a run number (the validity panel refuses a row without one for a scored text
    with a finite gold value; see `resolve_targets`). A frame with a ``target_id`` column
    (similarity or hybrid output; ``run_number`` is taken as 1 when absent) is keyed by
    target id, or by ``target_level`` and target id when that column exists, where a key may
    be written ``construct``, ``<rubric_id>__rollup``, ``<level>:<id>``, or as a plain id
    that exists at one level only. Run numbers keep the values the frame gives them (see the
    module).

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
            ending in ``__rollup``, case ignored), which the naming rule of rubrics refuses;
            and, for the generated construct of a long attribute table, as `construct_scores`
            refuses a score above `copernicus.coders.MAX_ABSOLUTE_VALUE` in absolute value.
    """
    resolved = _resolve(scores, keys, keep_unnumbered=False, finite_construct=False, bounded=False)
    return {key: target.runs for key, target in resolved.items()}
