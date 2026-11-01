"""Alignment of the baseline inputs on the rows of the gold table.

``resolve_encoder_sets`` expands the encoder-set names into embedding keys,
``align_inputs`` turns the gold table, the text table, the embedding sets and an optional
coder score into arrays in one common row order (the rows of the gold table), and
``binary_classes`` decides whether the gold is binary; ids are compared as strings
throughout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus.baseline.metrics import classes_are_separable
from copernicus.data.table import GROUP_COLUMN, TextTable
from copernicus.embeddings.sets import EmbeddingSet
from copernicus.errors import ConfigurationError

__all__ = [
    "STACKED",
    "AlignedInputs",
    "align_inputs",
    "binary_classes",
    "resolve_encoder_sets",
]

SINGLE = "single"
STACKED = "stacked"
MAX_IDS_IN_MESSAGE = 10


@dataclass(frozen=True)
class AlignedInputs:
    """Arrays of the analysis rows, all in the row order of the gold table.

    Attributes:
        text_ids: Text ids as strings.
        gold: Gold value per row.
        group_id: Group label per row, as strings.
        unit_id: Analysis unit per row, as strings.
        unit_column: Column of the text table that holds the unit (``group_id`` by default).
        blocks: Embedding vectors per encoder key, reordered to the gold rows.
        score: Coder score per row, or ``None``.
    """

    text_ids: np.ndarray
    gold: np.ndarray
    group_id: np.ndarray
    unit_id: np.ndarray
    unit_column: str
    blocks: Mapping[str, np.ndarray]
    score: np.ndarray | None


def _listed(values: Sequence[object]) -> str:
    """Render up to ``MAX_IDS_IN_MESSAGE`` values, with the count of the rest."""
    shown = ", ".join(str(value) for value in values[:MAX_IDS_IN_MESSAGE])
    rest = len(values) - MAX_IDS_IN_MESSAGE
    return f"{shown}, ... ({rest} more)" if rest > 0 else shown


def resolve_encoder_sets(names: Sequence[str], keys: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Expand encoder-set names into the embedding keys that make up each set.

    ``"single"`` is one set per key, named by the key; ``"stacked"`` concatenates every key in
    mapping order; any other name must be a key.

    Args:
        names: Requested encoder-set names.
        keys: Keys of the embedding mapping, in mapping order.

    Returns:
        The set name mapped to its keys, in request order.

    Raises:
        ConfigurationError: If a name is unknown, ``"stacked"`` is requested with fewer than
            two keys, or a set would be produced more than once.
    """
    resolved: dict[str, tuple[str, ...]] = {}
    produced: list[str] = []
    for name in names:
        new: dict[str, tuple[str, ...]]
        if name == SINGLE:
            new = {key: (key,) for key in keys}
        elif name == STACKED:
            if len(keys) < 2:
                raise ConfigurationError(
                    f"encoder set 'stacked' needs at least two embedding sets; got {len(keys)}"
                )
            new = {STACKED: tuple(keys)}
        elif name in keys:
            new = {name: (name,)}
        else:
            raise ConfigurationError(
                f"unknown encoder set {name!r}; use 'single', 'stacked' or one of the "
                f"embedding keys: {_listed(list(keys))}"
            )
        produced.extend(new)
        resolved.update(new)
    repeated = sorted({name for name in produced if produced.count(name) > 1})
    if repeated:
        raise ConfigurationError(f"encoder set(s) {_listed(repeated)} requested more than once")
    return resolved


def _gold_arrays(gold: pd.DataFrame, gold_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Validated text ids (strings) and numeric gold of the gold table."""
    absent = [name for name in ("text_id", gold_column) if name not in gold.columns]
    if absent:
        raise ConfigurationError(
            f"gold table lacks column(s) {_listed(absent)}; columns: {_listed(list(gold.columns))}"
        )
    column = gold[gold_column]
    if not pd.api.types.is_numeric_dtype(column):
        raise ConfigurationError(
            f"gold column {gold_column!r} must be numeric (a boolean counts); "
            f"got dtype {column.dtype}; code categories as numbers first"
        )
    ids = gold["text_id"].astype(str).to_numpy()
    values = column.astype(float).to_numpy()
    repeated = pd.Series(ids)[pd.Series(ids).duplicated()].unique().tolist()
    if repeated:
        raise ConfigurationError(f"gold table has duplicate text ids: {_listed(repeated)}")
    unusable = ids[~np.isfinite(values)].tolist()
    if unusable:
        raise ConfigurationError(
            f"gold column {gold_column!r} is missing or not finite for text ids {_listed(unusable)}"
        )
    return ids, values


def _positions(index_ids: np.ndarray, wanted: np.ndarray, source: str) -> np.ndarray:
    """Row positions of the wanted ids in a source's ids; missing ids raise."""
    positions = pd.Index(index_ids).get_indexer(wanted)
    absent = wanted[positions < 0].tolist()
    if absent:
        raise ConfigurationError(f"gold texts missing from {source}: {_listed(absent)}")
    return positions


def _table_labels(
    table: TextTable, ids: np.ndarray, unit_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Group and unit labels (strings) of the gold texts from the text table."""
    frame = table.frame
    if unit_column not in frame.columns:
        raise ConfigurationError(
            f"unit_id {unit_column!r} is not a column of the text table; columns: "
            f"{_listed(list(frame.columns))}"
        )
    table_ids = frame["text_id"].astype(str).to_numpy()
    if pd.Series(table_ids).duplicated().any():
        raise ConfigurationError("text table ids are not unique after conversion to strings")
    rows = _positions(table_ids, ids, "the text table")
    units = frame[unit_column].iloc[rows]
    if units.isna().any():
        raise ConfigurationError(
            f"unit column {unit_column!r} is missing for text ids "
            f"{_listed(ids[units.isna().to_numpy()].tolist())}"
        )
    return frame[GROUP_COLUMN].iloc[rows].astype(str).to_numpy(), units.astype(str).to_numpy()


def _check_units_nest(groups: np.ndarray, units: np.ndarray, unit_column: str) -> None:
    """Refuse an analysis unit whose texts lie in more than one group."""
    spans = pd.Series(groups).groupby(units).nunique()
    straddling = spans.index[spans > 1].tolist()
    if straddling:
        raise ConfigurationError(
            f"analysis units must nest within groups, because the folds and the draws take "
            f"whole groups; unit column {unit_column!r} has units in more than one group: "
            f"{_listed(straddling)}"
        )


def _score_array(coder_score: pd.Series, ids: np.ndarray) -> np.ndarray:
    """Coder score per gold text, with every text present and finite."""
    if not pd.api.types.is_numeric_dtype(coder_score):
        named = (
            "the coder score"
            if coder_score.name is None
            else f"coder score column {coder_score.name!r}"
        )
        raise ConfigurationError(
            f"{named} must be numeric (a boolean counts); got dtype {coder_score.dtype}"
        )
    index = coder_score.index.astype(str)
    if index.duplicated().any():
        raise ConfigurationError("coder score has duplicate text ids")
    values = pd.Series(coder_score.to_numpy(dtype=float, na_value=np.nan), index=index)
    absent = ids[~pd.Index(ids).isin(index)].tolist()
    if absent:
        raise ConfigurationError(f"coder score lacks gold text ids: {_listed(absent)}")
    aligned = values.reindex(ids).to_numpy()
    unusable = ids[~np.isfinite(aligned)].tolist()
    if unusable:
        raise ConfigurationError(f"coder score is not finite for text ids: {_listed(unusable)}")
    return aligned


def align_inputs(
    embeddings: Mapping[str, EmbeddingSet],
    used_keys: Sequence[str],
    gold: pd.DataFrame,
    table: TextTable,
    *,
    gold_column: str,
    unit_id: str | None,
    coder_score: pd.Series | None,
) -> AlignedInputs:
    """Align gold, groups, units, encoder vectors and the coder score on the gold rows.

    Args:
        embeddings: Embedding sets by encoder name.
        used_keys: The keys of ``embeddings`` that the requested encoder sets need.
        gold: Table with ``text_id`` and ``gold_column``; its rows are the analysis rows.
        table: Text table giving ``group_id`` and the unit column.
        gold_column: Name of the numeric gold column.
        unit_id: Column of ``table.frame`` holding the analysis unit; ``None`` uses
            ``group_id``.
        coder_score: Optional score indexed by text id.

    Returns:
        The aligned arrays.

    Raises:
        ConfigurationError: If the gold is non-numeric, incomplete or duplicated, a gold
            text is absent from the table, an encoder set or the coder score, the unit
            column is unknown, an analysis unit spans more than one group, or the coder
            score is not numeric.
    """
    ids, values = _gold_arrays(gold, gold_column)
    unit_column = GROUP_COLUMN if unit_id is None else unit_id
    groups, units = _table_labels(table, ids, unit_column)
    _check_units_nest(groups, units, unit_column)
    blocks = {
        key: np.asarray(
            embeddings[key].vectors[_positions(embeddings[key].ids, ids, f"encoder set {key!r}")],
            dtype=np.float64,
        )
        for key in used_keys
    }
    score = None if coder_score is None else _score_array(coder_score, ids)
    return AlignedInputs(
        text_ids=ids,
        gold=values,
        group_id=groups,
        unit_id=units,
        unit_column=unit_column,
        blocks=blocks,
        score=score,
    )


def binary_classes(
    aligned: AlignedInputs, gold_column: str, binary: bool | None
) -> tuple[float, float] | None:
    """The two values of a binary gold, or ``None`` when the gold is not treated as binary.

    ``binary=None`` treats the gold as binary when it takes exactly two values, ``True``
    requires exactly two, and ``False`` never treats it as binary. The fold metrics compare
    the unit means of a binary gold with its two values (the larger is the positive class),
    so every analysis unit must hold one value only; a unit holding both is refused here
    rather than leaving AUC and Somers' D undefined in every fold.

    Args:
        aligned: The aligned inputs.
        gold_column: Name of the gold column, for messages.
        binary: The ``binary`` setting.

    Returns:
        The smaller and the larger gold value, or ``None``.

    Raises:
        ConfigurationError: If ``binary`` is True and the gold does not take exactly two
            values, the two values differ by floating-point noise only (gold coded 0.3 and
            0.1 + 0.2), or a binary gold takes both values within one analysis unit; the
            message names the gold column, or the unit column and the first such units.
    """
    values = np.unique(aligned.gold)
    if binary is False or (binary is None and values.shape[0] != 2):
        return None
    if values.shape[0] != 2:
        raise ConfigurationError(
            f"binary is set, but gold column {gold_column!r} takes {values.shape[0]} distinct "
            f"value(s); a binary gold takes exactly two"
        )
    if not classes_are_separable(float(values[0]), float(values[1])):
        raise ConfigurationError(
            f"gold column {gold_column!r} takes two values that cannot be told apart, "
            f"{float(values[0])!r} and {float(values[1])!r}: they differ by floating-point noise "
            f"only, not as two classes; round the gold to one value per class"
        )
    per_unit = pd.Series(aligned.gold).groupby(aligned.unit_id).nunique()
    mixed = per_unit.index[per_unit > 1].tolist()
    if mixed:
        raise ConfigurationError(
            f"a binary gold must be constant within each analysis unit, but unit column "
            f"{aligned.unit_column!r} has units holding both values of {gold_column!r}: "
            f"{_listed(mixed)}; choose a unit within which the gold is constant, such as "
            f"text_id (unit_id; --unit-id text_id on the command line), or set binary to False "
            f"(--no-binary on the command line)"
        )
    return float(values[0]), float(values[1])
