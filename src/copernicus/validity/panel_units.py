"""The rows a validity-panel target is compared on: one per text, or one per group.

A gold value can belong to a group of texts rather than to one text, for example a
questionnaire answered by the person who wrote several texts. With ``unit="group"`` the panel
compares one score per group with the group's gold value; a group's score is the mean over its
texts that were scored for the target. This module reduces the run-averaged texts of a target
to those group rows, and the run scores to the group's mean score in each run that all its
texts have. With ``unit="text"`` every text with a score, a gold value and a covariate value
is a row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus.errors import ConfigurationError
from copernicus.validity._frames import listed

logger = logging.getLogger(__name__)

__all__ = ["UnitRows", "group_runs", "reduce_to_groups", "rows_per_group", "rows_per_text"]

GROUP_COLUMN = "group_id"
VALUE_COLUMNS = ("score", "gold", "covariate")
# The values that every scored text with a finite gold value must have finite under unit
# 'group': the rule that refuses it, the remedy, and what the text unit does with such a text.
REQUIRED_VALUES = (
    (
        "score",
        "run-averaged score is not finite (no run with a score, or an infinite value in a run)",
        "score the text and correct an infinite run score",
        "unit 'text' drops and counts a text without a run with a score, and refuses an "
        "infinite score too",
    ),
    (
        "covariate",
        "covariate value is missing or not finite",
        "give the text a finite covariate value",
        "unit 'text' drops and counts such a text",
    ),
)


@dataclass(frozen=True, eq=False)
class UnitRows:
    """The rows a target is compared on (texts or groups), with the texts behind them.

    Attributes:
        score: Score of every row.
        gold: Gold value of every row.
        covariate: Covariate value of every row.
        clusters: Cluster of every row in the bootstrap.
        used: Ids of the texts that entered the rows.
        dropped: Ids of the scored texts dropped for a missing value.
        noise_runs: The (row, run) scores whose spread across runs is the noise, with the
            columns ``text_id`` (the row), ``run_number`` and ``score``.
    """

    score: np.ndarray
    gold: np.ndarray
    covariate: np.ndarray
    clusters: np.ndarray
    used: pd.Index
    dropped: pd.Index
    noise_runs: pd.DataFrame


def _group_of(texts: pd.Index, groups: pd.Series, prefix: str) -> np.ndarray:
    """Return the group of every text, refusing a text without one.

    A text without a group would be left out of every group's mean without notice, because
    grouping skips a missing key.

    Raises:
        ConfigurationError: If a text is absent from ``groups`` or its group is missing.
    """
    found = groups.reindex(texts)
    absent = sorted({str(text) for text in texts[found.isna().to_numpy()]})
    if absent:
        raise ConfigurationError(
            f"{prefix}text(s) {listed([repr(text) for text in absent])} have no group; with unit "
            "'group' every scored text with a finite gold value enters the mean of its group, so "
            "give each of them a group in the table's group column"
        )
    values: np.ndarray = found.to_numpy()
    return values


def _values_with_texts(rows: pd.DataFrame) -> str:
    """Name each distinct gold value of the rows with the first text that carries it."""
    first_text: dict[float, str] = {}
    for text, value in zip(rows.index, rows["gold"].to_numpy(dtype=float), strict=True):
        first_text.setdefault(float(value), str(text))
    return ", ".join(f"{value!r} on {first_text[value]!r}" for value in sorted(first_text))


def _check_gold(prefix: str, gold_column: str, frame: pd.DataFrame) -> None:
    """Refuse a text without a finite gold value and a group whose texts disagree on it."""
    lacking = frame.index[~np.isfinite(frame["gold"].to_numpy(dtype=float))]
    if len(lacking) > 0:
        raise ConfigurationError(
            f"{prefix}text(s) {listed([repr(text) for text in lacking])} have no finite value of "
            f"gold column {gold_column!r}; leave such texts out before the reduction to groups, "
            "as the panel does when it drops and counts them"
        )
    distinct = frame.groupby(GROUP_COLUMN, sort=True)["gold"].nunique()
    clashing = distinct.index[distinct.to_numpy() > 1]
    if len(clashing) > 0:
        named = [
            f"{group!r} ({_values_with_texts(frame[frame[GROUP_COLUMN] == group])})"
            for group in clashing
        ]
        raise ConfigurationError(
            f"{prefix}with unit 'group' the gold value belongs to the whole group, but the texts "
            f"of group(s) {listed(named)} carry different values of gold column "
            f"{gold_column!r}; give every text of a group the group's value, or compare once per "
            "text with unit 'text'"
        )


def _check_finite(
    prefix: str, frame: pd.DataFrame, column: str, rule: str, remedy: str, by_text: str
) -> None:
    """Refuse a text of a used group whose value in ``column`` is missing or not finite."""
    lacking = ~np.isfinite(frame[column].to_numpy(dtype=float))
    if not lacking.any():
        return
    named = [
        f"{text!r} of group {group!r}"
        for text, group in zip(
            frame.index[lacking], frame[GROUP_COLUMN].to_numpy()[lacking], strict=True
        )
    ]
    raise ConfigurationError(
        f"{prefix}with unit 'group' every scored text with a finite gold value enters the mean "
        f"of its group, so the unit refuses a text whose {rule}: text(s) {listed(named)}; "
        f"{remedy}, or remove the text's gold value to leave it out of its group ({by_text})"
    )


def reduce_to_groups(
    key: str, gold_column: str, aligned: pd.DataFrame, groups: pd.Series
) -> pd.DataFrame:
    """Reduce the texts of one target to one row per group.

    A group's score is the mean over its texts that were scored for the target: the rows of
    ``aligned`` are those texts that have a finite gold value (the panel drops and counts the
    others before this call, so they do not enter their group's mean). The group's covariate
    is the mean of theirs, so every text weighs the same within its group whatever its number
    of runs, and its gold value is the one value all these texts carry. The gold column is
    converted to 64-bit floats once, so the values are compared, reported and returned in that
    form whatever their type; they are checked first (more than one distinct value in a group
    is refused, never averaged), then the scores, then the covariates.

    Args:
        key: Target name, for messages.
        gold_column: Name of the gold column, for messages.
        aligned: One row per text, indexed by string text id, with the columns ``score`` (the
            score averaged over runs), ``gold`` and ``covariate``.
        groups: Group of each text, indexed by string text id.

    Returns:
        One row per group, indexed by group id in sorted order, with the columns ``score``,
        ``gold``, ``covariate`` and ``n_texts`` (the number of texts behind the row).

    Raises:
        ConfigurationError: If a text has no group or no finite gold value, the texts of a group
            carry more than one distinct gold value (the message names the target, the gold
            column and the group), or there is a text whose run-averaged score is not finite (no
            run with a score, or an infinite value in a run) or whose covariate is missing or not
            finite (the message names the target, the group and the text).
    """
    prefix = f"target {key!r}: "
    frame = aligned[list(VALUE_COLUMNS)].assign(
        gold=aligned["gold"].to_numpy(dtype="float64", na_value=np.nan),
        **{GROUP_COLUMN: _group_of(aligned.index, groups, prefix)},
    )
    _check_gold(prefix, gold_column, frame)
    for column, rule, remedy, by_text in REQUIRED_VALUES:
        _check_finite(prefix, frame, column, rule, remedy, by_text)
    grouped = frame.groupby(GROUP_COLUMN, sort=True)
    reduced = pd.DataFrame(
        {
            "score": grouped["score"].mean(),
            "gold": grouped["gold"].first(),
            "covariate": grouped["covariate"].mean(),
            "n_texts": grouped.size(),
        }
    )
    logger.info("target %r: %d texts in %d groups", key, len(frame), len(reduced))
    return reduced


def group_runs(runs: pd.DataFrame, groups: pd.Series, used_texts: pd.Index) -> pd.DataFrame:
    """Return each group's mean score in every run in which all its used texts have a score.

    Under ``unit="group"`` this is the input of ``replicate_noise``, so the noise is the spread
    across runs of the group's mean score. Only the runs in which every used text of the group
    has a finite score enter, so each run's mean is over the same texts; a run that one of them
    lacks (no row, or a missing score) is left out for that group. A group with one such run
    gives no standard deviation, though its mean still counts toward the range of the 0-1
    scale, as a text with one run does under ``unit="text"``; a group with none gives no row.

    Args:
        runs: Frame with ``text_id`` (string), ``run_number`` and ``score``.
        groups: Group of each text, indexed by string text id.
        used_texts: Ids of the texts that entered the target.

    Returns:
        One row per group and run that all its used texts have a score for, sorted by both,
        with the columns ``text_id`` (holding the group id, the column ``replicate_noise``
        reads by default), ``run_number`` and ``score``.

    Raises:
        ConfigurationError: If a text of ``used_texts`` has no group, or a score row of such a
            text has no run number.
    """
    return _group_runs(runs, groups, used_texts, "")


def _check_run_numbers(prefix: str, kept: pd.DataFrame, group_of_rows: np.ndarray) -> None:
    """Refuse a score row of a text that entered the target whose run number is missing."""
    lacking = pd.isna(kept["run_number"]).to_numpy()
    if not lacking.any():
        return
    named = sorted(
        {
            f"{text!r} of group {group!r}"
            for text, group in zip(
                kept["text_id"].to_numpy()[lacking], group_of_rows[lacking], strict=True
            )
        }
    )
    raise ConfigurationError(
        f"{prefix}with unit 'group' the noise compares the runs of a group, but score row(s) of "
        f"text(s) {listed(named)} have no run number; give every score row its run number"
    )


def _group_runs(
    runs: pd.DataFrame, groups: pd.Series, used_texts: pd.Index, prefix: str
) -> pd.DataFrame:
    """`group_runs`, with ``prefix`` (naming the target) at the start of every message."""
    kept = runs[runs["text_id"].isin(used_texts)]
    group_of_rows = _group_of(pd.Index(kept["text_id"]), groups, prefix)
    _check_run_numbers(prefix, kept, group_of_rows)
    scores = kept["score"].to_numpy(dtype="float64", na_value=np.nan)
    finite = np.isfinite(scores)
    frame = pd.DataFrame(
        {
            "text_id": group_of_rows[finite],
            "run_number": kept["run_number"].to_numpy()[finite],
            "score": scores[finite],
            "text": kept["text_id"].to_numpy()[finite],
        }
    )
    by_run = frame.groupby(["text_id", "run_number"], sort=True)
    means = by_run["score"].mean()
    texts_per_group = pd.Series(_group_of(used_texts, groups, prefix)).value_counts()
    needed = texts_per_group.reindex(means.index.get_level_values("text_id")).to_numpy()
    complete = by_run["text"].nunique().to_numpy() == needed
    return means[complete].reset_index()


def _check_group_gold(
    prefix: str, gold_column: str, gold: pd.Series, groups: pd.Series, used: pd.Index
) -> None:
    """Refuse a group that takes part whose texts, scored or not, carry two gold values.

    The texts checked are all texts of the table in a group of the used texts that have a
    finite value of the gold column, whether they were scored for the target or not: the gold
    value belongs to the group, whichever of its texts were scored.
    """
    taking_part = pd.unique(_group_of(used, groups, prefix))
    finite = gold[np.isfinite(gold.to_numpy(dtype=float))]
    of_table = groups.reindex(finite.index)
    member = of_table.isin(taking_part).to_numpy()
    frame = pd.DataFrame(
        {"gold": finite.to_numpy()[member], GROUP_COLUMN: of_table.to_numpy()[member]},
        index=finite.index[member],
    )
    _check_gold(prefix, gold_column, frame)


def rows_per_text(runs: pd.DataFrame, aligned: pd.DataFrame, groups: pd.Series) -> UnitRows:
    """One row per text with a finite score, gold value and covariate; groups are the clusters.

    A text lacking one of the three is dropped and counted, never imputed. `compute_target`
    calls this after it has refused, before it averaged the runs per text, every text with a
    finite gold value whose input rows hold an infinite score or whose score rows lack a run
    number (see `panel_checks`), so such a text lacks at most its run-averaged score or its
    covariate.

    Args:
        runs: Frame with ``text_id`` (string), ``run_number`` and ``score``.
        aligned: One row per scored text, indexed by string text id, with the columns
            ``score`` (averaged over runs), ``gold`` and ``covariate``.
        groups: Group of each text, indexed by string text id.

    Returns:
        The rows, with the run scores of the used texts for the noise.
    """
    complete = np.isfinite(aligned.to_numpy(dtype=float)).all(axis=1)
    used = aligned[complete]
    return UnitRows(
        score=used["score"].to_numpy(),
        gold=used["gold"].to_numpy(),
        covariate=used["covariate"].to_numpy(),
        clusters=groups.reindex(used.index).to_numpy(),
        used=used.index,
        dropped=aligned.index[~complete],
        noise_runs=runs[runs["text_id"].isin(used.index)],
    )


def rows_per_group(
    key: str,
    gold_column: str,
    runs: pd.DataFrame,
    aligned: pd.DataFrame,
    groups: pd.Series,
    gold: pd.Series,
) -> UnitRows:
    """One row per group of the texts with a finite gold value; each group is its own cluster.

    A group's score is the mean over its texts that were scored for the target: a scored text
    without a finite gold value is dropped and counted and stays out of its group's mean, and
    a text never scored for the target does not enter. The gold values of every text of a
    group that takes part, scored or not, are checked first; then the scored texts are
    reduced by `reduce_to_groups`, and the noise is taken from the (group, run) means of
    `group_runs`, over the runs that all the group's used texts have a score for.

    Args:
        key: Target name, for messages.
        gold_column: Name of the gold column, for messages.
        runs: Frame with ``text_id`` (string), ``run_number`` and ``score``.
        aligned: One row per scored text, indexed by string text id, with the columns
            ``score`` (averaged over runs), ``gold`` and ``covariate``.
        groups: Group of each text, indexed by string text id.
        gold: Values of the gold column of every text of the gold frame, indexed by string
            text id, scored or not.

    Returns:
        The rows; ``used`` and ``dropped`` hold text ids.

    Raises:
        ConfigurationError: If the texts of a group that takes part, scored or not, carry two
            gold values, or a used score row has no run number; or as `reduce_to_groups`
            raises it.
    """
    prefix = f"target {key!r}: "
    has_gold = np.isfinite(aligned["gold"].to_numpy(dtype=float))
    used = aligned.index[has_gold]
    _check_group_gold(prefix, gold_column, gold, groups, used)
    reduced = reduce_to_groups(key, gold_column, aligned[has_gold], groups)
    return UnitRows(
        score=reduced["score"].to_numpy(),
        gold=reduced["gold"].to_numpy(),
        covariate=reduced["covariate"].to_numpy(),
        clusters=reduced.index.to_numpy(),
        used=used,
        dropped=aligned.index[~has_gold],
        noise_runs=_group_runs(runs, groups, used, prefix),
    )
