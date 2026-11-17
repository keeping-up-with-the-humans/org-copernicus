"""Score rows of a scored text with a finite gold value that stop the validity panel.

Two kinds of score rows of such a text cannot be read without changing a number of the panel
without notice, so they stop it, whatever the unit, with an error that names the target, the
texts and, where the table has groups, their groups:

* an infinite score, under both units and on every route, also where the construct generated
  from a long attribute table would average ``+inf`` and ``-inf`` of two attributes in one run
  to a missing value. A missing score (NaN) is no such row: the text's other runs carry it, and
  an infinite score of a text without a finite gold value is left to the drop and the count;
* under ``unit="text"``, a row without a run number, which the text unit would average into
  the text's score and count as a run of the noise. The group unit refuses it where it builds
  its noise (`group_runs`), after its own checks of the groups and before any metric.

The panel makes both checks of the text unit before it averages the target's runs per text.
An infinite score is found in the target's rows and in the record that `resolve_targets` keeps
of the raw rows: the construct generated from a long attribute table is averaged earlier, as
the target is built, from a copy in which an infinite score is missing. So the library's error,
and not numpy's warning about an average of ``+inf`` and ``-inf``, reports a broken score.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from copernicus.errors import ConfigurationError
from copernicus.validity._frames import listed

logger = logging.getLogger(__name__)

__all__ = ["check_infinite_scores", "check_run_numbers", "used_texts"]

LEAVE_OUT = (
    "or remove the text's gold value to leave the text out, as the panel drops and counts such "
    "a text"
)


def used_texts(runs: pd.DataFrame, gold: pd.Series) -> pd.Index:
    """The texts of the rows that have a finite gold value, in the order of first appearance.

    Args:
        runs: Frame with ``text_id`` (string), ``run_number`` and ``score``.
        gold: Gold values indexed by string text id.

    Returns:
        The ids of the texts of ``runs`` with a finite gold value.
    """
    texts = pd.Index(pd.unique(runs["text_id"]))
    return texts[np.isfinite(gold.reindex(texts).to_numpy(dtype=float))]


def _named(texts: list[str], groups: pd.Series | None) -> str:
    """The texts, each with its group when the groups are given."""
    if groups is None:
        return listed(repr(text) for text in texts)
    return listed(f"{text!r} of group {groups.loc[text]!r}" for text in texts)


def check_run_numbers(
    prefix: str, runs: pd.DataFrame, used: pd.Index, groups: pd.Series | None
) -> None:
    """Refuse a score row without a run number of a scored text with a finite gold value.

    The text unit makes this check; the group unit refuses such a row where it builds its noise.

    Args:
        prefix: Start of the message, naming the target.
        runs: Frame with ``text_id``, ``run_number`` and ``score``.
        used: Ids of the scored texts with a finite gold value (`used_texts`).
        groups: Group of each text, to name it beside the text, or ``None``.

    Raises:
        ConfigurationError: Naming the texts, if a score row of such a text has no run number.
    """
    lacking = pd.isna(runs["run_number"]).to_numpy() & runs["text_id"].isin(used).to_numpy()
    if not lacking.any():
        return
    texts = sorted(set(runs.loc[lacking, "text_id"]))
    raise ConfigurationError(
        f"{prefix}score row(s) of text(s) {_named(texts, groups)} have no run number, so the "
        "panel cannot tell which of the text's runs they belong to; give every score row its run "
        "number (a frame without a run_number column counts every row as run 1), "
        f"{LEAVE_OUT}"
    )


def check_infinite_scores(
    prefix: str,
    runs: pd.DataFrame,
    infinite: frozenset[str],
    used: pd.Index,
    groups: pd.Series | None,
) -> None:
    """Refuse a scored text with a finite gold value and an infinite score in its target's rows.

    Args:
        prefix: Start of the message, naming the target.
        runs: Frame with ``text_id``, ``run_number`` and ``score``; an infinite score in it
            counts as well.
        infinite: Ids of the texts whose input rows hold an infinite score, also one that the
            rows of ``runs`` no longer show (`resolve_targets` records them).
        used: Ids of the scored texts with a finite gold value (`used_texts`).
        groups: Group of each text, to name it beside the text, or ``None``.

    Raises:
        ConfigurationError: Naming the texts, if such a text has an infinite score.
    """
    scores = runs["score"].to_numpy(dtype="float64", na_value=np.nan)
    visible = set(runs.loc[np.isinf(scores), "text_id"])
    broken = sorted((visible | infinite) & set(used))
    if not broken:
        return
    raise ConfigurationError(
        f"{prefix}text(s) {_named(broken, groups)} carry an infinite score value; an infinite "
        "score is a broken value, not a missing one, so the panel refuses it for a text with a "
        "finite gold value: correct the score (a missing score may stay missing, and the text's "
        f"other runs carry it), {LEAVE_OUT}"
    )
