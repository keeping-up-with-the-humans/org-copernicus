"""The validated text table consumed by every coder, embedding, and estimator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from copernicus.data.readers import read_frame
from copernicus.errors import TextTableError

logger = logging.getLogger(__name__)

__all__ = ["GUARANTEED_COLUMNS", "TextTable"]

GUARANTEED_COLUMNS = ("text_id", "text", "group_id", "n_chars", "n_words")
DERIVED_COLUMNS = ("n_chars", "n_words")
GROUP_COLUMN = "group_id"
MAX_IDS_IN_MESSAGE = 10
ONE_GROUP_PER_TEXT = "groups: one per text (no group column)"
"""How the panel's report and the draw's log line name a table without a group column."""
NO_GROUP_WARNING = (
    "no group column: every text is its own group, so a human sample draws the texts "
    "independently and the bootstrap intervals and the PPI standard errors are not clustered; "
    "name the column that holds the groups (for example the writer) with group_id= in Python "
    "or --group-column on the command line"
)


def _normalise(value: str) -> str:
    """Collapse every whitespace run to one space and strip the ends."""
    return " ".join(value.split())


def _listed(values: list[Any]) -> str:
    """Render up to ``MAX_IDS_IN_MESSAGE`` values in ``repr`` form."""
    shown = ", ".join(repr(value) for value in values[:MAX_IDS_IN_MESSAGE])
    rest = len(values) - MAX_IDS_IN_MESSAGE
    return f"{shown}, ... ({rest} more)" if rest > 0 else shown


@dataclass(frozen=True, eq=False)
class TextTable:
    """One row per text, with validated identifiers and normalised text.

    The wrapped frame is read-only by convention; every method returns a new
    object. The first five columns are ``text_id``, ``text``, ``group_id``,
    ``n_chars``, and ``n_words``; any further columns of the input frame follow.

    Attributes:
        frame: The validated data frame with a fresh ``RangeIndex``.
        group_defaulted: True when the input had no group column (none was named
            and the frame had no ``group_id`` column), so ``group_id`` equals
            ``text_id`` and every text is its own group; False when a named or an unnamed
            ``group_id`` column was used.
    """

    frame: pd.DataFrame
    group_defaulted: bool
    _row_of_id: dict[Any, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Index the rows by text id for constant-time lookup."""
        positions = {text_id: row for row, text_id in enumerate(self.frame["text_id"])}
        object.__setattr__(self, "_row_of_id", positions)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        text_id: str = "text_id",
        text: str = "text",
        group_id: str | None = None,
    ) -> TextTable:
        """Validate a data frame and build a text table from it.

        The text is normalised with ``" ".join(text.split())``; ``n_chars`` and
        ``n_words`` describe the normalised text. A frame that already carries
        ``n_chars`` and ``n_words`` (for example a table's own frame written to a
        file) has them recomputed, and when no ``group_id`` argument is given but
        the frame has a ``group_id`` column, that column is used as the grouping.
        Without either, every text becomes its own group and a warning says what that
        means: a human sample draws the texts independently, and the bootstrap intervals of
        the validity panel and the standard errors of the prediction-powered estimators are
        not clustered. All problems found are reported in a single error.

        Args:
            frame: Input frame with one row per text. It is not modified.
            text_id: Name of the column holding the unique text identifiers.
            text: Name of the column holding the texts.
            group_id: Name of the column holding the grouping identifiers (for
                example the author); ``None`` uses a column named ``group_id`` if
                the frame has one and otherwise makes every text its own group, with a
                warning.

        Returns:
            The validated table.

        Raises:
            TextTableError: If a named column is absent, ids are missing or
                duplicated, a text is empty or not a string, a group id is missing,
                or an extra column carries a name reserved for the guaranteed ones.
        """
        if group_id is None and GROUP_COLUMN in frame.columns:
            group_id = GROUP_COLUMN
        source_names = [text_id, text] + ([group_id] if group_id is not None else [])
        absent = [name for name in source_names if name not in frame.columns]
        if absent:
            raise TextTableError(
                f"column(s) {_listed(absent)} not found; available columns: "
                f"{_listed(list(frame.columns))}"
            )

        ids = frame[text_id].astype(object)
        problems = _id_problems(ids)
        texts = frame[text].astype(object)
        problems += _text_problems(texts, ids)
        if group_id is not None:
            groups = frame[group_id].astype(object)
            no_group = groups.isna()
            if no_group.any():
                problems.append(
                    f"missing {group_id!r} values for text ids {_listed(_ids_where(no_group, ids))}"
                )
        reserved = [
            name
            for name in frame.columns
            if name in GUARANTEED_COLUMNS
            and name not in source_names
            and name not in DERIVED_COLUMNS
        ]
        if reserved:
            problems.append(
                f"extra column(s) {_listed(reserved)} collide with guaranteed column names"
            )
        if problems:
            raise TextTableError("invalid text table: " + "; ".join(problems))

        normalised = texts.map(_normalise)
        guaranteed = pd.DataFrame(
            {
                "text_id": ids,
                "text": normalised,
                "group_id": groups if group_id is not None else ids,
                "n_chars": normalised.str.len().astype(int),
                "n_words": normalised.map(lambda value: len(value.split())).astype(int),
            }
        )
        dropped = [*source_names, *(name for name in DERIVED_COLUMNS if name in frame.columns)]
        extras = frame.drop(columns=dropped)
        combined = pd.concat([guaranteed, extras], axis=1).reset_index(drop=True)
        if group_id is None:
            logger.warning(NO_GROUP_WARNING)
        return cls(frame=combined, group_defaulted=group_id is None)

    @classmethod
    def read(cls, path: str | Path, **column_names: str) -> TextTable:
        """Read a CSV, parquet, or JSON-lines file into a validated table.

        In CSV and JSON-lines files the id, text, and group columns are read as text,
        so an id ``007`` stays ``"007"`` and a numeric-looking text stays a text. In a
        CSV file the literal text ``NA`` (or ``null``) in these columns stays a string;
        only an empty field counts as missing there, and a missing id, text, or group
        is then reported by `from_frame`. Every other column keeps pandas' usual
        missing-value strings, and JSON-lines values are never turned into dates.
        Parquet columns keep their stored types.

        Args:
            path: File to read; ``.csv``, ``.parquet``, and ``.jsonl`` are supported.
            **column_names: ``text_id=``, ``text=``, and ``group_id=`` column names,
                forwarded to `from_frame`.

        Returns:
            The validated table.

        Raises:
            TextTableError: If the suffix is unsupported or the contents are invalid.
        """
        protected = [
            column_names.get("text_id", "text_id"),
            column_names.get("text", "text"),
            column_names.get("group_id", GROUP_COLUMN),
        ]
        return cls.from_frame(read_frame(Path(path), protected_columns=protected), **column_names)

    def ids(self) -> list[Any]:
        """Return the text ids in table order."""
        return list(self.frame["text_id"])

    def text_of(self, text_id: Any) -> str:
        """Return the normalised text of one id.

        Args:
            text_id: Identifier of the text.

        Returns:
            The normalised text.

        Raises:
            TextTableError: If the id is not in the table.
        """
        if text_id not in self._row_of_id:
            raise TextTableError(f"unknown text id {text_id!r}")
        return str(self.frame["text"].iloc[self._row_of_id[text_id]])

    def subset(self, ids: Any) -> TextTable:
        """Return a table holding only the given ids, in this table's order.

        Args:
            ids: Iterable of text ids to keep.

        Returns:
            A new table; this one is unchanged.

        Raises:
            TextTableError: If any id is not in the table.
        """
        wanted = list(ids)
        unknown = [text_id for text_id in wanted if text_id not in self._row_of_id]
        if unknown:
            raise TextTableError(f"unknown text id(s) {_listed(unknown)}")
        keep = self.frame["text_id"].isin(wanted)
        return TextTable(
            frame=self.frame[keep].reset_index(drop=True), group_defaulted=self.group_defaulted
        )

    def __len__(self) -> int:
        """Return the number of texts."""
        return len(self.frame)


def _ids_where(mask: pd.Series, ids: pd.Series) -> list[Any]:
    """Return the ids at the rows where ``mask`` is true."""
    return list(ids[mask])


def _id_problems(ids: pd.Series) -> list[str]:
    """Describe missing and duplicated text ids."""
    problems: list[str] = []
    missing = int(ids.isna().sum())
    if missing:
        problems.append(f"missing text ids: {missing} row(s) have no id")
    present = ids[ids.notna()]
    duplicated = list(dict.fromkeys(present[present.duplicated(keep=False)]))
    if duplicated:
        problems.append(f"duplicate text ids: {_listed(duplicated)}")
    return problems


def _text_problems(texts: pd.Series, ids: pd.Series) -> list[str]:
    """Describe empty, missing, and non-string texts."""
    problems: list[str] = []
    missing = texts.isna()
    not_string = ~missing & ~texts.map(lambda value: isinstance(value, str))
    if not_string.any():
        problems.append(f"non-string text for ids {_listed(_ids_where(not_string, ids))}")
    blank = missing | (
        ~not_string & texts.map(lambda value: isinstance(value, str) and not _normalise(value))
    )
    if blank.any():
        problems.append(f"empty text for ids {_listed(_ids_where(blank, ids))}")
    return problems
