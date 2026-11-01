"""The data every correction works on: texts, groups, scores, partial gold, a covariate.

``CorrectionData`` holds one row per text with one or more machine score columns, a gold
column that is NaN where a text has not been coded by a human, a covariate (the log of the
text length by default), the grouping, and optionally the inclusion probability of each
labelled text. The numeric helpers that the families and the battery share (the best raw
score column on a set of labelled rows, scaling with the range of the training values) live
in ``scaling`` and are re-exported here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from copernicus.corrections._fingerprint import population_hash_of
from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.scaling import best_score_column, minmax_apply, minmax_fit, scale_test
from copernicus.errors import ConfigurationError
from copernicus.validity.metrics import is_binary

if TYPE_CHECKING:
    from copernicus.data import TextTable
    from copernicus.embeddings.sets import EmbeddingSet

logger = logging.getLogger(__name__)

__all__ = [
    "CorrectionData",
    "best_score_column",
    "minmax_apply",
    "minmax_fit",
    "scale_test",
]

LOG_CHARS = "log_chars"
PI_COLUMN = "pi"
RESERVED_COLUMNS = ("text_id", "group_id", PI_COLUMN)


def numeric_values(column: pd.Series, description: str) -> np.ndarray:
    """The values of a series as floats, refusing text and other non-numeric data.

    Args:
        column: The series.
        description: How the series is named in the message, such as ``gold column 'gold'``.

    Returns:
        A float array with NaN where the series has a missing value.

    Raises:
        ConfigurationError: If the dtype is not numeric.
    """
    if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_complex_dtype(column):
        raise ConfigurationError(f"{description} must be numeric, got dtype {column.dtype}")
    return column.to_numpy(dtype=float, na_value=np.nan)


def _numeric_column(frame: pd.DataFrame, name: str, role: str) -> np.ndarray:
    """The named column as floats, refusing non-numeric data."""
    return numeric_values(frame[name], f"{role} column {name!r}")


def _check_role_names(score_columns: Sequence[str], gold_column: str, covariate: str) -> None:
    """Refuse reserved names and repeated names among the score, gold and covariate columns."""
    roles = [*score_columns, gold_column, covariate]
    reserved = sorted({name for name in roles if name in RESERVED_COLUMNS})
    if reserved:
        raise ConfigurationError(
            f"{list_first_ten(reserved)} cannot be a score, gold or covariate column: "
            f"{', '.join(RESERVED_COLUMNS)} are reserved for the identifiers and the "
            "inclusion probability"
        )
    repeated = sorted({name for name in roles if roles.count(name) > 1})
    if repeated:
        raise ConfigurationError(
            f"score_columns, gold_column and covariate_column must be distinct; "
            f"repeated: {list_first_ten(repeated)}"
        )


def _checked_extras(extra_columns: Sequence[str], roles: Sequence[str]) -> tuple[str, ...]:
    """The extra columns to carry, refusing repeated names and names the frame already uses."""
    if isinstance(extra_columns, str):
        raise ConfigurationError("extra_columns must be a sequence of column names")
    extras = tuple(extra_columns)
    taken = sorted({name for name in extras if name in RESERVED_COLUMNS or name in roles})
    if taken:
        raise ConfigurationError(
            f"extra columns {list_first_ten(taken)} take a name the frame already uses for the "
            "identifiers, the inclusion probability, a score, the gold or the covariate"
        )
    repeated = sorted({name for name in extras if extras.count(name) > 1})
    if repeated:
        raise ConfigurationError(f"extra columns repeat: {list_first_ten(repeated)}")
    return extras


def _extra_values(
    name: str, ids: pd.Series, by_table: pd.DataFrame, by_gold: pd.DataFrame
) -> np.ndarray:
    """An extra column per scored text, from the table's frame or else from gold."""
    if name in by_table.columns:
        return by_table.loc[ids, name].to_numpy()
    if name in by_gold.columns:
        return by_gold[name].reindex(ids).to_numpy()
    raise ConfigurationError(f"extra column {name!r} is neither a column of the table nor of gold")


def _require_finite(frame: pd.DataFrame, name: str, role: str) -> np.ndarray:
    """A numeric column that must be finite on every row, with the offending ids named."""
    values = _numeric_column(frame, name, role)
    bad = ~np.isfinite(values)
    if bad.any():
        raise ConfigurationError(
            f"{role} column {name!r} must be finite on every row; non-finite for text ids "
            f"{list_first_ten(frame.loc[bad, 'text_id'].tolist())}"
        )
    return values


@dataclass(frozen=True, eq=False)
class CorrectionData:
    """Texts with machine scores and, for a subset, human gold values.

    The frame is copied on construction: ``text_id`` and ``group_id`` become strings, the
    score, gold, covariate and ``pi`` columns become floats, and the index is reset.

    Attributes:
        frame: One row per text with the columns ``text_id`` (unique), ``group_id``, every
            score column (finite), ``gold_column`` (NaN where a text has no human value),
            ``covariate_column`` (finite) and, optionally, ``pi`` (the inclusion
            probability of the text in the human sample, in ``(0, 1]``; NaN is allowed
            where the text is unlabelled). Further columns are kept.
        score_columns: Names of the machine score columns, at least one.
        gold_column: Name of the gold column.
        covariate_column: Name of the covariate column (default ``log_chars``).
        embeddings: Embedding sets by encoder name, or ``None``.
    """

    frame: pd.DataFrame
    score_columns: tuple[str, ...]
    gold_column: str = "gold"
    covariate_column: str = LOG_CHARS
    embeddings: Mapping[str, EmbeddingSet] | None = None

    def __post_init__(self) -> None:
        """Validate the frame and store a cleaned copy.

        Raises:
            ConfigurationError: If a required column is missing, a score, gold or covariate
                column is named ``text_id``, ``group_id`` or ``pi``, text ids are missing or
                repeated, a group id is missing, a score or the covariate is not finite,
                the gold column is not numeric, or a ``pi`` value lies outside ``(0, 1]``.
        """
        if not isinstance(self.frame, pd.DataFrame):
            raise ConfigurationError("frame must be a pandas DataFrame")
        if isinstance(self.score_columns, str):
            raise ConfigurationError("score_columns must be a sequence of column names")
        scores = tuple(self.score_columns)
        if not scores:
            raise ConfigurationError("score_columns must name at least one score column")
        _check_role_names(scores, self.gold_column, self.covariate_column)
        required = ["text_id", "group_id", *scores, self.gold_column, self.covariate_column]
        missing = [name for name in required if name not in self.frame.columns]
        if missing:
            raise ConfigurationError(f"missing required column(s) {list_first_ten(missing)}")
        if len(self.frame) == 0:
            raise ConfigurationError("the frame has no rows")

        frame = self.frame.reset_index(drop=True).copy()
        absent = frame["text_id"].isna() | frame["group_id"].isna()
        if absent.any():
            raise ConfigurationError(
                f"text_id and group_id must be present on every row; missing at rows "
                f"{list_first_ten(frame.index[absent].tolist())}"
            )
        frame["text_id"] = frame["text_id"].astype(str)
        frame["group_id"] = frame["group_id"].astype(str)
        repeats = frame["text_id"][frame["text_id"].duplicated()].unique().tolist()
        if repeats:
            raise ConfigurationError(f"text_id must be unique; repeated: {list_first_ten(repeats)}")
        for name in scores:
            frame[name] = _require_finite(frame, name, "score")
        frame[self.covariate_column] = _require_finite(frame, self.covariate_column, "covariate")
        gold = _numeric_column(frame, self.gold_column, "gold")
        infinite = np.isinf(gold)
        if infinite.any():
            raise ConfigurationError(
                f"gold column {self.gold_column!r} must be finite or NaN; infinite for text "
                f"ids {list_first_ten(frame.loc[infinite, 'text_id'].tolist())}"
            )
        frame[self.gold_column] = gold
        if PI_COLUMN in frame.columns:
            frame[PI_COLUMN] = self._checked_pi(frame)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "score_columns", scores)

    @staticmethod
    def _checked_pi(frame: pd.DataFrame) -> np.ndarray:
        """The ``pi`` column as floats, each value NaN or in ``(0, 1]``."""
        pi = _numeric_column(frame, PI_COLUMN, "inclusion probability")
        bad = ~np.isnan(pi) & ~((pi > 0.0) & (pi <= 1.0))
        if bad.any():
            raise ConfigurationError(
                f"inclusion probability column {PI_COLUMN!r} must lie in (0, 1] or be NaN; "
                f"offending text ids {list_first_ten(frame.loc[bad, 'text_id'].tolist())}"
            )
        return pi

    @property
    def ids(self) -> np.ndarray:
        """Text ids as a string array, in row order."""
        return self.frame["text_id"].to_numpy(dtype=str)

    @property
    def n(self) -> int:
        """Number of texts."""
        return len(self.frame)

    @property
    def labelled_mask(self) -> np.ndarray:
        """Boolean array: True where the gold value is present."""
        return ~np.isnan(self.gold)

    @property
    def labelled_ids(self) -> np.ndarray:
        """Ids of the labelled texts, in row order."""
        return self.ids[self.labelled_mask]

    @property
    def n_labelled(self) -> int:
        """Number of labelled texts."""
        return int(self.labelled_mask.sum())

    @property
    def groups(self) -> np.ndarray:
        """Group ids as a string array, in row order."""
        return self.frame["group_id"].to_numpy(dtype=str)

    @cached_property
    def population_hash(self) -> str:
        """The fingerprint ``population_hash_of(ids, groups)`` of the texts and their groups.

        It is computed once per object, so draws of the data and their attachment share it.
        The frame is not changed in place: other texts or groups make new data (for example
        with ``dataclasses.replace``), whose fingerprint is computed anew.
        """
        return population_hash_of(self.ids, self.groups)

    @property
    def gold(self) -> np.ndarray:
        """Gold values as floats, NaN where unlabelled."""
        return self.frame[self.gold_column].to_numpy(dtype=float)

    @property
    def scores(self) -> pd.DataFrame:
        """The score columns as a frame with a fresh index."""
        return self.frame[list(self.score_columns)].copy()

    @property
    def covariate(self) -> np.ndarray:
        """Covariate values as floats."""
        return self.frame[self.covariate_column].to_numpy(dtype=float)

    @property
    def pi(self) -> np.ndarray | None:
        """Inclusion probabilities (NaN where unlabelled), or ``None`` without a ``pi`` column."""
        if PI_COLUMN not in self.frame.columns:
            return None
        return self.frame[PI_COLUMN].to_numpy(dtype=float)

    @property
    def is_binary_gold(self) -> bool:
        """Whether the labelled gold values take exactly two distinct values."""
        return is_binary(self.gold)

    @property
    def gold_classes(self) -> np.ndarray:
        """The sorted distinct labelled gold values."""
        return np.unique(self.gold[self.labelled_mask])

    @classmethod
    def from_scores(
        cls,
        scores: pd.DataFrame,
        table: TextTable,
        gold: pd.DataFrame,
        *,
        gold_column: str,
        score_columns: Sequence[str],
        covariate: str = LOG_CHARS,
        embeddings: Mapping[str, EmbeddingSet] | None = None,
        pi_column: str | None = None,
        extra_columns: Sequence[str] = (),
    ) -> CorrectionData:
        """Join wide machine scores, the text table and human gold values by text id.

        Ids are compared as strings. The rows are the scored texts, in the order of
        ``scores``. The group comes from the table. A text without a row in ``gold``, or
        with a NaN value, is unlabelled.

        Args:
            scores: A wide frame with ``text_id`` and the named score columns, such as the
                text scores that ``export_text_scores`` returns.
            table: The text table the scores were made for.
            gold: A frame with ``text_id`` and ``gold_column`` (it may hold more texts or
                columns than needed, so the table's own frame works).
            gold_column: Name of the gold column in ``gold``; it keeps this name.
            score_columns: Names of the score columns to use.
            covariate: ``"log_chars"`` for the log of the character count, or the name of
                a numeric column of the table's frame or of ``gold``.
            embeddings: Embedding sets by encoder name, attached as they are.
            pi_column: A numeric column of ``gold`` with the inclusion probability of each
                coded text (such as the ``pi`` of ``draw_human_sample``), stored as ``pi``;
                NaN for a text without a row in ``gold``. ``None`` adds no ``pi``.
            extra_columns: Further columns to carry into the frame, such as the outcome or
                the covariates of a regression, each taken as it is from the table's frame or
                else from ``gold`` (NaN for a text without a row there).

        Returns:
            The correction data.

        Raises:
            ConfigurationError: If a column is absent, a score, gold or covariate column is
                named ``text_id``, ``group_id`` or ``pi`` or two of them coincide, a scored
                text is not in the table, ``gold`` repeats a text id, the covariate cannot be
                found, ``pi_column`` is the gold, a score or the covariate column, or an extra
                column repeats or takes a name the frame already uses.
        """
        score_names = tuple(score_columns)
        _check_role_names(score_names, gold_column, covariate)
        extras = _checked_extras(extra_columns, (*score_names, gold_column, covariate))
        if pi_column is not None and pi_column in (*score_names, gold_column, covariate):
            raise ConfigurationError(
                f"pi_column {pi_column!r} must name a column of its own, not the gold, a score "
                "or the covariate column"
            )
        missing = [name for name in ("text_id", *score_names) if name not in scores.columns]
        if missing:
            raise ConfigurationError(f"scores lack column(s) {list_first_ten(missing)}")
        needed = ("text_id", gold_column, *([] if pi_column is None else [pi_column]))
        missing = [name for name in needed if name not in gold.columns]
        if missing:
            raise ConfigurationError(f"gold lacks column(s) {list_first_ten(missing)}")

        ids = scores["text_id"].astype(str).reset_index(drop=True)
        table_ids = table.frame["text_id"].astype(str)
        by_table = table.frame.set_index(table_ids.to_numpy())
        absent = ids[~ids.isin(by_table.index)].tolist()
        if absent:
            raise ConfigurationError(f"scored texts not in the table: {list_first_ten(absent)}")
        gold_ids = gold["text_id"].astype(str)
        repeated = gold_ids[gold_ids.duplicated()].unique().tolist()
        if repeated:
            raise ConfigurationError(f"gold repeats text ids {list_first_ten(repeated)}")
        by_gold = gold.set_index(gold_ids.to_numpy())

        frame = scores[list(score_names)].reset_index(drop=True).copy()
        frame.insert(0, "text_id", ids)
        frame.insert(1, "group_id", by_table.loc[ids, "group_id"].to_numpy())
        coded = pd.Series(_numeric_column(by_gold, gold_column, "gold"), index=by_gold.index)
        frame[gold_column] = coded.reindex(ids).to_numpy(dtype=float)
        frame[covariate] = cls._covariate_values(covariate, ids, by_table, by_gold)
        if pi_column is not None:
            values = _numeric_column(by_gold, pi_column, "inclusion probability")
            frame[PI_COLUMN] = pd.Series(values, index=by_gold.index).reindex(ids).to_numpy()
        for name in extras:
            frame[name] = _extra_values(name, ids, by_table, by_gold)
        return cls(
            frame=frame,
            score_columns=score_names,
            gold_column=gold_column,
            covariate_column=covariate,
            embeddings=embeddings,
        )

    @staticmethod
    def _covariate_values(
        covariate: str, ids: pd.Series, by_table: pd.DataFrame, by_gold: pd.DataFrame
    ) -> np.ndarray:
        """Covariate per scored text: ``log(n_chars)``, or a named table or gold column."""
        if covariate == LOG_CHARS:
            return np.log(by_table.loc[ids, "n_chars"].to_numpy(dtype=float))
        if covariate in by_table.columns:
            return _numeric_column(by_table.loc[ids], covariate, "covariate")
        if covariate in by_gold.columns:
            return _numeric_column(by_gold.reindex(ids), covariate, "covariate")
        raise ConfigurationError(
            f"covariate {covariate!r} is neither 'log_chars' nor a column of the table or gold"
        )
