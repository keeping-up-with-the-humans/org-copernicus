"""Row lookup, training rows, and the fitted object shared by the score-level families.

The families that choose a penalty on held-out groups (stacking, embedding-augmented
calibration) share ``grouped_training`` and ``labelled_group_shortage``: they need two groups
and draw the seed of their inner splits as the first draw of the fit's generator. Fewer than
two groups make such a family not apply: ``applies`` says so for the labelled rows, and the fit
raises ``NotApplicableError`` for its training rows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.data import CorrectionData, best_score_column
from copernicus.errors import ConfigurationError, NotApplicableError

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_GROUPS",
    "MIN_SCORE_COLUMNS",
    "SEED_BOUND",
    "GroupedTraining",
    "ScoreFitted",
    "TrainingRows",
    "grouped_training",
    "labelled_group_shortage",
    "labelled_positions",
    "require_columns",
    "row_positions",
    "score_column_shortage",
    "training_rows",
]

MIN_GROUPS = 2
MIN_SCORE_COLUMNS = 2
SEED_BOUND = 2**31 - 1

Predictor = Callable[[np.ndarray, np.ndarray], np.ndarray]


def row_positions(data: CorrectionData, ids: np.ndarray) -> np.ndarray:
    """Row positions of the named texts, in the order given.

    Args:
        data: The correction data.
        ids: Text ids.

    Returns:
        Integer positions into the data's frame.

    Raises:
        ConfigurationError: If an id is not in the data.
    """
    wanted = np.asarray(ids).astype(str)
    index = pd.Index(data.ids)
    positions = index.get_indexer(wanted)
    missing = wanted[positions < 0]
    if missing.size:
        raise ConfigurationError(f"text ids not in the data: {list_first_ten(missing.tolist())}")
    return positions


def require_columns(data: CorrectionData, columns: Sequence[str]) -> None:
    """Refuse data that lack columns a fitted correction reads.

    Args:
        data: The correction data.
        columns: Names of the columns the fitted correction needs.

    Raises:
        ConfigurationError: If a column is absent; the message names it.
    """
    missing = [name for name in columns if name not in data.frame.columns]
    if missing:
        raise ConfigurationError(
            f"the data lack the score column(s) {list_first_ten(missing)} that the fitted "
            "correction reads"
        )


@dataclass(frozen=True)
class TrainingRows:
    """The rows a fit learns from, with the best score column chosen on them.

    Attributes:
        column: The score column with the highest signed Spearman correlation with the
            gold values on these rows.
        score: The values of that column.
        covariate: The covariate values.
        gold: The gold values.
    """

    column: str
    score: np.ndarray
    covariate: np.ndarray
    gold: np.ndarray


def labelled_positions(
    data: CorrectionData, train_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row positions and gold values of the training rows.

    Args:
        data: The correction data.
        train_ids: Ids of the labelled rows to learn from.

    Returns:
        The integer positions into the data's frame and the gold values, in the order of
        ``train_ids``.

    Raises:
        ConfigurationError: If an id is repeated, is not in the data, or has no gold value.
    """
    wanted = np.asarray(train_ids).astype(str)
    repeated = pd.Series(wanted)[pd.Series(wanted).duplicated()].unique().tolist()
    if repeated:
        raise ConfigurationError(f"training ids repeat: {list_first_ten(repeated)}")
    positions = row_positions(data, train_ids)
    gold = data.gold[positions]
    unlabelled = np.isnan(gold)
    if unlabelled.any():
        raise ConfigurationError(
            "training ids with no gold value: "
            f"{list_first_ten(np.asarray(train_ids)[unlabelled].tolist())}"
        )
    return positions, gold


def training_rows(data: CorrectionData, train_ids: np.ndarray) -> TrainingRows:
    """Select the training rows and the best score column on them.

    Args:
        data: The correction data.
        train_ids: Ids of the labelled rows to learn from.

    Returns:
        The training rows.

    Raises:
        ConfigurationError: If an id is repeated, is not in the data, or has no gold value.
        EstimatorError: If no score column is usable on these rows.
    """
    positions, gold = labelled_positions(data, train_ids)
    column = best_score_column(data.scores.iloc[positions].reset_index(drop=True), gold)
    return TrainingRows(
        column=column,
        score=data.frame[column].to_numpy(dtype=float)[positions],
        covariate=data.covariate[positions],
        gold=gold,
    )


def _too_few_groups(method: str, n_groups: int, rows: str) -> str:
    """The reason a method that validates on held-out groups cannot run on some rows."""
    return (
        f"{method} chooses its penalty on held-out groups and needs at least two groups "
        f"among the {rows} rows; they hold {n_groups}"
    )


def labelled_group_shortage(data: CorrectionData, method: str) -> str:
    """Why a method that validates on held-out groups does not apply to the data.

    Args:
        data: The correction data.
        method: The method's name, for the message.

    Returns:
        ``""`` when the labelled rows hold at least two groups, otherwise the reason with
        their group count.
    """
    n_groups = int(np.unique(data.groups[data.labelled_mask]).size)
    return "" if n_groups >= MIN_GROUPS else _too_few_groups(method, n_groups, "labelled")


def score_column_shortage(data: CorrectionData, method: str) -> str:
    """Why a method that combines several score columns does not apply to the data.

    Args:
        data: The correction data.
        method: The method's name, for the message.

    Returns:
        ``""`` when the data hold at least two score columns, otherwise the reason with
        their column count.
    """
    if len(data.score_columns) >= MIN_SCORE_COLUMNS:
        return ""
    return f"{method} needs at least two score columns; the data have {len(data.score_columns)}"


@dataclass(frozen=True, eq=False)
class GroupedTraining:
    """The training rows of a fit that validates on held-out groups.

    Attributes:
        positions: Row positions of the training rows in the data's frame.
        gold: Their gold values.
        groups: Their group ids.
        n_groups: The number of distinct groups among them, at least two.
        inner_seed: The seed of the inner splits, the first draw of the fit's generator.
    """

    positions: np.ndarray
    gold: np.ndarray
    groups: np.ndarray
    n_groups: int
    inner_seed: int


def grouped_training(
    data: CorrectionData,
    train_ids: np.ndarray,
    rng: np.random.Generator,
    method: str,
    *,
    label: str,
) -> GroupedTraining:
    """The training rows of a method that validates on held-out groups, and its inner seed.

    Args:
        data: The correction data.
        train_ids: Ids of the labelled rows to learn from.
        rng: The fit's generator; the seed is its first draw, ``rng.integers(SEED_BOUND)``.
        method: The method's name, for the message.
        label: The family and variant, ``"<family>/<variant>"``, that frame a refusal as
            every refusal of ``fit`` is framed.

    Returns:
        The training rows with their groups and the seed of the inner splits.

    Raises:
        ConfigurationError: If an id is repeated, is not in the data, or has no gold value.
        NotApplicableError: If the training rows hold fewer than two groups, the condition
            that ``labelled_group_shortage`` reports for the labelled rows.
    """
    positions, gold = labelled_positions(data, train_ids)
    groups = data.groups[positions]
    n_groups = int(np.unique(groups).size)
    if n_groups < MIN_GROUPS:
        raise NotApplicableError(
            f"{label} does not apply: {_too_few_groups(method, n_groups, 'training')}"
        )
    return GroupedTraining(
        positions=positions,
        gold=gold,
        groups=groups,
        n_groups=n_groups,
        inner_seed=int(rng.integers(SEED_BOUND)),
    )


@dataclass(frozen=True)
class ScoreFitted:
    """A fitted correction that maps one score column (and the covariate) to a prediction.

    Attributes:
        column: The score column the prediction reads.
        predictor: A function of the score values and the covariate values returning the
            corrected values.
        provenance: What was fitted; stored as a read-only mapping.
    """

    column: str
    predictor: Predictor
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze the provenance."""
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        """The corrected values of the named rows.

        Args:
            data: The correction data holding the rows.
            ids: Text ids of the rows.

        Returns:
            Floats on the gold scale indexed by the ids.

        Raises:
            ConfigurationError: If the data lack the fitted score column (the message names
                it) or an id is not in the data.
        """
        require_columns(data, (self.column,))
        positions = row_positions(data, ids)
        score = data.frame[self.column].to_numpy(dtype=float)[positions]
        values = self.predictor(score, data.covariate[positions])
        return pd.Series(
            np.asarray(values, dtype=float), index=pd.Index(np.asarray(ids).astype(str))
        )
