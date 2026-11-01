"""The folds, the labelled draws and the tables of the correction battery.

A seed and a scheme give the outer partition: grouped folds for ``kfold<k>`` or one grouped
holdout split, from ``rng_for(seed, scheme, "partition")``. In each fold the training pool is
the training texts that carry a gold value, every text being one unit. A size gives the quota
of the pool (a count, or the ceiling of a share of the pool's texts); a quota below 2 or above
half the pool's texts is not drawn. Otherwise whole groups of the pool are drawn by simple
random sampling (``draw_groups``, with a seed taken from ``rng_for(seed, scheme, size, fold,
"draw")``), and the fold's correction data carry the gold values of the drawn texts only, with
the draw's inclusion probability ``m / G`` on every text.

Skip reasons begin with a code: ``holdout_share_outside_band``, ``test_fold_too_few_pairs``,
``test_fold_constant_gold``, ``quota_below_2``, ``quota_exceeds_half_units``,
``constant_training_range``, ``non_finite_predictions``, ``diagnostic_needs_full_gold`` and
``not_applicable`` (followed by the reason the entry's ``applies`` gives); a failure the battery
tolerates is recorded as ``<error class>: <message>``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from copernicus._io import json_ready
from copernicus.baseline.folds import rng_for
from copernicus.corrections.battery_metrics import FoldMetrics, FoldOutcome, aggregate, fold_metrics
from copernicus.corrections.battery_settings import scheme_folds
from copernicus.corrections.crossfit import grouped_folds, holdout_split
from copernicus.corrections.data import PI_COLUMN, CorrectionData, best_score_column
from copernicus.corrections.families._fitted import SEED_BOUND
from copernicus.corrections.sampling import SHARE_ROUNDING_DIGITS, draw_groups
from copernicus.errors import EstimatorError

logger = logging.getLogger(__name__)

__all__ = [
    "HOLDOUT_OUTSIDE_BAND",
    "MIN_TEST_PAIRS",
    "NO_FOLD",
    "QUOTA_BELOW_2",
    "NOT_APPLICABLE",
    "QUOTA_EXCEEDS_HALF",
    "RAW_COLUMNS",
    "RESULT_COLUMNS",
    "SKIPPED_COLUMNS",
    "TEST_FOLD_CONSTANT_GOLD",
    "TEST_FOLD_TOO_FEW_PAIRS",
    "FoldDraw",
    "RawFold",
    "Skip",
    "draw_fold",
    "fold_data",
    "keyed_seed",
    "partition",
    "quota_problem",
    "raw_baseline",
    "result_row",
    "table",
]

HOLDOUT_OUTSIDE_BAND = "holdout_share_outside_band"
TEST_FOLD_TOO_FEW_PAIRS = "test_fold_too_few_pairs"
TEST_FOLD_CONSTANT_GOLD = "test_fold_constant_gold"
NOT_APPLICABLE = "not_applicable"
QUOTA_BELOW_2 = "quota_below_2"
QUOTA_EXCEEDS_HALF = "quota_exceeds_half_units"
MIN_SIZE = 2
MIN_TEST_PAIRS = 3
NO_FOLD = -1
RESULT_COLUMNS = (
    "family",
    "variant",
    "seed",
    "scheme",
    "size",
    "n_labelled",
    "n_folds",
    "n_folds_used",
    "rank_validity_raw",
    "rank_validity_corrected",
    "spearman_raw",
    "spearman_corrected",
    "gain_rank_validity",
    "mae_raw",
    "mae_corrected",
    "gain_mae",
    "mae_native_raw",
    "mae_native_corrected",
    "gain_mae_native",
    "covariate_error_raw",
    "covariate_error_corrected",
    "gain_covariate_error",
    "raw_columns",
    "diagnostic",
    "provenance",
)
"""The columns of ``BatteryResult.results``: one row per entry, seed, scheme and size."""
SKIPPED_COLUMNS = ("family", "variant", "seed", "scheme", "size", "fold", "reason")
"""The columns of ``BatteryResult.skipped``: one row per entry and fold not scored, and why."""
RAW_COLUMNS = (
    "seed",
    "scheme",
    "size",
    "fold",
    "raw_column",
    "n_test",
    "rank_validity_raw",
    "spearman_raw",
    "mae_raw",
    "mae_native_raw",
    "covariate_error_raw",
)
"""The columns of ``BatteryResult.raw``: per scored fold, the raw score column chosen on the
drawn texts and its scores on the held-out texts."""
COLUMN_TYPES = {
    "seed": "int64",
    "size": str,
    "fold": "int64",
    "n_folds": "int64",
    "n_folds_used": "int64",
    "n_test": "int64",
    "n_labelled": float,
    "diagnostic": bool,
}


@dataclass(frozen=True)
class Skip:
    """Why a fit, a fold or a size was not scored.

    Attributes:
        reason: The reason, beginning with its code or the error class.
    """

    reason: str


@dataclass(frozen=True, eq=False)
class FoldDraw:
    """The labelled texts drawn from one fold's training pool.

    Attributes:
        positions: Row positions of the drawn texts in the data, in data order.
        quota: The requested number of texts.
        pool_units: Texts of the training pool (training texts with a gold value).
        pool_groups: Groups of the pool, ``G``.
        m: Drawn groups.
        pi: Inclusion probability ``m / G`` of every drawn text.
        seed: Seed of the group draw.
    """

    positions: np.ndarray
    quota: int
    pool_units: int
    pool_groups: int
    m: int
    pi: float
    seed: int

    @property
    def n_labelled(self) -> int:
        """Number of drawn texts."""
        return int(self.positions.size)

    def record(self) -> dict[str, object]:
        """The draw as plain values for the provenance."""
        return {
            "n_labelled": self.n_labelled,
            "quota": self.quota,
            "pool_units": self.pool_units,
            "pool_groups": self.pool_groups,
            "groups_drawn": self.m,
            "pi": self.pi,
            "draw_seed": self.seed,
        }


@dataclass(frozen=True)
class RawFold:
    """The raw baseline of one fold.

    Attributes:
        column: The score column with the highest signed Spearman correlation with the gold
            values of the drawn texts.
        metrics: Its scores on the held-out texts.
    """

    column: str
    metrics: FoldMetrics


def keyed_seed(*key: object) -> int:
    """The first integer of the generator ``rng_for(*key)``: the seed of a keyed draw.

    Args:
        *key: The parts of the key, such as the seed, the scheme, the size, the fold and
            ``"draw"``.

    Returns:
        An integer in ``[0, 2**31 - 1)`` that the same key always gives.
    """
    return int(rng_for(*key).integers(SEED_BOUND))


def partition(
    data: CorrectionData, scheme: str, seed: int
) -> list[tuple[np.ndarray, np.ndarray]] | Skip:
    """The outer folds of a scheme as ``(train_pos, test_pos)`` pairs, or why there are none.

    Args:
        data: The correction data; every text is placed by its group.
        scheme: ``kfold<k>`` or ``holdout``.
        seed: The battery seed.

    Returns:
        The folds, or a skip with ``holdout_share_outside_band`` when no whole-group holdout
        split lands in the band.
    """
    rng = rng_for(seed, scheme, "partition")
    k = scheme_folds(scheme)
    if k is not None:
        return grouped_folds(data.groups, k, rng)
    # The one partition failure the battery tolerates: no holdout split in the band.
    try:
        return [holdout_split(data.groups, rng)]
    except EstimatorError as error:
        return Skip(f"{HOLDOUT_OUTSIDE_BAND}: {error}")


def quota_problem(quota: int, units: int) -> str:
    """Why a quota cannot be drawn from a pool, or ``""`` when it can.

    Args:
        quota: The requested number of texts.
        units: The texts of the pool.

    Returns:
        ``quota_below_2: ...``, ``quota_exceeds_half_units: ...`` or ``""``.
    """
    if quota < MIN_SIZE:
        return f"{QUOTA_BELOW_2}: the quota of {quota} texts is below 2 ({units} texts in the pool)"
    if 2 * quota > units:
        return (
            f"{QUOTA_EXCEEDS_HALF}: the quota of {quota} texts exceeds half of the {units} texts "
            "in the pool"
        )
    return ""


def draw_fold(
    data: CorrectionData, train_pos: np.ndarray, size: int | float, key: Sequence[object]
) -> FoldDraw | Skip:
    """Draw whole groups of a fold's training pool for the quota of a size.

    Args:
        data: The user's correction data.
        train_pos: Row positions of the fold's training texts.
        size: A count of texts, or a share of the pool's texts.
        key: ``(seed, scheme, size label, fold)``, the key of the draw's generator.

    Returns:
        The draw, or a skip when the quota is below 2 or above half the pool.
    """
    pool = train_pos[data.labelled_mask[train_pos]]
    units = int(pool.size)
    if isinstance(size, float):
        quota = math.ceil(round(size * units, SHARE_ROUNDING_DIGITS))
    else:
        quota = int(size)
    problem = quota_problem(quota, units)
    if problem:
        return Skip(problem)
    seed = keyed_seed(*key, "draw")
    ids = data.ids[pool]
    # The pool is a part of the data, so its draw records the pool's own fingerprint.
    draw = draw_groups(ids, data.groups[pool], ids, quota, seed=seed)
    return FoldDraw(
        positions=pool[np.isin(ids, draw.ids)],
        quota=quota,
        pool_units=units,
        pool_groups=draw.n_groups,
        m=draw.m,
        pi=draw.pi,
        seed=seed,
    )


def fold_data(data: CorrectionData, draw: FoldDraw) -> CorrectionData:
    """The data a fold's corrections see: gold on the drawn texts only, ``pi`` on every text.

    Args:
        data: The user's correction data.
        draw: The fold's draw.

    Returns:
        New correction data with the same texts, scores, covariate and embeddings.
    """
    drawn = np.zeros(data.n, dtype=bool)
    drawn[draw.positions] = True
    frame = data.frame.copy()
    frame[data.gold_column] = np.where(drawn, data.gold, np.nan)
    frame[PI_COLUMN] = draw.pi
    return replace(data, frame=frame)


def raw_baseline(
    data: CorrectionData, draw: FoldDraw, eval_pos: np.ndarray, scale: str
) -> RawFold | Skip:
    """The best raw score column on the drawn texts, scored on the held-out texts.

    Args:
        data: The user's correction data.
        draw: The fold's draw.
        eval_pos: Row positions of the held-out texts with a gold value.
        scale: The scale of the covariate slope.

    Returns:
        The raw baseline, or a skip naming the error when no column is usable on the drawn
        texts (every column constant there, or constant gold).
    """
    gold = data.gold
    # Tolerated like a failing fit: no usable raw column leaves the fold without a baseline.
    try:
        column = best_score_column(
            data.scores.iloc[draw.positions].reset_index(drop=True), gold[draw.positions]
        )
    except EstimatorError as error:
        return Skip(f"{type(error).__name__}: {error}")
    values = data.frame[column].to_numpy(dtype=float)
    metrics = fold_metrics(
        values[draw.positions],
        gold[draw.positions],
        values[eval_pos],
        gold[eval_pos],
        data.covariate[eval_pos],
        scale,
    )
    return RawFold(column=column, metrics=metrics)


def result_row(
    family: str, variant: str, identity: dict[str, object], outcomes: Sequence[FoldOutcome]
) -> dict[str, object]:
    """One results row: an entry's fold outcomes at one seed, scheme and size.

    Args:
        family: The entry's family.
        variant: The entry's label.
        identity: ``seed``, ``scheme`` and ``size`` (its label).
        outcomes: The outcomes of the folds where the entry was scored, at least one.

    Returns:
        The row with every column of ``RESULT_COLUMNS``.
    """
    return {
        "family": family,
        "variant": variant,
        **identity,
        "n_labelled": float(np.mean([outcome.n_labelled for outcome in outcomes])),
        "n_folds": len(outcomes),
        **aggregate(outcomes),
        "raw_columns": ";".join(outcome.raw_column for outcome in outcomes),
        "diagnostic": False,
        "provenance": json.dumps(
            json_ready([outcome.provenance for outcome in outcomes]), default=str
        ),
    }


def table(
    rows: Sequence[tuple[tuple[int, ...], dict[str, object]]], columns: Sequence[str]
) -> pd.DataFrame:
    """A table of rows sorted by their order keys, with one type per column.

    Args:
        rows: ``(order key, row)`` pairs.
        columns: The columns, in order.

    Returns:
        The table with a fresh index.
    """
    ordered = [row for _, row in sorted(rows, key=lambda pair: pair[0])]
    frame = pd.DataFrame(ordered, columns=list(columns))
    return frame.astype({name: kind for name, kind in COLUMN_TYPES.items() if name in columns})
