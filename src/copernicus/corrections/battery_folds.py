"""One fold of the correction battery: draw, fit, predict, score and record.

Every entry that applies to the fold's data is fitted on the drawn texts with the generator
``rng_for(seed, scheme, size, fold, family)``: keyed by the family, not the variant, so the
variants of one family share their inner draws (the deletions of embedding-augmented
calibration, the inner splits of stacking). It predicts the drawn and the held-out texts, and
both are scored against the user's gold values. An entry whose ``applies`` refuses the fold's
data is recorded as ``not_applicable: <reason>``; only ``EstimatorError`` and
``NotApplicableError`` from the fit or the predictions are recorded as skip rows; any other
exception stops the battery. After the per-text entries, each variant of the correction
ensemble pools their held-out predictions (keyed ``family:label``), with the gold classes of
the drawn texts when they take at most five values. A fold whose held-out gold values are all
the same has no rank validity for any entry and is skipped as ``test_fold_constant_gold``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from copernicus.baseline.folds import rng_for
from copernicus.corrections.battery_metrics import (
    FoldMetrics,
    FoldOutcome,
    dropped_metrics,
    fold_metrics,
    prediction_problem,
    rank_metrics,
)
from copernicus.corrections.battery_rows import (
    MIN_TEST_PAIRS,
    NOT_APPLICABLE,
    TEST_FOLD_CONSTANT_GOLD,
    TEST_FOLD_TOO_FEW_PAIRS,
    FoldDraw,
    RawFold,
    Skip,
    draw_fold,
    fold_data,
    raw_baseline,
    result_row,
)
from copernicus.corrections.battery_settings import BatteryEntry, BatterySettings, size_label
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families import CorrectionEnsemble, ScoreCorrection
from copernicus.errors import EstimatorError, NotApplicableError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["MODE_MAX_CLASSES", "Order", "Point", "Row", "Tables", "run_fold"]

MODE_MAX_CLASSES = 5

Order = tuple[int, ...]
Row = tuple[Order, dict[str, object]]


@dataclass(frozen=True)
class Point:
    """One seed, scheme and size of the battery, with their positions in the settings.

    Attributes:
        seed: The seed.
        scheme: The scheme.
        size: The size, a count or a share.
        order: The positions of the seed, the scheme and the size in the settings.
    """

    seed: int
    scheme: str
    size: int | float
    order: Order

    @property
    def label(self) -> str:
        """The size's label."""
        return size_label(self.size)

    @property
    def identity(self) -> dict[str, object]:
        """The ``seed``, ``scheme`` and ``size`` columns of a row."""
        return {"seed": self.seed, "scheme": self.scheme, "size": self.label}


@dataclass(frozen=True, eq=False)
class _Fold:
    """What the entries of one fold share: the draw, the fold's data and the raw baseline."""

    index: int
    data: CorrectionData
    labelled: CorrectionData
    draw: FoldDraw
    raw: RawFold
    test_pos: np.ndarray
    eval_pos: np.ndarray

    @cached_property
    def train_ids(self) -> np.ndarray:
        """Ids of the drawn texts."""
        return self.data.ids[self.draw.positions]

    @cached_property
    def test_ids(self) -> np.ndarray:
        """Ids of the held-out texts."""
        return self.data.ids[self.test_pos]

    @cached_property
    def scored(self) -> np.ndarray:
        """Which held-out texts carry a gold value, as a mask over ``test_pos``."""
        return self.data.labelled_mask[self.test_pos]

    def outcome(self, corrected: FoldMetrics, fit: dict[str, object]) -> FoldOutcome:
        """An entry's outcome on this fold, with the fold's provenance."""
        raw = self.raw.metrics
        provenance = {
            "fold": self.index,
            "n_test": corrected.n_test,
            **self.draw.record(),
            "raw_column": self.raw.column,
            "rank_validity_raw": raw.rank_validity,
            "rank_validity_corrected": corrected.rank_validity,
            "spearman_raw": raw.spearman,
            "spearman_corrected": corrected.spearman,
            "mae_raw": raw.mae,
            "mae_corrected": corrected.mae,
            "mae_native_raw": raw.mae_native,
            "mae_native_corrected": corrected.mae_native,
            "covariate_error_raw": raw.covariate_error,
            "covariate_error_corrected": corrected.covariate_error,
            "status": corrected.status,
            "status_raw": raw.status,
            "dropped_from": dropped_metrics(raw, corrected),
            "fit": fit,
        }
        return FoldOutcome(
            fold=self.index,
            n_labelled=self.draw.n_labelled,
            raw_column=self.raw.column,
            raw=raw,
            corrected=corrected,
            provenance=provenance,
        )


def _check_index(correction: ScoreCorrection, fold: int, asked: np.ndarray, got: pd.Series) -> None:
    """Refuse predictions that are not indexed by the ids asked for, in their order."""
    if got.index.astype(str).tolist() != asked.tolist():
        raise EstimatorError(
            f"{correction.family}/{correction.variant}: the predictions of fold {fold} are not "
            "indexed by the ids asked for"
        )


def _fit_entry(
    correction: ScoreCorrection, fold: _Fold, rng: np.random.Generator, scale: str
) -> tuple[FoldOutcome, pd.Series] | Skip:
    """Fit one entry on the drawn texts, predict the held-out texts and score both."""
    applies, reason = correction.applies(fold.labelled)
    if not applies:
        return Skip(f"{NOT_APPLICABLE}: {reason}")
    # The failures the battery records instead of stopping: an undefined fit or prediction.
    try:
        fitted = correction.fit(fold.labelled, fold.train_ids, rng=rng)
        train = fitted.predict(fold.labelled, fold.train_ids)
        test = fitted.predict(fold.labelled, fold.test_ids)
    except (EstimatorError, NotApplicableError) as error:
        return Skip(f"{type(error).__name__}: {error}")
    _check_index(correction, fold.index, fold.train_ids, train)
    _check_index(correction, fold.index, fold.test_ids, test)
    train_values, test_values = train.to_numpy(dtype=float), test.to_numpy(dtype=float)
    problem = prediction_problem(train_values, test_values)
    if problem:
        return Skip(problem)
    corrected = fold_metrics(
        train_values,
        fold.data.gold[fold.draw.positions],
        test_values[fold.scored],
        fold.data.gold[fold.eval_pos],
        fold.data.covariate[fold.eval_pos],
        scale,
    )
    return fold.outcome(corrected, dict(fitted.provenance)), test


def _combine_entry(
    entry: BatteryEntry, members: dict[str, pd.Series], fold: _Fold
) -> FoldOutcome | Skip:
    """Pool the held-out predictions of the fold's fitted entries with one ensemble variant."""
    classes = fold.labelled.gold_classes
    gold_classes = classes if classes.size <= MODE_MAX_CLASSES else None
    # Fewer than two usable members, or mode without gold classes, is recorded as a skip.
    try:
        combined = CorrectionEnsemble(entry.variant).combine(
            members, fold.test_ids, gold_classes=gold_classes
        )
    except (EstimatorError, NotApplicableError) as error:
        return Skip(f"{type(error).__name__}: {error}")
    values = combined.column.to_numpy(dtype=float)
    corrected = rank_metrics(values[fold.scored], fold.data.gold[fold.eval_pos])
    return fold.outcome(corrected, dict(combined.provenance))


class Tables:
    """The rows of the three tables, and the fold outcomes of the current seed, scheme and size.

    Every row is kept with an order key (entry, seed, scheme, size, fold positions), so the
    tables can be sorted in the order of the settings at the end.
    """

    def __init__(self, settings: BatterySettings) -> None:
        """Start empty tables for a run with these settings."""
        self.settings = settings
        self.positions = {entry: index for index, entry in enumerate(settings.entries)}
        self.results: list[Row] = []
        self.skipped: list[Row] = []
        self.raw: list[Row] = []
        self.outcomes: dict[BatteryEntry, list[FoldOutcome]] = {}

    def skip(self, entry: BatteryEntry, point: Point, fold: int, reason: str) -> None:
        """Record that an entry was not scored on a fold (or, with fold -1, at a point)."""
        row = {"family": entry.family, "variant": entry.label, **point.identity}
        order = (self.positions[entry], *point.order, fold)
        self.skipped.append((order, {**row, "fold": fold, "reason": reason}))

    def skip_all(self, point: Point, fold: int, reason: str) -> None:
        """Record the same reason for every entry."""
        for entry in self.settings.entries:
            self.skip(entry, point, fold, reason)

    def add_raw(self, point: Point, fold: _Fold) -> None:
        """Record the raw baseline of a scored fold."""
        metrics = fold.raw.metrics
        row = {
            **point.identity,
            "fold": fold.index,
            "raw_column": fold.raw.column,
            "n_test": metrics.n_test,
            "rank_validity_raw": metrics.rank_validity,
            "spearman_raw": metrics.spearman,
            "mae_raw": metrics.mae,
            "mae_native_raw": metrics.mae_native,
            "covariate_error_raw": metrics.covariate_error,
        }
        self.raw.append(((*point.order, fold.index), row))

    def close(self, point: Point) -> None:
        """Turn the fold outcomes of a seed, scheme and size into results rows."""
        for entry, outcomes in self.outcomes.items():
            row = result_row(entry.family, entry.label, point.identity, outcomes)
            self.results.append(((self.positions[entry], *point.order), row))
        self.outcomes = {}


def run_fold(
    data: CorrectionData,
    corrections: dict[BatteryEntry, ScoreCorrection],
    tables: Tables,
    point: Point,
    index: int,
    split: tuple[np.ndarray, np.ndarray],
) -> None:
    """Draw, fit, predict and score every entry on one fold, recording rows in ``tables``.

    Args:
        data: The user's correction data.
        corrections: The built correction of every per-text entry, in the order of the
            settings.
        tables: The tables of the run.
        point: The seed, scheme and size.
        index: The fold index.
        split: The fold's ``(train_pos, test_pos)``.
    """
    train_pos, test_pos = split
    eval_pos = test_pos[data.labelled_mask[test_pos]]
    if eval_pos.size < MIN_TEST_PAIRS:
        tables.skip_all(
            point,
            index,
            f"{TEST_FOLD_TOO_FEW_PAIRS}: {eval_pos.size} held-out texts carry a gold value; at "
            f"least {MIN_TEST_PAIRS} are needed",
        )
        return
    held_out = data.gold[eval_pos]
    if is_constant(held_out):
        tables.skip_all(
            point,
            index,
            f"{TEST_FOLD_CONSTANT_GOLD}: the {eval_pos.size} held-out texts with a gold value all "
            f"have the value {float(held_out[0])!r}, so no prediction can be ranked against them",
        )
        return
    draw = draw_fold(data, train_pos, point.size, (point.seed, point.scheme, point.label, index))
    if isinstance(draw, Skip):
        tables.skip_all(point, index, draw.reason)
        return
    raw = raw_baseline(data, draw, eval_pos, tables.settings.covariate_scale)
    if isinstance(raw, Skip):
        tables.skip_all(point, index, raw.reason)
        return
    fold = _Fold(index, data, fold_data(data, draw), draw, raw, test_pos, eval_pos)
    tables.add_raw(point, fold)
    members: dict[str, pd.Series] = {}
    for entry, correction in corrections.items():
        rng = rng_for(point.seed, point.scheme, point.label, index, entry.family)
        fitted = _fit_entry(correction, fold, rng, tables.settings.covariate_scale)
        if isinstance(fitted, Skip):
            tables.skip(entry, point, index, fitted.reason)
            continue
        outcome, test_predictions = fitted
        tables.outcomes.setdefault(entry, []).append(outcome)
        members[entry.key] = test_predictions
    for entry in tables.settings.ensemble_entries:
        combined = _combine_entry(entry, members, fold)
        if isinstance(combined, Skip):
            tables.skip(entry, point, index, combined.reason)
            continue
        tables.outcomes.setdefault(entry, []).append(combined)
