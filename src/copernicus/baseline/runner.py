"""Run the NLP-ML baseline: grouped folds, training-size draws, ridge fits, curves.

For every seed the rows are split into grouped outer folds. For every requested training
size, each fold's training pool is cut down to a whole-group draw of that many analysis
units; a size the cohort cannot supply is recorded in ``skipped`` and fits nothing for that
seed and size. Three reasons exist: ``quota_exceeds_half_units`` (the quota is above half the
units), ``pool_below_quota`` (some fold's training pool holds fewer units than the quota) and
``below_floor`` (some fold's draw holds fewer units than the floor). No size is ever fitted
on fewer units than it names; in particular a pool smaller than the quota is not fitted as a
smaller run under the quota's label. For each
fitted (seed, size) and encoder set, the scaler is fitted on the draw, one ridge design is
shared by the variants, the penalty is chosen by exact grouped leave-one-group-out, and the
test fold is scored on unit means.

Variants: ``embeddings_only``, and ``embeddings_plus_score`` when a coder score is given
(the score is standardised on the draw and appended as one more penalised column).

Folds and draws that the user supplies (``BaselineSplits``) take the place of the keyed
generators once a checked copy of them passes the checks of ``runner_splits.check_splits``;
every result holds the folds and every draw that its run consulted (``draw_plan``).
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus.baseline.draw_plan import plan_draws, splits_from_positions
from copernicus.baseline.folds import FLOOR_UNITS, K_FOLDS, SIZES, grouped_kfold, rng_for
from copernicus.baseline.ridge import ALPHAS
from copernicus.baseline.runner_inputs import (
    align_inputs,
    binary_classes,
    resolve_encoder_sets,
)
from copernicus.baseline.runner_rows import (
    CURVE_COLUMNS,
    VARIANT_ONLY,
    VARIANT_PLUS,
    assemble_tables,
    fit_point,
    size_label,
    size_units,
    typed_sizes,
)
from copernicus.baseline.runner_splits import check_splits
from copernicus.baseline.splits import BaselineSplits, Draws, Folds, checked_splits
from copernicus.data.table import TextTable
from copernicus.embeddings.sets import EmbeddingSet
from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = ["BaselineResult", "BaselineSettings", "curves", "run_ridge_baseline"]

FULL = "full"
DEFAULT_SEEDS = (42, 43, 44)


def _is_integer(value: object) -> bool:
    """Whether the value is an integer (numpy integers included) and not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


@dataclass(frozen=True)
class BaselineSettings:
    """Settings of the ridge baseline.

    Attributes:
        k_folds: Number of grouped outer folds.
        seeds: Seeds of the fold assignment and the draws.
        sizes: Training sizes in analysis units; positive integers or ``"full"``.
        floor_units: Smallest number of units a draw may hold.
        alphas: Penalty grid of the ridge.
        encoder_sets: ``"single"`` (one set per embedding key), ``"stacked"`` (all keys
            concatenated), or a key of the embedding mapping.
        binary: Whether the gold is binary; ``None`` infers it from the gold values.
    """

    k_folds: int = K_FOLDS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    sizes: tuple[int | str, ...] = SIZES
    floor_units: int = FLOOR_UNITS
    alphas: tuple[float, ...] = tuple(float(alpha) for alpha in ALPHAS)
    encoder_sets: tuple[str, ...] = ("single",)
    binary: bool | None = None

    def __post_init__(self) -> None:
        """Check every setting against its range.

        Raises:
            SettingsError: If a setting is outside its documented range.
        """
        if not _is_integer(self.k_folds) or self.k_folds < 2:
            raise SettingsError(f"k_folds must be an integer of at least 2; got {self.k_folds!r}")
        if not self.seeds:
            raise SettingsError("seeds must not be empty")
        if not all(_is_integer(seed) for seed in self.seeds):
            raise SettingsError(f"seeds must be integers; got {self.seeds!r}")
        if len(set(self.seeds)) != len(self.seeds):
            raise SettingsError(f"seeds must not repeat; got {self.seeds!r}")
        if not self.sizes:
            raise SettingsError("sizes must not be empty")
        for size in self.sizes:
            is_count = not isinstance(size, str) and _is_integer(size) and size >= 1
            if not (is_count or size == FULL):
                raise SettingsError(f"each size must be a positive integer or 'full'; got {size!r}")
        if len({str(size) for size in self.sizes}) != len(self.sizes):
            raise SettingsError(f"sizes must not repeat; got {self.sizes!r}")
        if not _is_integer(self.floor_units) or self.floor_units < 1:
            raise SettingsError(
                f"floor_units must be an integer of at least 1; got {self.floor_units!r}"
            )
        if not self.alphas or not all(math.isfinite(alpha) and alpha > 0 for alpha in self.alphas):
            raise SettingsError("alphas must be a non-empty tuple of positive finite numbers")
        if not self.encoder_sets or not all(
            isinstance(name, str) and name for name in self.encoder_sets
        ):
            raise SettingsError("encoder_sets must be a non-empty tuple of names")
        if self.binary is not None and not isinstance(self.binary, bool):
            raise SettingsError(f"binary must be None or a bool; got {self.binary!r}")


@dataclass(frozen=True)
class BaselineResult:
    """Tables produced by ``run_ridge_baseline``, with what they were computed from.

    Every table names a training size by the string ``size`` (``"50"``, ``"full"``) and
    gives its unit quota in ``size_units`` (nullable integer, missing for ``"full"``). The
    ``seed`` columns of ``results``, ``oof_predictions`` and ``skipped`` hold the seeds'
    integer values; their dtype can be a signed integer, an unsigned integer or object, so
    do not assume ``int64``.

    Attributes:
        results: One row per fitted (encoder set, variant, seed, size): the cohort counts
            ``n_rows``, ``n_units`` and ``n_groups`` (all analysis rows, the same on every
            row; the size of each training draw is in the ``draw_*`` and ``pool_units_mean``
            columns), ``n_features``, the unit-weighted fold metrics, the median selected
            penalty, the number of folds with an edge penalty, and summaries of the draws.
        oof_predictions: Out-of-fold prediction of every text at size ``"full"``, with the
            penalty selected in its fold (``alpha``) and whether that penalty sits on an
            edge of the grid (``at_grid_edge``).
        skipped: One row per (seed, size) that was not fitted, with the reason.
        settings: The settings of the run.
        gold_column: The gold column that was predicted.
        unit_id: The column of the text table that held the analysis unit (``group_id``
            when none was named).
        binary: Whether the gold was treated as binary, as given in the settings or
            inferred from the gold values.
        splits: The folds and the training draws that the run consulted: the folds of every
            seed, each text in the order of the analysis rows, and the draws of every fitted
            (seed, size) and of every one then skipped below the floor, ``"full"`` included,
            seed by seed, size by size and fold by fold. Passed as ``splits`` to a run on the
            same inputs and settings, under any name of the gold column, they give this run's
            folds, draws and skips again, and on the same machine with the same thread count
            its tables bit for bit.
    """

    results: pd.DataFrame
    oof_predictions: pd.DataFrame
    skipped: pd.DataFrame
    settings: BaselineSettings
    gold_column: str
    unit_id: str
    binary: bool
    splits: BaselineSplits


def run_ridge_baseline(
    embeddings: Mapping[str, EmbeddingSet],
    gold: pd.DataFrame,
    table: TextTable,
    settings: BaselineSettings,
    *,
    gold_column: str,
    unit_id: str | None = None,
    coder_score: pd.Series | None = None,
    splits: BaselineSplits | None = None,
) -> BaselineResult:
    """Fit the NLP-ML baseline over seeds, training sizes and encoder sets.

    The random generators are keyed by the seed and the name of the gold column: the folds
    of seed ``s`` come from ``rng_for(s, gold_column, "folds")`` and the training draw of
    fold ``i`` at size ``q`` from ``rng_for(s, gold_column, q, i, "draw")``. The same inputs
    and settings therefore give the same tables, and renaming the gold column changes the
    folds and the draws.

    Supplied ``splits`` take the place of these generators: the folds always, and the draws
    unless ``splits.draws`` is ``None``, in which case the draws come from the generators
    above on the supplied folds. A checked copy of the splits is taken first, so that a table
    changed after construction meets every check. The three skip reasons apply to supplied
    splits as to drawn ones, and the draw receipts are computed from the supplied draws. A
    size skipped because its quota exceeds half the units or some fold's pool lies below it
    needs no draw; supplied draws of such a size are not used, and a warning says so. A count
    size without any supplied draw needs none either when its smallest pool holds fewer units
    than the floor: it is skipped with the reason ``below_floor(<floor>):min_pool_units=<n>``.
    ``result.splits`` holds the folds and every draw that the run consulted, those of a size
    then skipped below the floor included, so that a run given them reaches the same skips.

    Args:
        embeddings: Embedding sets keyed by encoder name.
        gold: Table with ``text_id`` and ``gold_column``; its rows are the analysis rows and
            the gold must be numeric and complete.
        table: Text table giving ``group_id`` (the folds and the whole-group draws) and the
            unit column.
        settings: Baseline settings.
        gold_column: Name of the numeric gold column in ``gold``.
        unit_id: Column of ``table.frame`` holding the analysis unit (metrics are computed
            on unit means and training sizes count units); ``None`` uses ``group_id``.
        coder_score: Optional score indexed by text id; adds the ``embeddings_plus_score``
            variant.
        splits: Folds, and optionally training draws, to use in place of the keyed
            generators (see ``BaselineSplits``); ``None`` draws both from the generators.

    Returns:
        The results, the out-of-fold predictions at size ``"full"`` and the skipped
        (seed, size) combinations, with the settings, the gold column, the unit column, the
        binary decision and the folds and draws they come from.

    Raises:
        ConfigurationError: If the inputs disagree (see ``align_inputs``), an encoder-set
            name is unknown, a binary gold varies within an analysis unit or ``binary`` is
            set for a gold without exactly two values (see ``binary_classes``), or there are
            fewer groups than folds. With ``splits``, also before any fit if ``splits`` is not a
            ``BaselineSplits`` or a table of it is now refused by ``BaselineSplits``, if the
            seeds of the folds are not exactly the seeds of the settings; a fold label lies outside
            ``0`` to ``k_folds - 1`` or a fold holds no text; the folds of a seed do not hold
            every analysis text exactly once, or hold a text that is not an analysis text; a
            group has texts in two folds of one seed; or, with draws, the draws hold a seed,
            a size or a fold label that the settings do not ask for, or for a (seed, size)
            whose draws the run consults a fold's draw is missing, holds a text outside that
            fold's training pool, holds part of a group, holds fewer units than the size names
            although the pool holds more, or is not the whole pool at size ``"full"``. The
            message names the seed, the size and the fold where they apply, and the first ten
            offending ids.
        EstimatorError: If the coder score is constant on a draw or a design is degenerate.
    """
    splits = None if splits is None else checked_splits(splits)
    sets = resolve_encoder_sets(settings.encoder_sets, list(embeddings))
    used = [key for key in embeddings if any(key in keys for keys in sets.values())]
    aligned = align_inputs(
        embeddings,
        used,
        gold,
        table,
        gold_column=gold_column,
        unit_id=unit_id,
        coder_score=coder_score,
    )
    classes = binary_classes(aligned, gold_column, settings.binary)
    supplied = None
    if splits is not None:
        supplied = check_splits(
            splits,
            aligned,
            seeds=settings.seeds,
            sizes=settings.sizes,
            k_folds=settings.k_folds,
            floor_units=settings.floor_units,
        )
    n_units = int(np.unique(aligned.unit_id).shape[0])
    counts = {
        "n_rows": int(aligned.gold.shape[0]),
        "n_units": n_units,
        "n_groups": int(np.unique(aligned.group_id).shape[0]),
    }
    alphas = np.asarray(settings.alphas, dtype=np.float64)
    rows: list[dict[str, object]] = []
    oof_parts: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    used_folds: dict[int, Folds] = {}
    used_draws: dict[tuple[int, str], Draws] = {}
    for seed in settings.seeds:
        if supplied is None:
            folds = grouped_kfold(
                aligned.group_id, settings.k_folds, rng_for(seed, gold_column, "folds")
            )
        else:
            folds = supplied.folds[int(seed)]
        used_folds[int(seed)] = folds
        given = None if supplied is None else supplied.draws
        pool_skips = {} if supplied is None else supplied.pool_skips
        for size in settings.sizes:
            labels = {"size": size_label(size), "size_units": size_units(size)}
            reason, draws = plan_draws(
                aligned,
                folds,
                seed=seed,
                size=size,
                n_units=n_units,
                floor_units=settings.floor_units,
                gold_column=gold_column,
                supplied=given,
                pool_skips=pool_skips,
            )
            if draws:
                used_draws[(int(seed), size_label(size))] = draws
            if reason:
                skipped.append({"seed": seed, **labels, "reason": reason})
                logger.info("seed %s size %s skipped: %s", seed, size, reason)
                continue
            logger.info("seed %s size %s fitted on %d encoder set(s)", seed, size, len(sets))
            for name, keys in sets.items():
                identity: dict[str, object] = {
                    "encoder_set": name,
                    "variant": "",
                    "seed": seed,
                    **labels,
                    **counts,
                }
                point_rows, point_oof = fit_point(
                    aligned, keys, folds, draws, alphas, classes, identity
                )
                rows.extend(point_rows)
                oof_parts.extend(point_oof)
    orders = {
        "encoder_set": list(sets),
        "variant": [VARIANT_ONLY, VARIANT_PLUS],
        "size": [size_label(size) for size in settings.sizes],
        "seed": [str(seed) for seed in settings.seeds],
    }
    results, oof, skip_table = assemble_tables(rows, oof_parts, skipped, orders)
    return BaselineResult(
        results=results,
        oof_predictions=oof,
        skipped=skip_table,
        settings=settings,
        gold_column=gold_column,
        unit_id=aligned.unit_column,
        binary=classes is not None,
        splits=splits_from_positions(aligned.text_ids, used_folds, used_draws),
    )


def _mean_over_seeds(values: pd.Series) -> float:
    return float(values.mean(skipna=False))


def _sd_over_seeds(values: pd.Series) -> float:
    return float(values.std(ddof=1, skipna=False))


def curves(result: BaselineResult) -> pd.DataFrame:
    """Rank validity and the signed Spearman correlation by training size, over seeds.

    ``rank_validity`` is the mean over seeds of the unit-weighted per-fold absolute
    Spearman correlation (``spearman_abs``) and ``spearman`` the mean over seeds of the
    unit-weighted per-fold signed one. The absolute value inflates weak predictors: a fold
    whose out-of-fold correlation is negative still adds to the rank validity, so an encoder
    without signal plots above zero. The orientation of a fitted model is fixed, so a
    ``spearman`` well below ``rank_validity`` marks such folds.

    Args:
        result: Output of ``run_ridge_baseline``.

    Returns:
        One row per (encoder set, variant, size) in the order of the settings, with
        ``size`` and ``size_units`` as in the results, ``rank_validity`` (mean over seeds),
        ``rank_validity_sd`` (sample SD, NaN with one seed), ``spearman`` (mean over seeds)
        and ``n_seeds``. Sizes that were skipped for every seed are absent.
    """
    keys = ["encoder_set", "variant", "size", "size_units"]
    grouped = result.results.groupby(keys, sort=False, dropna=False)
    table = grouped.agg(
        rank_validity=("spearman_abs", _mean_over_seeds),
        rank_validity_sd=("spearman_abs", _sd_over_seeds),
        spearman=("spearman", _mean_over_seeds),
        n_seeds=("spearman_abs", "size"),
    ).reset_index()
    return typed_sizes(table)[list(CURVE_COLUMNS)]
