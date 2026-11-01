"""The fits of one point of the baseline and the assembly of its result tables.

``fit_point`` fits every variant of one encoder set at one (seed, size) over the folds,
``fit_variant`` fits one variant on a shared ridge design, ``summarise_point`` condenses the
folds of one fitted (encoder set, variant, seed, size) into a results row, and
``assemble_tables`` builds the tables in the order of the settings. Every table labels a size
with the string ``size`` (``"50"`` or ``"full"``) and gives its unit quota in ``size_units``
(a nullable integer, missing for ``"full"``), so a column never mixes numbers and text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from copernicus.baseline.folds import FULL, DrawReceipt
from copernicus.baseline.metrics import aggregate_units, fold_metrics, weighted_over_folds
from copernicus.baseline.ridge import RidgeDesign, RidgeFit, apply_scaler, fit_scaler, predict
from copernicus.baseline.runner_inputs import AlignedInputs
from copernicus.errors import EstimatorError
from copernicus.validity.metrics import is_constant

__all__ = [
    "CURVE_COLUMNS",
    "IDENTITY_COLUMNS",
    "OOF_COLUMNS",
    "SKIPPED_COLUMNS",
    "VARIANT_ONLY",
    "VARIANT_PLUS",
    "assemble_tables",
    "fit_point",
    "fit_variant",
    "result_columns",
    "size_label",
    "size_units",
    "sort_by_settings",
    "summarise_point",
    "typed_sizes",
]

VARIANT_ONLY = "embeddings_only"
VARIANT_PLUS = "embeddings_plus_score"
IDENTITY_COLUMNS = (
    "encoder_set",
    "variant",
    "seed",
    "size",
    "size_units",
    "n_rows",
    "n_units",
    "n_groups",
    "n_features",
)
TAIL_COLUMNS = (
    "alpha_median",
    "n_edge_folds",
    "n_folds",
    "draw_units_min",
    "draw_units_mean",
    "draw_units_max",
    "draw_rows_mean",
    "draw_groups_mean",
    "draw_overshoot_max",
    "pool_units_mean",
)
OOF_COLUMNS = (
    "text_id",
    "seed",
    "size",
    "size_units",
    "encoder_set",
    "variant",
    "fold",
    "prediction",
    "gold",
    "alpha",
    "at_grid_edge",
)
SKIPPED_COLUMNS = ("seed", "size", "size_units", "reason")
CURVE_COLUMNS = (
    "encoder_set",
    "variant",
    "size",
    "size_units",
    "rank_validity",
    "rank_validity_sd",
    "spearman",
    "n_seeds",
)


def result_columns() -> list[str]:
    """Column names of the results table, in order."""
    return [*IDENTITY_COLUMNS, *weighted_over_folds([]), *TAIL_COLUMNS]


def size_label(size: int | str) -> str:
    """The ``size`` label of a training size: its digits, or ``"full"``."""
    return FULL if size == FULL else str(int(size))


def size_units(size: int | str) -> int | None:
    """The ``size_units`` value of a training size: its unit quota, ``None`` for ``"full"``."""
    return None if size == FULL else int(size)


def typed_sizes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the table with ``size`` as strings and ``size_units`` as nullable integers."""
    return frame.assign(
        size=frame["size"].astype(str), size_units=frame["size_units"].astype("Int64")
    )


def assemble_tables(
    rows: Sequence[Mapping[str, object]],
    oof_parts: Sequence[pd.DataFrame],
    skipped: Sequence[Mapping[str, object]],
    orders: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the results, out-of-fold and skipped tables in the order of the settings.

    Args:
        rows: One results row per fitted (encoder set, variant, seed, size).
        oof_parts: Out-of-fold predictions, one frame per fold and variant at ``"full"``.
        skipped: One row per (seed, size) that was not fitted.
        orders: For ``encoder_set``, ``variant``, ``size`` and ``seed``, the values as
            strings in the order of the settings.

    Returns:
        The results, the out-of-fold predictions and the skipped sizes, each with a fresh
        index, string ``size`` labels and nullable integer ``size_units``.
    """
    results = pd.DataFrame(list(rows), columns=result_columns())
    oof = (
        pd.concat(oof_parts, ignore_index=True)[list(OOF_COLUMNS)]
        if oof_parts
        else pd.DataFrame(columns=list(OOF_COLUMNS))
    )
    skip_table = pd.DataFrame(list(skipped), columns=list(SKIPPED_COLUMNS))
    results = sort_by_settings(results, orders, ["encoder_set", "variant", "size", "seed"])
    oof = sort_by_settings(oof, orders, ["encoder_set", "variant", "size", "seed", "fold"])
    skip_table = sort_by_settings(skip_table, orders, ["seed", "size"])
    return typed_sizes(results), typed_sizes(oof), typed_sizes(skip_table)


def summarise_point(
    identity: Mapping[str, object],
    fold_values: Sequence[Mapping[str, float | str]],
    alphas: Sequence[float],
    edge_flags: Sequence[bool],
    receipts: Sequence[DrawReceipt],
) -> dict[str, object]:
    """One results row from the folds of a fitted (encoder set, variant, seed, size).

    Args:
        identity: The leading identity columns.
        fold_values: ``fold_metrics`` output of each fold.
        alphas: Selected penalty of each fold.
        edge_flags: Whether each fold's penalty sits on a grid edge.
        receipts: Draw receipt of each fold.

    Returns:
        The row, with the weighted metrics, the median penalty, the edge count and the
        receipt summaries.
    """
    units = np.array([receipt.actual_units for receipt in receipts], dtype=float)
    row: dict[str, object] = dict(identity)
    row.update(weighted_over_folds(fold_values))
    row["alpha_median"] = float(np.median(alphas))
    row["n_edge_folds"] = int(sum(edge_flags))
    row["n_folds"] = len(fold_values)
    row["draw_units_min"] = float(units.min())
    row["draw_units_mean"] = float(units.mean())
    row["draw_units_max"] = float(units.max())
    row["draw_rows_mean"] = float(np.mean([receipt.actual_rows for receipt in receipts]))
    row["draw_groups_mean"] = float(np.mean([receipt.actual_groups for receipt in receipts]))
    row["draw_overshoot_max"] = float(max(receipt.overshoot for receipt in receipts))
    row["pool_units_mean"] = float(np.mean([receipt.pool_units for receipt in receipts]))
    return row


def sort_by_settings(
    frame: pd.DataFrame, orders: Mapping[str, Sequence[str]], by: Sequence[str]
) -> pd.DataFrame:
    """Stable sort by columns whose order is given by the settings.

    Args:
        frame: Table to sort.
        orders: For a column name, its values as strings in settings order; the column
            ``fold`` and any column without an entry sorts by its own values.
        by: Sort columns, most significant first.

    Returns:
        The sorted table with a fresh index.
    """
    if frame.empty:
        return frame.reset_index(drop=True)
    keys = pd.DataFrame(index=frame.index)
    for name in by:
        if name in orders:
            keys[name] = frame[name].astype(str).map(list(orders[name]).index)
        else:
            keys[name] = frame[name]
    return frame.loc[keys.sort_values(list(by), kind="stable").index].reset_index(drop=True)


def fit_variant(
    design: RidgeDesign,
    test_features: np.ndarray,
    alphas: np.ndarray,
    score_draw: np.ndarray | None,
    score_test: np.ndarray | None,
) -> tuple[np.ndarray, RidgeFit]:
    """Fit one variant on the shared design and predict the test fold.

    Args:
        design: Ridge design of the draw, shared by the variants.
        test_features: Scaled design of the test fold.
        alphas: Penalty grid.
        score_draw: Coder score on the draw for the plus-score variant, else ``None``.
        score_test: Coder score on the test fold for the plus-score variant, else ``None``.

    Returns:
        The test-fold predictions and the fit.

    Raises:
        EstimatorError: If the coder score is constant on the draw.
    """
    if score_draw is None or score_test is None:
        fit = design.fit(alphas)
        return predict(fit, test_features), fit
    if is_constant(score_draw):
        raise EstimatorError("embeddings_plus_score: constant coder score on the draw")
    mean, sd = float(score_draw.mean()), float(score_draw.std())
    fit = design.fit(alphas, extra=(score_draw - mean) / sd)
    features = np.hstack([test_features, ((score_test - mean) / sd)[:, None]])
    return predict(fit, features), fit


def fit_point(
    aligned: AlignedInputs,
    keys: tuple[str, ...],
    folds: list[tuple[np.ndarray, np.ndarray]],
    draws: list[tuple[np.ndarray, DrawReceipt]],
    alphas: np.ndarray,
    classes: tuple[float, float] | None,
    identity: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    """Fit every variant of one encoder set at one fitted (seed, size).

    Args:
        aligned: The aligned inputs.
        keys: The embedding keys of the encoder set.
        folds: Training and test positions of each fold.
        draws: The training draw and its receipt for each fold.
        alphas: Penalty grid.
        classes: The two values of a binary gold, or ``None``.
        identity: The identity columns of the point (``size`` is the string label).

    Returns:
        One results row per variant, and the out-of-fold predictions of each fold and
        variant when the size is ``"full"``.
    """
    variants = [VARIANT_ONLY] + ([VARIANT_PLUS] if aligned.score is not None else [])
    blocks = [aligned.blocks[key] for key in keys]
    per_fold: dict[str, list[dict[str, float | str]]] = {name: [] for name in variants}
    chosen: dict[str, list[float]] = {name: [] for name in variants}
    edges: dict[str, list[bool]] = {name: [] for name in variants}
    oof: list[pd.DataFrame] = []
    for fold_index, ((_, test), (draw, _)) in enumerate(zip(folds, draws, strict=True)):
        scaler = fit_scaler([block[draw] for block in blocks])
        train_features = apply_scaler([block[draw] for block in blocks], scaler)
        test_features = apply_scaler([block[test] for block in blocks], scaler)
        design = RidgeDesign(train_features, aligned.gold[draw], aligned.group_id[draw])
        for variant in variants:
            with_score = variant == VARIANT_PLUS
            prediction, fit = fit_variant(
                design,
                test_features,
                alphas,
                aligned.score[draw] if with_score and aligned.score is not None else None,
                aligned.score[test] if with_score and aligned.score is not None else None,
            )
            _, unit_pred, unit_gold = aggregate_units(
                prediction, aligned.gold[test], aligned.unit_id[test]
            )
            per_fold[variant].append(
                fold_metrics(unit_pred, unit_gold, classes, aligned.gold[draw])
            )
            chosen[variant].append(fit.alpha)
            edges[variant].append(fit.at_grid_edge)
            if identity["size"] == FULL:
                oof.append(
                    pd.DataFrame(
                        {
                            "text_id": aligned.text_ids[test],
                            "seed": identity["seed"],
                            "size": identity["size"],
                            "size_units": identity["size_units"],
                            "encoder_set": identity["encoder_set"],
                            "variant": variant,
                            "fold": fold_index,
                            "prediction": prediction,
                            "gold": aligned.gold[test],
                            "alpha": fit.alpha,
                            "at_grid_edge": fit.at_grid_edge,
                        }
                    )
                )
    receipts = [receipt for _, receipt in draws]
    n_features = sum(block.shape[1] for block in blocks)
    rows = [
        summarise_point(
            {**identity, "variant": variant, "n_features": n_features + (variant == VARIANT_PLUS)},
            per_fold[variant],
            chosen[variant],
            edges[variant],
            receipts,
        )
        for variant in variants
    ]
    return rows, oof
