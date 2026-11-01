"""The computation of one target's row of the validity panel."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from copernicus.errors import ConfigurationError, EstimatorError
from copernicus.validity.anatomy import (
    COVARIATE_SCALES,
    ScoreErrors,
    constant_score_errors,
    replicate_noise,
    score_errors,
)
from copernicus.validity.bootstrap import BootstrapCI, VectorMetric, bootstrap_many
from copernicus.validity.metrics import Association, association, is_constant
from copernicus.validity.reliability import share_of_ceiling

if TYPE_CHECKING:
    from copernicus.validity.panel import PanelSettings

logger = logging.getLogger(__name__)

__all__ = ["PANEL_COLUMNS", "TargetResult", "compute_target"]

PANEL_COLUMNS = (
    "target",
    "gold_column",
    "n_used",
    "n_dropped_missing",
    "n_runs",
    "status",
    "direction_applied",
    "n",
    "spearman",
    "rank_validity",
    "rank_validity_low",
    "rank_validity_high",
    "pearson",
    "pearson_low",
    "pearson_high",
    "kendall",
    "auc",
    "somers_d",
    "mae_01",
    "mae_01_low",
    "mae_01_high",
    "signed_gap_01",
    "level_displacement",
    "level_displacement_low",
    "level_displacement_high",
    "calibration_shortfall",
    "calibration_shortfall_low",
    "calibration_shortfall_high",
    "covariate_linked_error",
    "covariate_linked_error_low",
    "covariate_linked_error_high",
    "covariate_linked_error_raw",
    "covariate_linked_error_standardized",
    "covariate",
    "covariate_scale",
    "noise",
    "ceiling",
    "share_of_ceiling",
    "n_boot",
    "n_undefined_max",
)
NAN = float("nan")
ERROR_FIELDS = ("mae_01", "level_displacement", "calibration_shortfall", "covariate_linked_error")


@dataclass(frozen=True)
class TargetResult:
    """One panel row with the texts that entered it and the texts dropped from it.

    Attributes:
        row: Column name to value, in the panel's column order.
        used: Text ids with a score, a gold value and a covariate value.
        dropped: Scored text ids lacking a score (in every run), a gold value or a covariate.
    """

    row: dict[str, Any]
    used: frozenset[str]
    dropped: frozenset[str]


def _rank_validity(score: np.ndarray, gold: np.ndarray, covariate: np.ndarray) -> float:
    if is_constant(score) or is_constant(gold):
        return NAN
    return abs(float(spearmanr(score, gold).statistic))


def _pearson(score: np.ndarray, gold: np.ndarray, covariate: np.ndarray) -> float:
    if is_constant(score) or is_constant(gold):
        return NAN
    return float(pearsonr(score, gold).statistic)


def _error_values(
    scale: Literal["standardized", "raw"],
    score: np.ndarray,
    gold: np.ndarray,
    covariate: np.ndarray,
) -> dict[str, float]:
    """The four bootstrapped score errors from one ``score_errors`` call."""
    errors = score_errors(score, gold, covariate, covariate_scale=scale)
    return {name: float(getattr(errors, name)) for name in ERROR_FIELDS}


def _applied_direction(
    direction: Literal["positive", "negative", "auto"], score: np.ndarray, gold: np.ndarray
) -> Literal["positive", "negative"]:
    if direction == "auto":
        probe = association(score, gold)
        return "negative" if probe.status == "ok" and probe.spearman < 0 else "positive"
    return direction


def _empty_row(key: str, gold_column: str, settings: PanelSettings) -> dict[str, Any]:
    """A row of NaN estimates in the panel's column order, with the settings filled in."""
    row: dict[str, Any] = dict.fromkeys(PANEL_COLUMNS, NAN)
    row.update(
        target=key,
        gold_column=gold_column,
        covariate_scale=settings.covariate_scale,
    )
    return row


def _intervals(
    key: str,
    point: Mapping[str, float],
    score: np.ndarray,
    gold: np.ndarray,
    covariate: np.ndarray,
    clusters: np.ndarray,
    settings: PanelSettings,
) -> dict[str, BootstrapCI]:
    """Bootstrap the metrics whose point estimate exists, on one shared set of resamples."""
    association_metrics: dict[str, Callable[..., float]] = {
        "rank_validity": _rank_validity,
        "pearson": _pearson,
    }
    defined = {
        name: metric for name, metric in association_metrics.items() if np.isfinite(point[name])
    }
    error_names = tuple(name for name in ERROR_FIELDS if np.isfinite(point[name]))
    skipped = sorted(
        (set(association_metrics) - set(defined)) | (set(ERROR_FIELDS) - set(error_names))
    )
    if skipped:
        logger.info(
            "target %r: no interval for %s, whose point estimate is undefined", key, skipped
        )
    errors = VectorMetric(error_names, partial(_error_values, settings.covariate_scale))
    try:
        return bootstrap_many(
            defined,
            score,
            gold,
            clusters,
            covariate,
            n_boot=settings.n_boot,
            seed=settings.seed,
            vector_metrics=(errors,) if error_names else (),
        )
    except (EstimatorError, ConfigurationError) as error:
        raise type(error)(f"target {key}: {error}") from error


def compute_target(
    key: str,
    gold_column: str,
    runs: pd.DataFrame,
    gold: pd.Series,
    covariate: pd.Series,
    covariate_name: str,
    groups: pd.Series,
    ceiling: float,
    settings: PanelSettings,
) -> TargetResult:
    """Compute one target's row.

    The score is averaged over runs per text. A scored text whose averaged score, gold
    value or covariate is missing (or whose id has no gold row) is dropped and counted,
    never imputed. The noise uses the (text, run) rows of the texts that entered only: their
    run scores are min-max scaled to 0-1 over all those rows before the per-text standard
    deviation is taken, so it is on the scale of the other errors. When the association is
    not ``ok`` (too few pairs or a constant score or gold) the association fields, the
    errors, the intervals and ``n_boot`` stay missing (no bootstrap runs), except that a
    constant score keeps its calibration shortfall (-1) and its covariate-linked error (see
    ``constant_score_errors``) unless the gold is constant too.
    When the covariate-linked error is undefined (a constant covariate) its interval is NaN
    and the other intervals are still computed.

    Args:
        key: Target name for the output.
        gold_column: Name of the gold column, recorded in the row.
        runs: Frame with ``text_id`` (string), ``run_number`` and ``score``.
        gold: Gold values indexed by string text id.
        covariate: Covariate values indexed by string text id.
        covariate_name: Name recorded in the ``covariate`` column.
        groups: Cluster of each text, indexed by string text id.
        ceiling: Reliability ceiling of this row; NaN when none applies.
        settings: Panel settings.

    Returns:
        The row and the sets of used and dropped text ids.

    Raises:
        EstimatorError: If a bootstrap metric is undefined on too many replicates; the
            message starts with the target name.
        ConfigurationError: If the bootstrap cannot run (for example fewer than two
            clusters among the used texts); the message starts with the target name.
    """
    per_text = runs.groupby("text_id")["score"].mean()
    ids = per_text.index
    aligned = pd.DataFrame(
        {
            "score": per_text.to_numpy(),
            "gold": gold.reindex(ids).to_numpy(),
            "covariate": covariate.reindex(ids).to_numpy(),
        },
        index=ids,
    )
    complete = np.isfinite(aligned.to_numpy(dtype=float)).all(axis=1)
    used = aligned[complete]
    score = used["score"].to_numpy()
    gold_values = used["gold"].to_numpy()
    covariate_values = used["covariate"].to_numpy()
    direction = _applied_direction(settings.direction, score, gold_values)
    if direction == "negative":
        score = -score
    found = association(score, gold_values)
    row = _empty_row(key, gold_column, settings)
    row.update(
        n_used=int(used.shape[0]),
        n_dropped_missing=int((~complete).sum()),
        n_runs=int(runs["run_number"].nunique()),
        status=found.status,
        direction_applied=direction,
        covariate=covariate_name,
        noise=replicate_noise(runs[runs["text_id"].isin(used.index)], scale="observed"),
        ceiling=ceiling,
        n=found.n,
    )
    if found.status == "ok":
        row.update(
            _estimates(
                key, found, score, gold_values, covariate_values, used.index, groups, settings
            )
        )
    elif found.status == "constant_score" and not is_constant(gold_values):
        kept = {
            scale: constant_score_errors(gold_values, covariate_values, covariate_scale=scale)
            for scale in COVARIATE_SCALES
        }
        row.update(
            calibration_shortfall=kept[settings.covariate_scale].calibration_shortfall,
            covariate_linked_error=kept[settings.covariate_scale].covariate_linked_error,
            **_both_scales(kept["raw"], kept["standardized"]),
        )
    row["share_of_ceiling"] = (
        share_of_ceiling(row["rank_validity"], ceiling) if np.isfinite(ceiling) else NAN
    )
    return TargetResult(row=row, used=frozenset(used.index), dropped=frozenset(ids[~complete]))


def _both_scales(raw: ScoreErrors, standardized: ScoreErrors) -> dict[str, float]:
    """The covariate-linked error on both scales, as the ``covariate_linked_error_*`` columns."""
    return {
        "covariate_linked_error_raw": raw.covariate_linked_error,
        "covariate_linked_error_standardized": standardized.covariate_linked_error,
    }


def _estimates(
    key: str,
    found: Association,
    score: np.ndarray,
    gold: np.ndarray,
    covariate: np.ndarray,
    used_ids: pd.Index,
    groups: pd.Series,
    settings: PanelSettings,
) -> dict[str, Any]:
    """Point estimates and intervals of a target whose association is defined."""
    by_scale = {
        scale: score_errors(score, gold, covariate, covariate_scale=scale)
        for scale in COVARIATE_SCALES
    }
    errors = by_scale[settings.covariate_scale]
    point = {
        "rank_validity": found.rank_validity,
        "pearson": found.pearson,
        "mae_01": errors.mae_01,
        "level_displacement": errors.level_displacement,
        "calibration_shortfall": errors.calibration_shortfall,
        "covariate_linked_error": errors.covariate_linked_error,
    }
    clusters = groups.reindex(used_ids).to_numpy()
    intervals = _intervals(key, point, score, gold, covariate, clusters, settings)
    estimates: dict[str, Any] = {
        "spearman": found.spearman,
        "kendall": found.kendall,
        "auc": found.auc,
        "somers_d": found.somers_d,
        "signed_gap_01": errors.signed_gap_01,
        **point,
        **_both_scales(by_scale["raw"], by_scale["standardized"]),
    }
    for name, interval in intervals.items():
        estimates[f"{name}_low"] = interval.low
        estimates[f"{name}_high"] = interval.high
    estimates["n_boot"] = settings.n_boot
    estimates["n_undefined_max"] = max(interval.n_undefined for interval in intervals.values())
    return estimates
