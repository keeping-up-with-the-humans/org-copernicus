"""The correction battery: every applicable score-level correction compared out of sample.

For each seed, scheme and size the texts are split into outer folds; in each fold a labelled
sample is drawn by whole groups from the training texts that carry a gold value
(``battery_rows``), every entry is fitted on the drawn texts and predicts the held-out texts
(``battery_folds``), and both the entry and the raw baseline (the score column with the
highest signed Spearman correlation with the gold values of the drawn texts) are scored on the
held-out texts that carry a gold value (``battery_metrics``). The correction ensemble pools
the held-out predictions of the entries fitted in a fold; its output is rank-only, so only
its rank validity and signed Spearman correlation are scored. An entry whose fit is undefined
on a fold is recorded in ``skipped`` with the reason and the battery goes on; any other
exception stops it. A metric that is undefined on a scored fold (a constant prediction or raw
score on the held-out texts, a constant covariate) leaves that fold out of its mean; the fold
is named in the provenance and ``n_folds_used`` counts the folds of the rank validity.

Per seed and integer size, when every text carries a gold value, one diagnostic row scores the
pseudo-values of a whole-data draw with the identity rectifier (the first score column)
against the gold values of every text, beside the rank validity of that score column
(``diagnostic = True``, scheme ``whole_data``). Estimate-level corrections are not part of the
battery.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus._io import json_ready
from copernicus.corrections.battery_folds import Point, Tables, run_fold
from copernicus.corrections.battery_rows import (
    NO_FOLD,
    RAW_COLUMNS,
    RESULT_COLUMNS,
    SKIPPED_COLUMNS,
    Skip,
    keyed_seed,
    partition,
    quota_problem,
    table,
)
from copernicus.corrections.battery_settings import BatterySettings, scheme_folds, size_label
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families import get_correction, pseudo_values
from copernicus.corrections.sampling import attach_draw, draw_from_data
from copernicus.errors import ConfigurationError
from copernicus.validity.metrics import association

logger = logging.getLogger(__name__)

__all__ = [
    "RAW_COLUMNS",
    "RESULT_COLUMNS",
    "SKIPPED_COLUMNS",
    "SUMMARY_COLUMNS",
    "BatteryResult",
    "BatterySettings",
    "run_battery",
    "summarize_battery",
]

DIAGNOSTIC_FAMILY = "pseudo_values"
DIAGNOSTIC_VARIANT = "identity"
WHOLE_DATA = "whole_data"
DIAGNOSTIC_NEEDS_FULL_GOLD = "diagnostic_needs_full_gold"
MIN_HOLDOUT_GROUPS = 2
MIN_SEEDS_FOR_SE = 2
NOT_SCORED_BY_DIAGNOSTIC = (
    "mae_raw",
    "mae_corrected",
    "gain_mae",
    "mae_native_raw",
    "mae_native_corrected",
    "gain_mae_native",
    "covariate_error_raw",
    "covariate_error_corrected",
    "gain_covariate_error",
)
GAINS = ("gain_rank_validity", "gain_mae", "gain_mae_native", "gain_covariate_error")
SUMMARY_COLUMNS = (
    "family",
    "variant",
    *(name for gain in GAINS for name in (gain, f"{gain}_se", f"n_defined_{gain}")),
    "n_rows",
    "n_seeds",
    "n_skipped",
)
"""The columns of ``summarize_battery``: the family and the variant, each gain with its
seed-clustered standard error and ``n_defined_<gain>``, then ``n_rows``, ``n_seeds`` and
``n_skipped``."""


@dataclass(frozen=True, eq=False)
class BatteryResult:
    """The tables of ``run_battery`` with the settings they come from.

    The tables are not the paper's battery tables and do not reproduce its numbers: each metric
    is a mean of per-fold values weighted by the fold's scored texts, where the paper pooled the
    scaled held-out texts of all folds for its error metrics (so the covariate-linked error here
    is a mean of fold slopes, not one pooled slope); the native MAE is the mean absolute error
    per text, where the paper reported the level error ``|mean prediction - mean gold|``; no
    fold is dropped for holding too few units; and the ensemble pools the entries fitted in each
    fold, so its members can change from fold to fold, where the paper admitted only variants
    fitted on every fold.

    Attributes:
        results: One row per entry, seed, scheme and size with at least one scored fold, with
            the columns of ``RESULT_COLUMNS``; entries in the order of the settings, then the
            diagnostic rows. ``size`` is the size's label (``"100"``, ``"0.05"``), ``variant``
            the entry's label (the variant, with its parameters in parentheses when it has
            any), ``n_labelled`` the mean number of drawn texts over the scored folds,
            ``n_folds_used`` the scored folds whose raw and corrected rank validities are both
            defined (the other metrics are averaged over their own defined folds), and
            ``provenance`` the strict-JSON list of the scored folds (draw, raw column, both
            scores, ``status``, ``status_raw``, ``dropped_from`` and what was fitted).
        skipped: One row per entry, seed, scheme, size and fold that was not scored, with the
            columns of ``SKIPPED_COLUMNS``; ``fold`` is -1 for a decision not tied to a fold.
        raw: One row per scored fold with the columns of ``RAW_COLUMNS``: the raw column
            chosen on the drawn texts and its scores on the held-out texts.
        settings: The settings of the run, with ``"all"`` fitted to the data's embedding sets
            (``BatterySettings.for_embedding_sets``).
    """

    results: pd.DataFrame
    skipped: pd.DataFrame
    raw: pd.DataFrame
    settings: BatterySettings


def _check_data(data: CorrectionData, settings: BatterySettings) -> None:
    """Refuse data without gold values or with fewer groups than a scheme needs."""
    if data.n_labelled == 0:
        raise ConfigurationError(
            "the battery compares corrections against gold values, and no text of the data has "
            "a gold value"
        )
    n_groups = int(np.unique(data.groups).size)
    for scheme in settings.schemes:
        folds = scheme_folds(scheme)
        needed = MIN_HOLDOUT_GROUPS if folds is None else folds
        if n_groups < needed:
            raise ConfigurationError(
                f"scheme {scheme} needs at least {needed} groups; the data have {n_groups}"
            )


def _diagnostic(data: CorrectionData, seed: int, size: int) -> dict[str, object] | Skip:
    """The pseudo-value diagnostic of one seed and size, or why it cannot be computed."""
    missing = data.n - data.n_labelled
    if missing:
        return Skip(
            f"{DIAGNOSTIC_NEEDS_FULL_GOLD}: {missing} of the {data.n} texts have no gold value; "
            "the diagnostic scores a whole-data draw against the gold value of every text"
        )
    problem = quota_problem(size, data.n)
    if problem:
        return Skip(problem)
    draw_seed = keyed_seed(seed, size, "diagnostic")
    draw = draw_from_data(data, size, seed=draw_seed)
    values = pseudo_values(
        attach_draw(data, pd.Series(data.gold, index=data.ids), draw), rectifier="identity"
    )
    raw = association(data.frame[values.score_column].to_numpy(dtype=float), data.gold)
    corrected = association(values.column.to_numpy(dtype=float), data.gold)
    provenance = {
        "quota": size,
        "units": data.n,
        "groups_drawn": draw.m,
        "n_groups": draw.n_groups,
        "pi": draw.pi,
        "draw_seed": draw_seed,
        "lambda_star": values.lambda_star,
        "mean_plugin": values.mean_plugin,
        "mean_rectified": values.mean_rectified,
        "mean_rectified_tuned": values.mean_rectified_tuned,
        "status_raw": raw.status,
        "status_corrected": corrected.status,
    }
    nan = float("nan")
    return {
        "family": DIAGNOSTIC_FAMILY,
        "variant": DIAGNOSTIC_VARIANT,
        "seed": seed,
        "scheme": WHOLE_DATA,
        "size": size_label(size),
        "n_labelled": float(draw.ids.size),
        "n_folds": 1,
        "n_folds_used": int(
            np.isfinite(raw.rank_validity) and np.isfinite(corrected.rank_validity)
        ),
        "rank_validity_raw": raw.rank_validity,
        "rank_validity_corrected": corrected.rank_validity,
        "spearman_raw": raw.spearman,
        "spearman_corrected": corrected.spearman,
        "gain_rank_validity": corrected.rank_validity - raw.rank_validity,
        **dict.fromkeys(NOT_SCORED_BY_DIAGNOSTIC, nan),
        "raw_columns": values.score_column,
        "diagnostic": True,
        "provenance": json.dumps(json_ready([provenance]), default=str),
    }


def _add_diagnostics(data: CorrectionData, tables: Tables, seed: int, seed_index: int) -> None:
    """Add the diagnostic row, or its skip row, of every integer size of one seed."""
    settings = tables.settings
    for size_index, size in enumerate(settings.sizes):
        if not isinstance(size, int):
            continue
        order = (len(settings.entries), seed_index, len(settings.schemes), size_index)
        row = _diagnostic(data, seed, size)
        if isinstance(row, Skip):
            identity = {"seed": seed, "scheme": WHOLE_DATA, "size": size_label(size)}
            skip = {"family": DIAGNOSTIC_FAMILY, "variant": DIAGNOSTIC_VARIANT, **identity}
            tables.skipped.append(
                ((*order, NO_FOLD), {**skip, "fold": NO_FOLD, "reason": row.reason})
            )
        else:
            tables.results.append((order, row))


def run_battery(data: CorrectionData, settings: BatterySettings) -> BatteryResult:
    """Compare the corrections of the settings out of sample over seeds, schemes and sizes.

    Args:
        data: The correction data; the labelled samples are drawn from the texts with a gold
            value and the corrections are scored against the gold values of the held-out
            texts. A ``pi`` column is replaced by the inclusion probability of each draw.
        settings: The entries, sizes, schemes, seeds and covariate scale. Under ``"all"`` the
            embedding-augmented entries read the stacked embedding sets, the one set of the
            data, or are left out without embeddings.

    Returns:
        The results, the skipped fits and the raw baseline of every scored fold, with the
        settings as run. The tables are not the paper's battery tables (see ``BatteryResult``
        for the four ways in which their numbers differ).

    Raises:
        ConfigurationError: If no text has a gold value, the data have fewer groups than a
            scheme needs, or the data do not match what an entry reads (such as an embedding
            set that misses texts).
        EstimatorError: If a correction returns predictions that are not indexed by the ids
            asked for.
    """
    settings = settings.for_embedding_sets(
        () if data.embeddings is None else tuple(data.embeddings)
    )
    _check_data(data, settings)
    corrections = {
        entry: get_correction(entry.family, entry.variant, **entry.kwargs)
        for entry in settings.per_text_entries
    }
    tables = Tables(settings)
    for seed_index, seed in enumerate(settings.seeds):
        for scheme_index, scheme in enumerate(settings.schemes):
            folds = partition(data, scheme, seed)
            for size_index, size in enumerate(settings.sizes):
                point = Point(seed, scheme, size, (seed_index, scheme_index, size_index))
                counts = (len(tables.raw), len(tables.skipped))
                if isinstance(folds, Skip):
                    tables.skip_all(point, NO_FOLD, folds.reason)
                else:
                    for index, split in enumerate(folds):
                        run_fold(data, corrections, tables, point, index, split)
                    tables.close(point)
                logger.info(
                    "battery seed %s, scheme %s, size %s: %d folds scored, %d skip rows",
                    seed,
                    scheme,
                    point.label,
                    len(tables.raw) - counts[0],
                    len(tables.skipped) - counts[1],
                )
        _add_diagnostics(data, tables, seed, seed_index)
    return BatteryResult(
        results=table(tables.results, RESULT_COLUMNS),
        skipped=table(tables.skipped, SKIPPED_COLUMNS),
        raw=table(tables.raw, RAW_COLUMNS),
        settings=settings,
    )


def _mean_and_se(rows: pd.DataFrame, gain: str) -> tuple[float, float, int]:
    """The mean of the per-seed means of a gain over its defined rows, the seed-clustered
    standard error, and the number of defined rows."""
    values = rows[gain].astype(float)
    defined = np.isfinite(values.to_numpy())
    if not defined.any():
        return float("nan"), float("nan"), 0
    seed_means = values[defined].groupby(rows["seed"][defined]).mean()
    se = float("nan")
    if seed_means.size >= MIN_SEEDS_FOR_SE:
        se = float(seed_means.std(ddof=1)) / math.sqrt(seed_means.size)
    return float(seed_means.mean()), se, int(defined.sum())


def summarize_battery(result: BatteryResult) -> pd.DataFrame:
    """The mean gains of every entry over seeds, schemes and sizes, with seed-clustered errors.

    The diagnostic rows are left out. Each gain is averaged over the rows where it is defined
    (finite): first over the defined rows of a seed (its schemes and sizes), then over the
    seeds with a defined row, with the standard error of these per-seed means (their standard
    deviation over the square root of their number; NaN with fewer than two). The mean agrees
    with the plain mean over the defined rows when every seed has the same defined rows. A gain
    is NaN only when no row defines it, as for the error gains of the rank-only ensemble.

    Args:
        result: The output of ``run_battery``.

    Returns:
        One row per entry of the settings, in their order, with the columns of
        ``SUMMARY_COLUMNS``: the family, the entry's label, each gain's mean, its standard
        error and ``n_defined_<gain>`` (the rows where the gain is defined), ``n_rows``
        (results rows), ``n_seeds`` and ``n_skipped`` (skip rows).
    """
    fitted = result.results[~result.results["diagnostic"]]
    skipped = result.skipped
    rows: list[dict[str, object]] = []
    for entry in result.settings.entries:
        mine = fitted[(fitted["family"] == entry.family) & (fitted["variant"] == entry.label)]
        row: dict[str, object] = {"family": entry.family, "variant": entry.label}
        for gain in GAINS:
            row[gain], row[f"{gain}_se"], row[f"n_defined_{gain}"] = _mean_and_se(mine, gain)
        own = (skipped["family"] == entry.family) & (skipped["variant"] == entry.label)
        row.update(n_rows=len(mine), n_seeds=int(mine["seed"].nunique()), n_skipped=int(own.sum()))
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
