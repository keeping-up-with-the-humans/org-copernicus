"""The validity panel: one row per target with associations, errors, intervals and ceiling."""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

from copernicus.data import TextTable
from copernicus.errors import ConfigurationError, SettingsError
from copernicus.validity.anatomy import COVARIATE_SCALES
from copernicus.validity.panel_inputs import (
    CONSTRUCT_KEYS,
    check_gold_overlap,
    check_scored_in_table,
    indexed_gold,
    numeric_column,
    resolve_covariate,
    resolve_groups,
    resolve_target_scores,
)
from copernicus.validity.panel_targets import PANEL_COLUMNS, compute_target
from copernicus.validity.reliability import cronbach_alpha, reliability_ceiling
from copernicus.validity.report import render_report

logger = logging.getLogger(__name__)

__all__ = ["PanelSettings", "ValidityPanel", "validity_panel"]

DIRECTIONS = ("positive", "negative", "auto")
INTEGER_COLUMNS = ("n_used", "n_dropped_missing", "n_runs", "n", "n_boot", "n_undefined_max")
ALPHA_ROUNDING_SLACK = 1e-9


@dataclass(frozen=True)
class PanelSettings:
    """Settings of the validity panel.

    Attributes:
        covariate: Name of the covariate for the covariate-linked error: a column of the
            gold frame or of the table. ``None`` uses ``log(n_chars)`` of the table.
        covariate_scale: ``standardized`` (slope per standard deviation of the covariate)
            or ``raw`` (slope per unit).
        n_boot: Number of bootstrap replicates; at least 1.
        seed: Seed of the bootstrap; a non-negative integer.
        direction: ``positive`` uses the score as given, ``negative`` negates it before
            every metric, ``auto`` negates when the signed Spearman correlation is
            negative.
        reliability: Reliability of the gold values in (0, 1], which sets the ceiling
            ``sqrt(reliability)``. A number applies to every row, which is right only when
            every target's gold column comes from one instrument. A mapping from gold column
            to reliability sets the ceiling of the rows of those columns and leaves the other
            rows without one; it is copied into a read-only mapping, and every key must be a
            gold column of the targets. A read-only mapping can be neither pickled nor
            deep-copied: use ``dict(settings.reliability)`` for that. It is compared but left
            out of the hash.
        items: Units-by-items matrix of the gold instrument (for example one column per
            human coder); Cronbach's alpha of it sets the ceiling of the construct row only.
            The items belong to the gold, not to the scored texts, so their rows need not
            align with the texts. Not compared and not shown in the representation.
    """

    covariate: str | None = None
    covariate_scale: Literal["standardized", "raw"] = "standardized"
    n_boot: int = 1000
    seed: int = 42
    direction: Literal["positive", "negative", "auto"] = "positive"
    reliability: float | Mapping[str, float] | None = field(default=None, hash=False)
    items: pd.DataFrame | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Refuse a setting outside its range or two settings that conflict.

        Raises:
            SettingsError: If a setting is outside its range, or both ``reliability`` and
                ``items`` are given.
        """
        if self.n_boot < 1:
            raise SettingsError(f"n_boot must be at least 1; got {self.n_boot}")
        if not _is_integer(self.seed) or self.seed < 0:
            raise SettingsError(f"seed must be a non-negative integer; got {self.seed!r}")
        if self.covariate_scale not in COVARIATE_SCALES:
            raise SettingsError(
                f"covariate_scale must be one of {COVARIATE_SCALES}; got {self.covariate_scale!r}"
            )
        if self.direction not in DIRECTIONS:
            raise SettingsError(f"direction must be one of {DIRECTIONS}; got {self.direction!r}")
        if isinstance(self.reliability, Mapping):
            copied = _reliability_by_column(self.reliability)
            object.__setattr__(self, "reliability", MappingProxyType(copied))
        elif self.reliability is not None and not (
            np.isfinite(self.reliability) and 0.0 < self.reliability <= 1.0
        ):
            raise SettingsError(f"reliability must lie in (0, 1]; got {self.reliability!r}")
        if self.reliability is not None and self.items is not None:
            raise SettingsError("give either reliability or items for the ceiling, not both")


def _is_integer(value: object) -> bool:
    """Whether the value is an integer (numpy integers included) and not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _reliability_by_column(given: Mapping[str, float]) -> dict[str, float]:
    """Copy a mapping from gold column to reliability, refusing a key or value out of range."""
    if not given:
        raise SettingsError("a reliability mapping must name at least one gold column")
    copied: dict[str, float] = {}
    for column, value in given.items():
        if not isinstance(column, str) or not column:
            raise SettingsError(f"reliability keys must be gold column names; got {column!r}")
        usable = isinstance(value, numbers.Real) and not isinstance(value, bool)
        if not (usable and math.isfinite(value) and 0.0 < value <= 1.0):
            raise SettingsError(f"reliability of {column!r} must lie in (0, 1]; got {value!r}")
        copied[column] = float(value)
    return copied


DEFAULT_SETTINGS = PanelSettings()


@dataclass(frozen=True, eq=False)
class ValidityPanel:
    """The panel frame with the settings it was computed under.

    Attributes:
        frame: One row per target; see `validity_panel` for the columns.
        settings: The settings used.
        n_used: Distinct texts that entered at least one target.
        n_dropped_missing: Distinct texts dropped from at least one target for a missing
            score, gold value or covariate.
        groups_defaulted: True when every text is its own cluster of the bootstrap: no
            table was given, or the table had no group column (``group_defaulted``), so the
            intervals treat the texts as independent.
    """

    frame: pd.DataFrame
    settings: PanelSettings
    n_used: int
    n_dropped_missing: int
    groups_defaulted: bool

    def report(self) -> str:
        """Return the one-page plain-text report of the panel."""
        return render_report(self)


def _construct_ceiling(items: pd.DataFrame) -> float:
    """Ceiling from Cronbach's alpha of the items, refused when alpha is not a reliability."""
    alpha = cronbach_alpha(items)
    if alpha > 1.0 and alpha - 1.0 <= ALPHA_ROUNDING_SLACK:
        alpha = 1.0
    if not 0.0 < alpha <= 1.0:
        raise ConfigurationError(
            f"Cronbach's alpha of the items is {alpha:.3f}; a ceiling needs a reliability in (0, 1]"
        )
    return reliability_ceiling(alpha)


def _ceilings(targets: Mapping[str, str], settings: PanelSettings) -> dict[str, float]:
    """Reliability ceiling per target key; NaN where none applies."""
    nan = float("nan")
    keys = list(targets)
    reliability = settings.reliability
    if isinstance(reliability, Mapping):
        used = sorted(set(targets.values()))
        unused = sorted(set(reliability) - set(used))
        if unused:
            raise ConfigurationError(
                f"reliability names gold column(s) {unused} that no target uses; the targets "
                f"use {used}"
            )
        return {
            key: reliability_ceiling(reliability[column]) if column in reliability else nan
            for key, column in targets.items()
        }
    if reliability is not None:
        return dict.fromkeys(keys, reliability_ceiling(reliability))
    if settings.items is None:
        return dict.fromkeys(keys, nan)
    construct_keys = [key for key in keys if key in CONSTRUCT_KEYS]
    if not construct_keys:
        raise ConfigurationError(
            "items give the ceiling of the construct target, but targets does not name 'construct'"
        )
    ceiling = _construct_ceiling(settings.items)
    return {key: ceiling if key in construct_keys else nan for key in keys}


def validity_panel(
    scores: pd.DataFrame,
    gold: pd.DataFrame,
    table: TextTable | None,
    *,
    targets: Mapping[str, str],
    settings: PanelSettings = DEFAULT_SETTINGS,
) -> ValidityPanel:
    """Compute the validity panel of scores against gold values.

    ``scores`` is either the long table of ``export_scores`` (``text_id``, ``run_number``,
    ``attribute_id``, ``score``), where a target key is an attribute id or ``construct``
    (the mean of the attribute scores that exist, per text and run), or a frame with a
    ``target_id`` column such as similarity or hybrid output (``run_number`` is taken as 1
    when absent). With a ``target_level`` column a key may be ``construct``,
    ``<rubric_id>__rollup``, ``<level>:<id>`` or a plain id found at one level only.
    The ``construct`` row of a similarity frame (a score against the construct poles) and
    the roll-up row (the mean of attribute-level similarity scores) are different
    estimators and are reported as different targets. Filter a grid of settings to one
    setting before calling.

    Per target the score is averaged over runs per text and compared with the gold
    column; a scored text with no gold row, or a missing score, gold or covariate value,
    is dropped and counted in ``n_dropped_missing``, never imputed. Gold and covariate
    columns must be numeric. Text ids are compared as strings. Clusters of the bootstrap
    are the table's ``group_id`` (the text ids themselves without a table, which then
    needs a ``covariate`` column). The output has one row per target, in the order of
    ``targets``, with the columns: ``target``, ``gold_column``, ``n_used``,
    ``n_dropped_missing``, ``n_runs``, ``status``, ``direction_applied``, ``n``,
    ``spearman``, ``rank_validity``, ``pearson``, ``kendall``, ``auc``, ``somers_d``,
    ``mae_01``, ``signed_gap_01``, ``level_displacement``, ``calibration_shortfall``,
    ``covariate_linked_error`` (the six intervals add ``_low`` and ``_high``; the
    covariate-linked error and its interval are on the scale of ``covariate_scale``),
    ``covariate_linked_error_raw`` and ``covariate_linked_error_standardized`` (the point
    estimate on each scale, without interval), ``covariate``, ``covariate_scale``,
    ``noise``, ``ceiling``, ``share_of_ceiling``,
    ``n_boot`` and ``n_undefined_max`` (largest count of undefined replicates over the
    intervals). The errors are on the 0-1 scale; the noise is the mean over the texts that
    entered of the standard deviation across runs of their run scores, min-max scaled over
    all those (text, run) rows (NaN without two runs). A status other than ``ok`` leaves
    the association metrics, the errors, the intervals and ``n_boot`` missing; the noise is
    still computed, and a constant score (``constant_score``) keeps its calibration
    shortfall of -1 and its covariate-linked error, minus the slope of gold on the
    covariate (see ``constant_score_errors``), unless the gold is constant too.

    Args:
        scores: The score frame.
        gold: Frame with ``text_id`` and the gold columns named in ``targets``.
        table: The text table supplying groups and ``n_chars``, or None.
        targets: Target key to gold column name.
        settings: Panel settings.

    Returns:
        The panel.

    Raises:
        SettingsError: If ``targets`` is empty.
        ConfigurationError: If the inputs are inconsistent: a target matching no scores,
            more than one score per text, run and target, no scored text with a gold row
            (usually ids written differently on the two sides), a missing or non-numeric
            gold or covariate column, a scored text absent from the table, no table and no
            covariate, or a reliability mapping naming a column that no target uses.
        EstimatorError: If a bootstrap metric is undefined on more than 10 percent of the
            replicates; the message starts with the target name. The error stops the whole
            panel, so leave such a target out (for example one whose binary gold has a rare
            class) to compute the others. A bootstrap that cannot run (fewer than two
            clusters among a target's used texts) raises ``ConfigurationError`` with the
            same prefix.
    """
    if not targets:
        raise SettingsError("targets must name at least one target")
    keys = list(targets)
    runs_by_key = resolve_target_scores(scores, keys)
    gold_by_id = indexed_gold(gold)
    gold_values = {
        column: numeric_column(gold_by_id, column, "gold") for column in set(targets.values())
    }
    scored_ids = sorted({text for runs in runs_by_key.values() for text in runs["text_id"]})
    check_gold_overlap(scored_ids, gold_by_id.index)
    if table is not None:
        check_scored_in_table(scored_ids, table)
    covariate_name, covariate = resolve_covariate(settings.covariate, gold_by_id, table)
    groups = resolve_groups(scored_ids, table)
    ceilings = _ceilings(targets, settings)
    results = [
        compute_target(
            key,
            targets[key],
            runs_by_key[key],
            gold_values[targets[key]],
            covariate,
            covariate_name,
            groups,
            ceilings[key],
            settings,
        )
        for key in keys
    ]
    frame = pd.DataFrame([result.row for result in results], columns=list(PANEL_COLUMNS))
    frame[list(INTEGER_COLUMNS)] = frame[list(INTEGER_COLUMNS)].astype("Int64")
    used = frozenset().union(*(result.used for result in results))
    dropped = frozenset().union(*(result.dropped for result in results))
    logger.info(
        "validity panel: %d texts used, %d dropped for missing values", len(used), len(dropped)
    )
    return ValidityPanel(
        frame=frame,
        settings=settings,
        n_used=len(used),
        n_dropped_missing=len(dropped),
        groups_defaulted=table is None or table.group_defaulted,
    )
