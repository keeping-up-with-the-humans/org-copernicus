"""The human-sample draw and its attachment to correction data.

A human sample is drawn by whole groups (the clusters that must not be split, such as the
writers of the texts): ``m`` of the ``G`` groups by simple random sampling without
replacement, so every text of a drawn group has the inclusion probability ``m / G``. The
design-based estimators divide by that probability, so a labelled sample without it cannot
be used by them; ``attach_draw`` is what supplies it. A draw records the fingerprint of the
texts and groups it was drawn from (``population_hash_of``), and ``attach_draw`` refuses data
with any other texts or groups, because ``m / G`` and the whole-group clustering describe that
table only. The size planner lives in ``planning``.
"""

from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from copernicus.corrections._draw_match import check_draw_matches
from copernicus.corrections._fingerprint import POPULATION_HASH_PATTERN, population_hash_of
from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.data import PI_COLUMN, CorrectionData, numeric_values
from copernicus.data.table import ONE_GROUP_PER_TEXT
from copernicus.errors import ConfigurationError, SettingsError

if TYPE_CHECKING:
    from copernicus.data import TextTable

logger = logging.getLogger(__name__)

__all__ = [
    "SampleDraw",
    "attach_draw",
    "draw_from_data",
    "draw_groups",
    "draw_human_sample",
    "groups_for_quota",
    "population_hash_of",
]

SHARE_ROUNDING_DIGITS = 9


@dataclass(frozen=True, eq=False)
class SampleDraw:
    """A drawn human sample: whole groups with their inclusion probability.

    A draw binds to the exact table it was made on: ``population_hash`` fingerprints the texts
    of that table and their groups, and ``attach_draw`` refuses data with other texts or groups.

    Attributes:
        ids: Ids of the texts to code: the texts of the drawn groups, in the order of the
            table (not sorted).
        groups: The drawn group ids, sorted.
        pi: Inclusion probability ``m / n_groups`` of every text of a drawn group; 1.0 for
            a full draw.
        m: Number of drawn groups.
        n_groups: Number of groups in the table.
        n_units: Number of analysis units in the table.
        quota: Requested number of units; for a full draw it equals ``n_units``.
        seed: Seed of the random generator.
        population_hash: ``population_hash_of`` the table's texts and groups, 64 lowercase
            hexadecimal characters.
    """

    ids: np.ndarray
    groups: np.ndarray
    pi: float
    m: int
    n_groups: int
    n_units: int
    quota: int
    seed: int
    population_hash: str

    def __post_init__(self) -> None:
        """Refuse a draw without the fingerprint of its table.

        Raises:
            ConfigurationError: If ``population_hash`` is not 64 lowercase hexadecimal
                characters, as for a draw recorded by an earlier version without one.
        """
        value = self.population_hash
        if not isinstance(value, str) or POPULATION_HASH_PATTERN.fullmatch(value) is None:
            raise ConfigurationError(
                "population_hash must be population_hash_of the texts and groups of the draw's "
                f"table, 64 lowercase hexadecimal characters; got {value!r}. A draw recorded "
                "without it was made by an earlier version: draw again"
            )


def groups_for_quota(quota_units: int, group_id: np.ndarray, unit_id: np.ndarray) -> int:
    """Number of groups to draw for a quota of units: ``min(G, ceil(quota * G / n_units))``.

    The number depends only on the structure of the pool, never on gold values, so it is
    fixed before the draw and the inclusion probability ``m / G`` is exact. The ceiling is
    computed in integer arithmetic; at least one group is drawn.

    Args:
        quota_units: Requested number of analysis units, at least 1.
        group_id: Group id of every row of the pool.
        unit_id: Analysis-unit id of every row of the pool.

    Returns:
        The number of groups ``m``, between 1 and the number of groups.

    Raises:
        SettingsError: If ``quota_units`` is below 1.
        ConfigurationError: If the pool is empty or the arrays differ in length.
    """
    if quota_units < 1:
        raise SettingsError(f"the quota must be at least 1 unit, got {quota_units}")
    groups, units = np.asarray(group_id), np.asarray(unit_id)
    if groups.shape != units.shape or groups.ndim != 1:
        raise ConfigurationError("group_id and unit_id must be one-dimensional and of equal length")
    if groups.size == 0:
        raise ConfigurationError("the pool is empty: no group and no unit to draw from")
    n_groups = int(np.unique(groups).size)
    n_units = int(np.unique(units).size)
    needed = -(-int(quota_units) * n_groups // n_units)
    return int(min(n_groups, max(1, needed)))


def _quota(n: int | None, share: float | None, n_units: int) -> int:
    """The requested number of units from exactly one of ``n`` and ``share``."""
    if (n is None) == (share is None):
        raise SettingsError("give exactly one of n and share")
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, numbers.Integral):
            raise SettingsError(f"n must be an integer, got {n!r}")
        return int(n)
    if share is None or not 0.0 < share <= 1.0:
        raise SettingsError(f"share must lie in (0, 1], got {share!r}")
    return math.ceil(round(share * n_units, SHARE_ROUNDING_DIGITS))


def draw_human_sample(
    table: TextTable,
    *,
    n: int | None = None,
    share: float | None = None,
    seed: int = 42,
    unit_id: str | None = None,
) -> SampleDraw:
    """Draw the texts a human should code: whole groups by simple random sampling.

    The quota is ``n`` units, or ``ceil(share * n_units)``. The number of groups is
    ``m = min(G, ceil(quota * G / n_units))`` (see ``groups_for_quota``); the groups are
    drawn without replacement with ``numpy.random.default_rng(seed)`` from the sorted group
    ids, and every text of a drawn group has the inclusion probability ``m / G``. Attach the
    human values to the correction data with ``attach_draw``. The draw records the fingerprint
    of the table's texts and groups and attaches only to correction data with exactly those,
    so draw from the final table that the correction data are built from.

    A quota equal to the number of units is the full draw: every group, ``pi = 1``, and no
    warning. A quota above half of the units and below the full draw is allowed but logged
    as a warning, because the battery refuses such sizes and the correction has little left
    to work on.

    Args:
        table: The text table; its ``group_id`` column defines the groups. A table built
            without a group column makes every text its own group, so the draw takes texts
            independently; the log line of the draw says so.
        n: Number of analysis units to code.
        share: Share of the analysis units to code, in ``(0, 1]``.
        seed: Seed of the generator, an integer of at least 0.
        unit_id: Name of the column that identifies the analysis unit; ``None`` counts
            every text as one unit.

    Returns:
        The draw.

    Raises:
        SettingsError: If not exactly one of ``n`` and ``share`` is given, ``n`` is not an
            integer, ``share`` is outside ``(0, 1]``, or the seed is not an integer of at
            least 0.
        ConfigurationError: If ``unit_id`` names no column of the table, or the quota is
            below 1 or above the number of units.
    """
    frame = table.frame
    if unit_id is not None and unit_id not in frame.columns:
        raise ConfigurationError(
            f"unit_id {unit_id!r} is not a column of the table; columns: "
            f"{list_first_ten(list(frame.columns))}"
        )
    units = frame["text_id" if unit_id is None else unit_id].astype(str).to_numpy()
    n_units = int(np.unique(units).size)
    quota = _quota(n, share, n_units)
    if n_units > quota > n_units / 2:
        logger.warning(
            "the quota of %d units exceeds half of the %d units; the battery refuses sizes "
            "above half the units",
            quota,
            n_units,
        )
    draw = draw_groups(
        frame["text_id"].astype(str).to_numpy(),
        frame["group_id"].astype(str).to_numpy(),
        units,
        quota,
        seed=seed,
    )
    grouping = f"; {ONE_GROUP_PER_TEXT}" if table.group_defaulted else ""
    logger.info("drew %d of %d groups (%d texts)%s", draw.m, draw.n_groups, draw.ids.size, grouping)
    return draw


def draw_groups(
    text_id: np.ndarray,
    group_id: np.ndarray,
    unit_id: np.ndarray,
    quota: int,
    *,
    seed: int,
) -> SampleDraw:
    """Draw whole groups of a pool by simple random sampling for a quota of units.

    ``m = groups_for_quota(quota, group_id, unit_id)`` of the ``G`` groups are drawn without
    replacement with ``numpy.random.default_rng(seed)`` from the sorted group ids, and every row
    of a drawn group has the inclusion probability ``m / G``. The draw is a function of the
    arrays, the quota and the seed, so the recorded seed reproduces it. ``draw_human_sample``
    draws a table this way; the repeated study draws its corpus this way too. The draw records
    ``population_hash_of(text_id, group_id)``, computed from the arrays it draws from.

    Args:
        text_id: Text id of every row of the pool.
        group_id: Group id of every row, aligned with ``text_id``.
        unit_id: Analysis-unit id of every row, aligned with ``text_id``.
        quota: Requested number of units, between 1 and the pool's unit count.
        seed: Seed of the generator, an integer of at least 0.

    Returns:
        The draw, with the ids of the drawn rows in pool order and the pool's
        ``population_hash_of(text_id, group_id)``.

    Raises:
        SettingsError: If the seed is not an integer of at least 0.
        ConfigurationError: If the arrays are not one-dimensional or differ in length, the
            pool is empty, or the quota lies outside 1 and the pool's unit count.
    """
    return _draw(text_id, group_id, unit_id, quota, seed, population_hash=None)


def draw_from_data(data: CorrectionData, quota: int, *, seed: int) -> SampleDraw:
    """Draw whole groups of the data's own texts, with the fingerprint the data already hold.

    The result equals ``draw_groups(data.ids, data.groups, data.ids, quota, seed=seed)``: the
    texts are the analysis units and ``population_hash`` is ``data.population_hash``, which is
    ``population_hash_of(data.ids, data.groups)`` computed once per data object. The
    fingerprint travels only with the arrays it describes.

    Args:
        data: The correction data to draw from; it is not modified.
        quota: Requested number of texts, between 1 and the number of texts of the data.
        seed: Seed of the generator, an integer of at least 0.

    Returns:
        The draw.

    Raises:
        SettingsError: If the seed is not an integer of at least 0.
        ConfigurationError: If the quota lies outside 1 and the number of texts.
    """
    return _draw(data.ids, data.groups, data.ids, quota, seed, population_hash=data.population_hash)


def _draw(
    text_id: np.ndarray,
    group_id: np.ndarray,
    unit_id: np.ndarray,
    quota: int,
    seed: int,
    population_hash: str | None,
) -> SampleDraw:
    """Draw the groups; ``population_hash`` is that of the arrays, or ``None`` to compute it."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise SettingsError(f"seed must be an integer of at least 0, got {seed!r}")
    texts = np.asarray(text_id).astype(str)
    groups = np.asarray(group_id).astype(str)
    units = np.asarray(unit_id).astype(str)
    if texts.ndim != 1 or texts.shape != groups.shape or texts.shape != units.shape:
        raise ConfigurationError(
            "text_id, group_id and unit_id must be one-dimensional and of equal length"
        )
    n_units = int(np.unique(units).size)
    if not 1 <= quota <= n_units:
        raise ConfigurationError(
            f"the quota of {quota} units must lie between 1 and the {n_units} units of the pool"
        )
    m = groups_for_quota(quota, groups, units)
    all_groups = np.unique(groups)
    drawn = np.sort(np.random.default_rng(int(seed)).choice(all_groups, size=m, replace=False))
    in_draw = np.isin(groups, drawn)
    return SampleDraw(
        ids=texts[in_draw],
        groups=drawn,
        pi=m / all_groups.size,
        m=m,
        n_groups=int(all_groups.size),
        n_units=n_units,
        quota=int(quota),
        seed=int(seed),
        population_hash=(
            population_hash_of(texts, groups) if population_hash is None else population_hash
        ),
    )


def attach_draw(data: CorrectionData, gold_values: pd.Series, draw: SampleDraw) -> CorrectionData:
    """Set the human values of a draw on the correction data, with the inclusion probability.

    The returned data have the gold value on the drawn texts and NaN elsewhere (values for
    other texts are ignored), and the column ``pi`` equal to ``draw.pi`` on every row.

    A draw binds to the exact table it was made on. The data must hold the same texts in the
    same groups, compared through ``population_hash`` before anything else, because
    ``pi = m / n_groups`` and the clustering by whole groups describe that table only: a text
    added, removed or moved to another group after the draw, or groups renamed (also a pure
    relabelling of the same partition), make the draw one of another table. Draw from the
    final table that the correction data are built from.

    Args:
        data: The correction data; it is not modified.
        gold_values: The human values, a series indexed by text id (ids are compared as
            strings).
        draw: The draw the values were coded for.

    Returns:
        New correction data.

    Raises:
        ConfigurationError: If the data's texts or groups differ from the draw's table (the
            message gives the draw's ``n_units`` and ``n_groups``, which count the texts
            unless the draw named a unit column, beside the texts and groups of the data, and
            the first ten drawn texts the data lack), the draw does not describe its table (a
            drawn id repeats, a drawn text or group is not in the data, a drawn text lies
            outside the drawn groups, a text of a drawn group is not among the drawn ids,
            ``m`` or ``n_groups`` differs from the groups, or ``pi`` is not ``m / n_groups``;
            the message names the first ten ids), the series repeats an id or is not numeric,
            or a drawn text has no finite value.
    """
    drawn = [str(text_id) for text_id in draw.ids]
    check_draw_matches(data, draw, drawn)
    values = pd.Series(
        numeric_values(gold_values, "gold_values"), index=gold_values.index.astype(str)
    )
    repeated = values.index[values.index.duplicated()].unique().tolist()
    if repeated:
        raise ConfigurationError(f"gold_values repeats text ids {list_first_ten(repeated)}")
    coded = values.reindex(drawn)
    unusable = [text_id for text_id, value in coded.items() if not np.isfinite(value)]
    if unusable:
        raise ConfigurationError(
            f"no finite human value for drawn texts {list_first_ten(unusable)}"
        )
    frame = data.frame.copy()
    in_draw = frame["text_id"].isin(drawn).to_numpy()
    frame[data.gold_column] = np.where(
        in_draw, values.reindex(frame["text_id"]).to_numpy(dtype=float), np.nan
    )
    frame[PI_COLUMN] = float(draw.pi)
    return replace(data, frame=frame)
