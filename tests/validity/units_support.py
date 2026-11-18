"""Hand-built frames and helpers for the tests of the validity panel's unit setting."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.panel_inputs import (
    indexed_gold,
    numeric_column,
    resolve_covariate,
    resolve_groups,
    resolve_target_scores,
)
from copernicus.validity.panel_targets import TargetResult, compute_target

# The target key is no part of the gold column's name or of any group id, so a message that
# names one of them cannot pass for naming another.
TARGET = "alpha"
GOLD = "questionnaire"
TARGETS = {TARGET: GOLD}
N_BOOT = 50
Entry = tuple[str, str, list[float], float | None]

# Nine texts in five groups, one run each; every text of a group carries its group's gold value.
NINE_TEXTS: list[Entry] = [
    ("a1", "A", [1.0], 10.0),
    ("a2", "A", [3.0], 10.0),
    ("b1", "B", [5.0], 30.0),
    ("c1", "C", [2.0], 25.0),
    ("c2", "C", [2.0], 25.0),
    ("c3", "C", [5.0], 25.0),
    ("d1", "D", [4.0], 20.0),
    ("d2", "D", [5.0], 20.0),
    ("e1", "E", [1.0], 5.0),
]

# Six texts in five groups: G1 holds t1 (run scores 1 and 3, gold 3) and t2 (run score 5, gold 3).
SIX_TEXTS: list[Entry] = [
    ("t1", "G1", [1.0, 3.0], 3.0),
    ("t2", "G1", [5.0], 3.0),
    ("u1", "G2", [3.25], 2.0),
    ("v1", "G3", [1.0], 1.0),
    ("w1", "G4", [6.0], 4.0),
    ("x1", "G5", [8.0], 5.0),
]


# The routes from a score frame to a target, as (key, frame, target id of the rows the key
# selects). The long attribute table ("attributes") holds the attribute alpha, from which the
# panel generates the construct; the frames with target_id ("targets", and "levels" with
# target_level as well) hold four targets, so that each key has to pick its own rows.
TARGET_LEVELS = {
    "alpha": "attribute",
    "beta": "attribute",
    "construct": "construct",
    "toy__rollup": "attribute",
}
ROUTES = [
    ("alpha", "attributes", None),
    ("construct", "attributes", None),
    ("construct:construct", "attributes", None),
    ("alpha", "targets", "alpha"),
    ("construct", "targets", "construct"),
    ("toy__rollup", "targets", "toy__rollup"),
    ("alpha", "levels", "alpha"),
    ("attribute:alpha", "levels", "alpha"),
    ("construct", "levels", "construct"),
    ("toy__rollup", "levels", "toy__rollup"),
]
ROUTE_IDS = [f"{frame}-{key}" for key, frame, _ in ROUTES]
KEYS_AND_FRAMES = [(key, frame) for key, frame, _ in ROUTES]


def build(entries: list[Entry]) -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """Long scores, a gold frame and a grouped text table from (text, group, run scores, gold).

    The run scores of a text are its runs 1, 2, ... in order, so an empty list gives the text
    no score row; a gold value of None gives the text no gold row. The texts differ in length,
    so the default covariate (log characters) varies.
    """
    scores = pd.DataFrame(
        [
            {"text_id": text, "run_number": run, "attribute_id": TARGET, "score": value}
            for text, _, runs, _ in entries
            for run, value in enumerate(runs, start=1)
        ]
    )
    gold = pd.DataFrame(
        [{"text_id": text, GOLD: value} for text, _, _, value in entries if value is not None]
    )
    texts = pd.DataFrame(
        {
            "text_id": [text for text, _, _, _ in entries],
            "text": [" ".join(["word"] * (2 + index)) for index in range(len(entries))],
            "group_id": [group for _, group, _, _ in entries],
        }
    )
    return scores, gold, TextTable.from_frame(texts, group_id="group_id")


def group_settings(**changes: object) -> PanelSettings:
    """Settings of a group-unit panel with a small bootstrap."""
    return PanelSettings(n_boot=N_BOOT, unit="group", **changes)  # type: ignore[arg-type]


def panel_row(entries: list[Entry], unit: str) -> pd.Series:
    """The panel row of the target under a unit, with a small bootstrap."""
    scores, gold, table = build(entries)
    settings = PanelSettings(n_boot=N_BOOT, unit=unit)  # type: ignore[arg-type]
    return validity_panel(scores, gold, table, targets=TARGETS, settings=settings).frame.iloc[0]


def refusal(entries: list[Entry], **changes: object) -> str:
    """The message of the configuration error that a group-unit panel raises."""
    scores, gold, table = build(entries)
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(scores, gold, table, targets=TARGETS, settings=group_settings(**changes))
    return str(refused.value)


def computed(entries: list[Entry], unit: str, runs: pd.DataFrame | None = None) -> TargetResult:
    """The result of `compute_target` for the target, from the inputs the panel would give it.

    The panel refuses a score, a gold value or a covariate above the library's limit before it
    calls `compute_target`, so a test of what the function does with larger values calls it
    directly. ``runs`` replaces the target's rows read from the entries.
    """
    scores, gold, table = build(entries)
    rows = resolve_target_scores(scores, [TARGET])[TARGET] if runs is None else runs
    gold_by_id = indexed_gold(gold)
    name, covariate = resolve_covariate(None, gold_by_id, table)
    groups = resolve_groups(sorted(set(rows["text_id"])), table)
    settings = PanelSettings(n_boot=N_BOOT, unit=unit)  # type: ignore[arg-type]
    gold_values = numeric_column(gold_by_id, GOLD, "gold")
    return compute_target(
        TARGET,
        GOLD,
        rows,
        gold_values,
        covariate,
        name,
        groups,
        math.nan,
        settings,
        named_groups=groups,
    )


def on_route(long: pd.DataFrame, frame: str) -> pd.DataFrame:
    """The long attribute scores of alpha, written as the score frame of a route."""
    if frame == "attributes":
        return long
    rows = long.drop(columns="attribute_id")
    copies = [
        rows.assign(target_id=name, **({"target_level": level} if frame == "levels" else {}))
        for name, level in TARGET_LEVELS.items()
    ]
    return pd.concat(copies, ignore_index=True)
