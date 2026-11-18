"""A score, a gold value or a covariate above the library's limit stops the validity panel.

Above `MAX_ABSOLUTE_VALUE` (1e150) in absolute value a score is refused before the selected
scores are averaged, and a gold value or a covariate before any metric, under both units and on
every route, because sums of such values can overflow the range of 64-bit floats: an average
that overflowed would become infinite or missing, and a missing one would leave a run or a text
out of the panel without notice. The score is checked in every row a target is built from,
whether or not its text has a gold value; the gold value and the covariate in their whole
column. Up to the limit the panel computes what it computed; a missing value and an infinity
keep their rules, and the items of the gold instrument are not covered.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from units_support import (
    GOLD,
    N_BOOT,
    ROUTE_IDS,
    ROUTES,
    SIX_TEXTS,
    TARGET,
    TARGETS,
    build,
    on_route,
)

from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.validity.panel import PanelSettings, validity_panel

UNITS = ["text", "group"]
ABOVE = math.nextafter(1e150, math.inf)
LIMIT = "above 1e+150 in absolute value"
COVARIATE = "length"
Frames = tuple[pd.DataFrame, pd.DataFrame, TextTable]


def settings_of(unit: str, **changes: object) -> PanelSettings:
    """A panel of the unit with a small bootstrap."""
    return PanelSettings(n_boot=N_BOOT, unit=unit, **changes)  # type: ignore[arg-type]


def refusal(
    frames: Frames, unit: str, targets: Mapping[str, str] | None = None, **changes: object
) -> str:
    """The message of the panel's configuration error, with warnings turned into errors.

    ``targets`` defaults to the target alpha against the gold column.
    """
    scores, gold, table = frames
    keys = TARGETS if targets is None else targets
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            validity_panel(scores, gold, table, targets=keys, settings=settings_of(unit, **changes))
    return str(refused.value)


def row_of(frames: Frames, unit: str, **changes: object) -> pd.Series:
    """The panel row of the target alpha, with warnings turned into errors."""
    scores, gold, table = frames
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        panel = validity_panel(
            scores, gold, table, targets=TARGETS, settings=settings_of(unit, **changes)
        )
    return panel.frame.iloc[0]


def with_covariate(frames: Frames, text: str, value: float) -> Frames:
    """The frames with a covariate column of the gold frame, ``value`` for one text."""
    scores, gold, table = frames
    length = np.arange(1.0, len(gold) + 1.0)
    length[(gold["text_id"] == text).to_numpy()] = value
    return scores, gold.assign(**{COVARIATE: length}), table


@pytest.mark.parametrize("unit", UNITS)
def test_a_large_attribute_triplet_is_refused_before_the_construct_is_averaged(unit: str) -> None:
    # Twelve texts with two runs of three attributes. Averaged, the three scores of 1e308 of t00's
    # run 1 would give a missing construct score, and the panel would then take t00's run 2, a
    # score of 0, for the text: a Spearman correlation of 1 where the right value is 0.5385.
    scores, gold, table = build(
        [(f"t{index:02}", f"g{index:02}", [float(index)] * 2, float(index)) for index in range(12)]
    )
    scores = pd.concat([scores.assign(attribute_id=name) for name in "abc"], ignore_index=True)
    scores.loc[(scores["text_id"] == "t00") & (scores["run_number"] == 1), "score"] = 1e308
    message = refusal((scores, gold, table), unit, {"construct": GOLD})
    assert message.startswith(f"target 'construct': text(s) 't00' have a score {LIMIT}")
    assert "(the largest is 1e+308)" in message
    assert message.endswith("divide the column by one constant before the call")


@pytest.mark.parametrize("unit", UNITS)
def test_scores_near_the_largest_float_are_refused(unit: str) -> None:
    # Twelve scores from 1e308 / 12 to 1e308, one text per group: their sum overflows, which
    # left the Pearson correlation missing with numpy's warning as the only notice.
    values = np.arange(1.0, 13.0) / 12 * 1e308
    entries = [
        (f"t{index}", f"g{index}", [value], float(index)) for index, value in enumerate(values)
    ]
    message = refusal(build(entries), unit)
    shown = ", ".join(repr(text) for text in sorted(f"t{index}" for index in range(12))[:10])
    assert message.startswith(f"target {TARGET!r}: text(s) {shown} and 2 more have a score {LIMIT}")
    assert "(the largest is 1e+308)" in message


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize(("key", "frame", "target"), ROUTES, ids=ROUTE_IDS)
def test_a_score_above_the_limit_is_refused_on_every_route(key, frame, target, unit):
    scores, gold, table = build(SIX_TEXTS)
    routed = on_route(scores, frame)
    row = (routed["text_id"] == "t1") & (routed["run_number"] == 1)
    if target is not None:
        row &= routed["target_id"] == target
    routed.loc[row, "score"] = -1e200
    message = refusal((routed, gold, table), unit, {key: GOLD})
    assert message.startswith(f"target {key!r}: text(s) 't1' have a score {LIMIT}")
    assert "(the largest is 1e+200)" in message


@pytest.mark.parametrize("unit", UNITS)
def test_a_score_of_a_text_without_a_gold_value_is_refused_too(unit: str) -> None:
    scores, gold, table = build([*SIX_TEXTS, ("z1", "G6", [1e200], None)])
    assert refusal((scores, gold, table), unit).startswith(f"target {TARGET!r}: text(s) 'z1'")


@pytest.mark.parametrize("unit", UNITS)
def test_scores_that_a_target_is_not_built_from_are_not_read(unit: str) -> None:
    # beta's large score enters the construct, which averages every attribute, but not alpha.
    scores, gold, table = build(SIX_TEXTS)
    beta = scores.assign(
        attribute_id="beta", score=scores["score"].where(scores["text_id"] != "t1", 1e200)
    )
    both = pd.concat([scores, beta], ignore_index=True)
    alone = row_of((scores, gold, table), unit)
    pd.testing.assert_series_equal(row_of((both, gold, table), unit), alone)
    message = refusal((both, gold, table), unit, {"construct": GOLD})
    assert message.startswith(f"target 'construct': text(s) 't1' have a score {LIMIT}")


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("text", ["u1", "z1"], ids=["scored", "not-scored"])
def test_a_gold_value_above_the_limit_is_refused_in_the_whole_column(unit: str, text: str) -> None:
    # z1 has a gold row and no score row; the gold column is checked as a whole.
    scores, gold, table = build([*SIX_TEXTS, ("z1", "G6", [], 2.0)])
    gold.loc[gold["text_id"] == text, GOLD] = 1e200
    message = refusal((scores, gold, table), unit)
    assert message.startswith(f"text(s) {text!r} have a gold value in column {GOLD!r} {LIMIT}")


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("place", ["gold", "table"])
def test_a_covariate_value_above_the_limit_is_refused(unit: str, place: str) -> None:
    scores, gold, table = with_covariate(build(SIX_TEXTS), "w1", -1e200)
    if place == "table":
        covariates = gold.set_index("text_id")[COVARIATE]
        table = TextTable.from_frame(
            table.frame.assign(**{COVARIATE: table.frame["text_id"].map(covariates)})
        )
        gold = gold.drop(columns=COVARIATE)
    message = refusal((scores, gold, table), unit, covariate=COVARIATE)
    assert message.startswith(
        f"text(s) 'w1' have a covariate value in column {COVARIATE!r} {LIMIT}"
    )
    assert "(the largest is 1e+200)" in message


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("value", [1e150, -1e150], ids=["plus", "minus"])
def test_values_at_the_limit_are_accepted(unit: str, value: float) -> None:
    # u1 is alone in its group, so its gold value can differ from the others of its group.
    scores, gold, table = with_covariate(build(SIX_TEXTS), "w1", value)
    scores.loc[scores["text_id"] == "x1", "score"] = value
    gold.loc[gold["text_id"] == "u1", GOLD] = value
    row = row_of((scores, gold, table), unit, covariate=COVARIATE)
    assert (row["status"], row["n_texts"], row["n_dropped_missing"]) == ("ok", 6, 0)
    assert np.isfinite(
        row[["spearman", "pearson", "mae_01", "covariate_linked_error"]].astype(float)
    ).all()


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("kind", ["score", "gold value", "covariate value"])
@pytest.mark.parametrize("value", [ABOVE, -ABOVE], ids=["plus", "minus"])
def test_the_next_float_above_the_limit_is_refused(unit: str, kind: str, value: float) -> None:
    scores, gold, table = with_covariate(
        build(SIX_TEXTS), "w1", value if kind == "covariate value" else 1.0
    )
    if kind == "score":
        scores.loc[scores["text_id"] == "x1", "score"] = value
    if kind == "gold value":
        gold.loc[gold["text_id"] == "u1", GOLD] = value
    message = refusal((scores, gold, table), unit, covariate=COVARIATE)
    assert f"have a {kind}" in message and LIMIT in message
    assert "(the largest is 1.0000000000000002e+150)" in message


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("form", ["frame", "array", "lists"])
def test_the_items_of_the_gold_instrument_are_left_to_cronbachs_alpha(unit: str, form: str) -> None:
    # The limit covers scores, gold values and covariates. The items go to `cronbach_alpha` as
    # they are given, in every form it reads.
    values = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.5], [4.0, 4.0]]
    items = {"frame": pd.DataFrame(values), "array": np.array(values), "lists": values}[form]
    scores, gold, table = build(SIX_TEXTS)
    panel = validity_panel(
        scores, gold, table, targets={"construct": GOLD}, settings=settings_of(unit, items=items)
    )
    assert panel.frame["status"].tolist() == ["ok"]
    assert 0.9 < panel.frame["ceiling"].iloc[0] <= 1.0


def test_missing_and_infinite_gold_values_and_covariates_keep_their_rules() -> None:
    # An infinite or missing gold value drops the text under both units; an infinite covariate
    # drops it under the text unit and stops the group unit with the error it gave before. Three
    # more texts keep enough groups for the bootstrap.
    more = [("y1", "G6", [2.0], 2.5), ("y2", "G7", [7.0], 4.5), ("y3", "G8", [4.0], 3.0)]
    frames = with_covariate(build([*SIX_TEXTS, *more]), "w1", math.inf)
    scores, gold, table = frames
    gold.loc[gold["text_id"] == "u1", GOLD] = math.inf
    gold.loc[gold["text_id"] == "v1", GOLD] = math.nan
    by_text = row_of(frames, "text", covariate=COVARIATE)
    assert (by_text["status"], by_text["n_used"], by_text["n_dropped_missing"]) == ("ok", 6, 3)
    message = refusal(frames, "group", covariate=COVARIATE)
    assert "covariate value is missing or not finite: text(s) 'w1' of group 'G4'" in message
    assert LIMIT not in message


@pytest.mark.parametrize("unit", UNITS)
def test_an_infinite_score_keeps_its_refusal(unit: str) -> None:
    scores, gold, table = build(SIX_TEXTS)
    scores.loc[scores["text_id"] == "x1", "score"] = -math.inf
    message = refusal((scores, gold, table), unit)
    assert "text(s) 'x1' of group 'G5' carry an infinite score value" in message
    assert LIMIT not in message
