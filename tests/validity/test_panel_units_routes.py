"""Both units on every route from a score frame to a target: rows and run numbers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from units_support import (
    GOLD,
    KEYS_AND_FRAMES,
    N_BOOT,
    ROUTE_IDS,
    ROUTES,
    SIX_TEXTS,
    build,
    group_settings,
    on_route,
)

from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.panel_scores import resolve_targets

# The run of a text whose run number is left missing: t1's run 2 sits beside its numbered run
# 1, and t2's run 1 is the only row it has.
UNNUMBERED = [("t1", 2), ("t2", 1)]
UNNUMBERED_IDS = ["beside-a-numbered-run", "only-row"]
# 2**53 and 2**53 + 1 are two integers but one 64-bit float.
LARGE_RUN = 2**53
UNITS = ["text", "group"]


def route_inputs(
    frame: str, target: str | None, text: str, run: int
) -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """SIX_TEXTS on a route, with the run number of one text's run missing in the key's rows."""
    scores, gold, table = build(SIX_TEXTS)
    routed = on_route(scores, frame).astype({"run_number": float})
    row = (routed["text_id"] == text) & (routed["run_number"] == run)
    if target is not None:
        row &= routed["target_id"] == target
    routed.loc[row, "run_number"] = np.nan
    return routed, gold, table


@pytest.mark.parametrize(("text", "run"), UNNUMBERED, ids=UNNUMBERED_IDS)
@pytest.mark.parametrize(("key", "frame", "target"), ROUTES, ids=ROUTE_IDS)
def test_a_used_score_row_without_a_run_number_is_refused_on_every_route(
    key, frame, target, text, run
):
    scores, gold, table = route_inputs(frame, target, text, run)
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(scores, gold, table, targets={key: GOLD}, settings=group_settings())
    message = str(refused.value)
    assert f"target {key!r}" in message
    assert f"{text!r} of group 'G1'" in message
    assert "have no run number" in message


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("gold_row", [False, True], ids=["no-gold-row", "missing-gold-value"])
@pytest.mark.parametrize(("text", "run"), UNNUMBERED, ids=UNNUMBERED_IDS)
@pytest.mark.parametrize(("key", "frame", "target"), ROUTES, ids=ROUTE_IDS)
def test_an_unnumbered_row_of_a_text_without_gold_is_dropped_and_counted_on_every_route(
    key, frame, target, text, run, gold_row, unit
):
    # The text has no gold value, so none of its rows is used: it is dropped and counted, and
    # under the group unit G1 is the other text alone. On the generated construct this holds
    # for t2 too, whose only row the unnumbered one is.
    scores, gold, table = route_inputs(frame, target, text, run)
    if gold_row:
        gold = gold.assign(**{GOLD: gold[GOLD].where(gold["text_id"] != text)})
    else:
        gold = gold[gold["text_id"] != text]
    settings = PanelSettings(n_boot=N_BOOT, unit=unit)
    panel = validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings)
    row = panel.frame.iloc[0]
    assert (row["n_used"], row["n_texts"], row["n_dropped_missing"]) == (5, 5, 1)
    assert (panel.n_used, panel.n_dropped_missing) == (5, 1)


@pytest.mark.parametrize(("text", "run"), UNNUMBERED, ids=UNNUMBERED_IDS)
@pytest.mark.parametrize(("key", "frame", "target"), ROUTES, ids=ROUTE_IDS)
def test_the_text_unit_refuses_a_used_score_row_without_a_run_number_on_every_route(
    key, frame, target, text, run
):
    # Also on the construct generated from a long attribute table, which version 0.1.0 built
    # without the row, so that t2, whose only row it is, left the target without a count.
    scores, gold, table = route_inputs(frame, target, text, run)
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(
            scores, gold, table, targets={key: GOLD}, settings=PanelSettings(n_boot=N_BOOT)
        )
    message = str(refused.value)
    assert f"target {key!r}" in message
    assert f"text(s) {text!r} of group 'G1' have no run number" in message


@pytest.mark.parametrize("table_kind", ["none", "without-groups"])
def test_without_groups_of_a_table_the_text_unit_names_the_text_only(table_kind):
    scores, gold, table = route_inputs("attributes", None, "t1", 2)
    gold = gold.assign(size=[float(index) for index in range(len(gold))])
    plain = None if table_kind == "none" else TextTable.from_frame(table.frame[["text_id", "text"]])
    settings = PanelSettings(n_boot=N_BOOT, covariate="size")
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(scores, gold, plain, targets={"alpha": GOLD}, settings=settings)
    assert "text(s) 't1' have no run number" in str(refused.value)


def large_runs(extra_row: bool) -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """Thirteen groups of one text; text t<i> scores i in run 2**53 and i + 2 in run 2**53 + 1.

    The run numbers are Python integers in a column of objects, so they are two numbers when
    they reach the panel. t12 has no gold value; with ``extra_row`` it also has a score row
    without a run number.
    """
    ids = [f"t{index}" for index in range(13)]
    texts = pd.DataFrame(
        {
            "text_id": ids,
            "text": [" ".join(["word"] * (index + 1)) for index in range(13)],
            "group_id": [f"G{index}" for index in range(13)],
        }
    )
    gold = pd.DataFrame(
        {"text_id": ids, GOLD: [*range(12), math.nan], "size": [float(i) for i in range(13)]}
    )
    pairs = [(index, run) for index in range(13) for run in (0, 1)]
    scores = pd.DataFrame(
        {
            "text_id": [f"t{index}" for index, _ in pairs],
            "run_number": pd.Series([LARGE_RUN + run for _, run in pairs], dtype=object),
            "attribute_id": "alpha",
            "score": [float(index + 2 * run) for index, run in pairs],
        }
    )
    if extra_row:
        extra = {"text_id": ["t12"], "run_number": pd.Series([None], dtype=object)}
        extra |= {"attribute_id": ["alpha"], "score": [0.0]}
        scores = pd.concat([scores, pd.DataFrame(extra)], ignore_index=True)
    return scores, gold, TextTable.from_frame(texts, group_id="group_id")


@pytest.mark.parametrize(("key", "frame"), KEYS_AND_FRAMES, ids=ROUTE_IDS)
def test_large_run_numbers_stay_apart_beside_an_unnumbered_row_of_an_excluded_text(key, frame):
    # Every used group has its two runs, with the means i and i + 2, whose standard deviation is
    # sqrt(2); the means span 0 to 13 over the groups, so the noise is sqrt(2) / 13. t12 is
    # dropped for its missing gold value, so its row without a run number must change nothing,
    # not even the two run numbers that one 64-bit float cannot tell apart.
    plain, gold, table = large_runs(extra_row=False)
    extra, _, _ = large_runs(extra_row=True)
    settings = group_settings(covariate="size")
    routed = on_route(extra, frame)
    panel = validity_panel(routed, gold, table, targets={key: GOLD}, settings=settings)
    row = panel.frame.iloc[0]
    assert (row["n_used"], row["n_dropped_missing"], row["n_runs"]) == (12, 1, 2)
    assert row["noise"] == pytest.approx(math.sqrt(2) / 13, abs=1e-12)
    without = validity_panel(
        on_route(plain, frame), gold, table, targets={key: GOLD}, settings=settings
    )
    pd.testing.assert_frame_equal(panel.frame, without.frame)
    runs = resolve_targets(routed, [key])[key].runs
    assert set(runs["run_number"].dropna()) == {LARGE_RUN, LARGE_RUN + 1}


def nullable_runs(dtype: str) -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """Group A holds a (runs L and L + 2) and b (runs L + 1 and L + 2), with L = 2**53.

    Eleven groups of one text c<i> score i + 1 in the runs L and L + 2, and the text excluded has
    no gold value and one score row without a run number. The run numbers are built from Python
    integers in a column of ``dtype``, so they are exact when they reach the panel.
    """
    records = [("a", LARGE_RUN, 0.0), ("a", LARGE_RUN + 2, 4.0)]
    records += [("b", LARGE_RUN + 1, 2.0), ("b", LARGE_RUN + 2, 6.0)]
    records += [
        (f"c{i}", run, float(i + 1)) for i in range(11) for run in (LARGE_RUN, LARGE_RUN + 2)
    ]
    records += [("excluded", None, 0.0)]
    ids = ["a", "b", *[f"c{i}" for i in range(11)], "excluded"]
    scores = pd.DataFrame(
        {
            "text_id": [text for text, _, _ in records],
            "run_number": pd.Series([run for _, run, _ in records], dtype=dtype),
            "attribute_id": "alpha",
            "score": [value for _, _, value in records],
        }
    )
    texts = pd.DataFrame(
        {
            "text_id": ids,
            "text": ["a valid text"] * len(ids),
            "group_id": ["A", "A", *[f"G{i}" for i in range(11)], "X"],
        }
    )
    gold = pd.DataFrame(
        {
            "text_id": ids,
            GOLD: [0.0, 0.0, *[float(i + 1) for i in range(11)], math.nan],
            "size": [float(index) for index in range(len(ids))],
        }
    )
    return scores, gold, TextTable.from_frame(texts, group_id="group_id")


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize(("key", "frame"), KEYS_AND_FRAMES, ids=ROUTE_IDS)
def test_nullable_integer_run_numbers_keep_their_value_on_every_route(key, frame, unit):
    # Read as 64-bit floats, L + 1 would become L: three runs would count as two, and group A
    # would gain a complete run L that its texts do not share. A's only common run is L + 2, so
    # it adds no standard deviation, every other group is constant, and the group noise is 0.
    settings = PanelSettings(n_boot=N_BOOT, unit=unit, covariate="size")
    rows = {}
    for dtype in ("Int64", "object"):
        scores, gold, table = nullable_runs(dtype)
        panel = validity_panel(
            on_route(scores, frame), gold, table, targets={key: GOLD}, settings=settings
        )
        rows[dtype] = panel.frame
    row = rows["Int64"].iloc[0]
    assert (row["n_used"], row["n_dropped_missing"], row["n_runs"]) == (
        12 if unit == "group" else 13,
        1,
        3,
    )
    if unit == "group":
        assert row["noise"] == 0.0
    pd.testing.assert_frame_equal(rows["Int64"], rows["object"])
