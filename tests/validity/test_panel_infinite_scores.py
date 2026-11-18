"""An infinite score of a used text stops the panel under both units, on every route."""

from __future__ import annotations

import math
import warnings

import pandas as pd
import pytest
from units_support import GOLD, N_BOOT, ROUTE_IDS, ROUTES, SIX_TEXTS, build, computed, on_route

from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.panel_inputs import resolve_target_scores

UNITS = ["text", "group"]
SIGNS = [math.inf, -math.inf]
SIGN_IDS = ["plus", "minus"]
# t1 has the runs 1 and 2, so its run 1 sits beside a finite run; t2's run 1 is its only row.
TEXTS = ["t1", "t2"]


def settings_of(unit: str) -> PanelSettings:
    """A panel of the unit with a small bootstrap."""
    return PanelSettings(n_boot=N_BOOT, unit=unit)  # type: ignore[arg-type]


def with_score(
    frame: str, target: str | None, text: str, value: float
) -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """SIX_TEXTS on a route, with ``value`` as the score of a text's run 1 in the key's rows."""
    scores, gold, table = build(SIX_TEXTS)
    routed = on_route(scores, frame)
    row = (routed["text_id"] == text) & (routed["run_number"] == 1)
    if target is not None:
        row &= routed["target_id"] == target
    routed.loc[row, "score"] = value
    return routed, gold, table


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("value", SIGNS, ids=SIGN_IDS)
@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize(("key", "frame", "target"), ROUTES, ids=ROUTE_IDS)
def test_an_infinite_score_of_a_used_text_is_refused_on_every_route(
    key, frame, target, text, value, unit
):
    # An infinite score is a broken value, not a missing one. Version 0.1.0 dropped and counted
    # such a text under the text unit.
    scores, gold, table = with_score(frame, target, text, value)
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings_of(unit))
    message = str(refused.value)
    assert f"target {key!r}" in message
    assert f"text(s) {text!r} of group 'G1' carry an infinite score" in message


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize(("key", "frame", "target"), ROUTES, ids=ROUTE_IDS)
def test_an_infinite_score_of_a_text_without_gold_is_dropped_and_counted(key, frame, target, unit):
    scores, gold, table = with_score(frame, target, "t1", math.inf)
    gold = gold[gold["text_id"] != "t1"]
    panel = validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings_of(unit))
    row = panel.frame.iloc[0]
    assert (row["n_used"], row["n_texts"], row["n_dropped_missing"]) == (5, 5, 1)


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize(("key", "frame", "target"), ROUTES, ids=ROUTE_IDS)
def test_a_missing_score_stays_admissible_on_every_route(key, frame, target, unit):
    # t1's missing run 1 leaves its run 2; nothing is dropped and nothing refused.
    scores, gold, table = with_score(frame, target, "t1", math.nan)
    panel = validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings_of(unit))
    row = panel.frame.iloc[0]
    assert (row["status"], row["n_texts"], row["n_dropped_missing"]) == ("ok", 6, 0)
    assert row["n_used"] == (6 if unit == "text" else 5)


def two_attributes() -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """SIX_TEXTS with two attributes of equal scores; t1's run 2 holds +inf in alpha, -inf in beta.

    The construct averages the two attributes of a text and run, so its run 2 of t1 is the mean
    of +inf and -inf, a missing value that would let t1 pass as a text with a missing run.
    """
    scores, gold, table = build(SIX_TEXTS)
    both = pd.concat([scores, scores.assign(attribute_id="beta")], ignore_index=True)
    run_two = (both["text_id"] == "t1") & (both["run_number"] == 2)
    both.loc[run_two & (both["attribute_id"] == "alpha"), "score"] = math.inf
    both.loc[run_two & (both["attribute_id"] == "beta"), "score"] = -math.inf
    return both, gold, table


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("key", ["alpha", "beta", "construct", "construct:construct"])
def test_opposing_infinities_of_two_attributes_are_refused(key, unit):
    scores, gold, table = two_attributes()
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings_of(unit))
    message = str(refused.value)
    assert f"target {key!r}" in message
    assert "'t1'" in message and "infinite" in message


def overflowing_construct() -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """SIX_TEXTS with two attributes; t2's only run scores 1e308 in both, a finite value.

    The construct averages the attribute scores of a text and run, and the sum of 1e308 and
    1e308 is larger than the largest float, so t2's construct score would be infinite although
    every input score is finite.
    """
    scores, gold, table = build(SIX_TEXTS)
    both = pd.concat([scores, scores.assign(attribute_id="beta")], ignore_index=True)
    both.loc[both["text_id"] == "t2", "score"] = 1e308
    return both, gold, table


def construct_missing_by_overflow() -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """SIX_TEXTS with three attributes, one attribute after the other; t2 scores 1e308 in each.

    Summed in that order, the three finite scores of t2's only run would give a missing construct
    score, not an infinite one: the sum passes the largest float at the second score, and the
    correction of the running sum then subtracts infinity from infinity.
    """
    scores, gold, table = build(SIX_TEXTS)
    others = [scores.assign(attribute_id=name) for name in ("beta", "gamma")]
    three = pd.concat([scores, *others], ignore_index=True)
    three.loc[three["text_id"] == "t2", "score"] = 1e308
    return three, gold, table


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("key", ["construct", "construct:construct"])
@pytest.mark.parametrize(
    "frames", [overflowing_construct, construct_missing_by_overflow], ids=["infinite", "missing"]
)
def test_attribute_scores_whose_construct_would_overflow_are_refused_by_the_limit(
    frames, key, unit
):
    # Version 0.1.0 dropped and counted t2 under the text unit in both cases. Scores of 1e308 are
    # above the library's limit, so the panel refuses them before the construct is averaged, and
    # with warnings turned into errors its error is what is raised.
    scores, gold, table = frames()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings_of(unit))
    message = str(refused.value)
    assert message.startswith(f"target {key!r}: text(s) 't2' have a score above 1e+150")


@pytest.mark.parametrize("unit", UNITS)
def test_an_infinite_construct_score_of_finite_attribute_scores_is_refused_as_infinite(unit):
    # A construct score that is infinite although no attribute score is: the rows that
    # compute_target receives show it, and the record of the raw rows does not hold it.
    scores, _, _ = build(SIX_TEXTS)
    runs = resolve_target_scores(scores, ["construct"])["construct"]
    runs.loc[runs["text_id"] == "t2", "score"] = math.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            computed(SIX_TEXTS, unit, runs=runs)
    assert "text(s) 't2' of group 'G1' carry an infinite score" in str(refused.value)


def adjacent_two_attributes(text: str, alpha: float, beta: float) -> tuple[pd.DataFrame, ...]:
    """SIX_TEXTS with two attributes whose rows of one text and run stand next to each other.

    The attribute values of the text's run 1 are replaced by ``alpha`` and ``beta``. With the
    rows of a group adjacent, pandas averages through ``numpy.add.reduceat``, which warns when
    ``+inf`` meets ``-inf``.
    """
    scores, gold, table = build(SIX_TEXTS)
    both = pd.concat([scores, scores.assign(attribute_id="beta")]).sort_index(kind="stable")
    both = both.reset_index(drop=True)
    run_one = (both["text_id"] == text) & (both["run_number"] == 1)
    both.loc[run_one & (both["attribute_id"] == "alpha"), "score"] = alpha
    both.loc[run_one & (both["attribute_id"] == "beta"), "score"] = beta
    return both, gold, table


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("key", ["construct", "construct:construct"])
def test_cancelling_infinities_end_in_the_library_error_when_warnings_are_errors(key, unit):
    # The construct averages only the finite attribute scores, and the panel refuses the recorded
    # infinite ones before it averages the runs, so the refusal, not numpy's warning of the
    # average of +inf and -inf, stops the panel.
    scores, gold, table = adjacent_two_attributes("t1", math.inf, -math.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings_of(unit))
    assert "text(s) 't1' of group 'G1' carry an infinite score" in str(refused.value)


@pytest.mark.parametrize("unit", UNITS)
def test_cancelling_infinities_across_runs_end_in_the_library_error(unit):
    # +inf in t1's run 1 and -inf in its run 2 of one attribute: the text's run average.
    scores, gold, table = build(SIX_TEXTS)
    scores.loc[scores["text_id"] == "t1", "score"] = [math.inf, -math.inf]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            validity_panel(scores, gold, table, targets={"alpha": GOLD}, settings=settings_of(unit))
    assert "text(s) 't1' of group 'G1' carry an infinite score" in str(refused.value)


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("key", ["alpha", "construct", "construct:construct"])
def test_cancelling_infinities_of_a_text_without_gold_are_dropped_and_counted(key, unit):
    scores, gold, table = adjacent_two_attributes("t1", math.inf, -math.inf)
    if key == "alpha":
        scores.loc[(scores["text_id"] == "t1") & (scores["attribute_id"] == "alpha"), "score"] = [
            math.inf,
            -math.inf,
        ]
    gold = gold[gold["text_id"] != "t1"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        panel = validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings_of(unit))
    row = panel.frame.iloc[0]
    assert (row["n_used"], row["n_texts"], row["n_dropped_missing"]) == (5, 5, 1)
