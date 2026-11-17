"""On inputs with every run number and finite or missing scores, both units keep every value.

The scores that must be finite or missing are the input scores and those of any construct
generated from them (an average of finite attribute scores can overflow).

The pinned values are the values that the panel returned before it read run numbers exactly and
refused infinite scores; those of the text unit equal the values of version 0.1.0. A value pinned
with thirteen or more decimals is compared to twelve significant digits, because platforms differ
in the last digits of floating-point calculations, linear algebra among them; every other value
is compared exactly.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pandas as pd
import pytest
from units_support import GOLD, N_BOOT, on_route

from copernicus.data import TextTable
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.panel_targets import PANEL_COLUMNS

# (text, group, alpha's run scores, gold); beta scores alpha + 0.5, and t07 has no beta row.
TEXTS = [
    ("t01", "A", [1.0, 1.5, 0.5], 2.0),
    ("t02", "A", [2.0, math.nan], 2.0),
    ("t03", "B", [4.0, 3.5], 4.0),
    ("t04", "C", [3.0], 3.0),
    ("t05", "C", [2.5, 3.0, 2.0], 3.0),
    ("t06", "C", [3.5, 4.0], 3.0),
    ("t07", "D", [5.0, 4.5], 5.0),
    ("t08", "E", [0.5, 1.0, 1.5], 1.0),
    ("t09", "E", [2.0], None),
    ("t10", "F", [4.5, 5.0], 4.5),
    ("t11", "G", [1.5, 2.5], 1.5),
    ("t12", "G", [2.0, 2.0, 3.0], 1.5),
]
# Every field of the panel's four rows on the long attribute table, column by column, in the order
# of ROWS. On the frame with target_id the key construct names copies of alpha's rows, so both of
# that route's rows carry alpha's values.
ROWS = (("text", "alpha"), ("text", "construct"), ("group", "alpha"), ("group", "construct"))
NAN = math.nan
PINNED = {
    "target": ("alpha", "construct", "alpha", "construct"),
    "gold_column": ("questionnaire", "questionnaire", "questionnaire", "questionnaire"),
    "unit": ("text", "text", "group", "group"),
    "n_used": (11, 11, 7, 7),
    "n_texts": (11, 11, 11, 11),
    "n_dropped_missing": (1, 1, 1, 1),
    "n_runs": (3, 3, 3, 3),
    "status": ("ok", "ok", "ok", "ok"),
    "direction_applied": ("positive", "positive", "positive", "positive"),
    "n": (11, 11, 7, 7),
    "spearman": (0.9000097351498404, 0.8932931017148286, 0.9549937104572924, 0.9285714285714288),
    "rank_validity": (
        0.9000097351498404,
        0.8932931017148286,
        0.9549937104572924,
        0.9285714285714288,
    ),
    "rank_validity_low": (
        0.3917987003166699,
        0.3917987003166699,
        0.4761931187569368,
        0.4761931187569368,
    ),
    "rank_validity_high": (0.9875839974996093, 0.9692605119099177, 1.0, 1.0),
    "pearson": (0.915478441781493, 0.9074446304034032, 0.9688644350059928, 0.9627517576932227),
    "pearson_low": (0.6915661305545112, 0.6915661305545112, 0.8415456668634167, 0.8415456668634167),
    "pearson_high": (
        0.9753413354720217,
        0.9725333605290468,
        0.9947502482242282,
        0.9985183506762912,
    ),
    "kendall": (0.7921180343813393, 0.7648529270389177, 0.8783100656536799, 0.8095238095238096),
    "auc": (NAN, NAN, NAN, NAN),
    "somers_d": (NAN, NAN, NAN, NAN),
    "mae_01": (0.10429292929292928, 0.11035353535353534, 0.07142857142857147, 0.08095238095238098),
    "mae_01_low": (
        0.053552827380952385,
        0.055770502645502644,
        0.02153628117913832,
        0.01607142857142859,
    ),
    "mae_01_high": (
        0.24599242424242412,
        0.24699242424242412,
        0.15842355889724308,
        0.1594549560714974,
    ),
    "signed_gap_01": (
        0.03762626262626262,
        0.03156565656565656,
        0.03333333333333334,
        0.023809523809523815,
    ),
    "level_displacement": (
        0.03762626262626262,
        0.03156565656565656,
        0.03333333333333334,
        0.023809523809523815,
    ),
    "level_displacement_low": (
        -0.058295454545454546,
        -0.058295454545454546,
        -0.05296703296703298,
        -0.05296703296703298,
    ),
    "level_displacement_high": (
        0.14315672422815273,
        0.14120290423861848,
        0.07755378028404346,
        0.07195332080200505,
    ),
    "calibration_shortfall": (
        -0.013991769547325283,
        -0.04855967078189283,
        0.0022988505747125743,
        -0.034482758620689724,
    ),
    "calibration_shortfall_low": (
        -0.49403409090909084,
        -0.49403409090909084,
        -0.2354842254784691,
        -0.2354842254784691,
    ),
    "calibration_shortfall_high": (
        0.12193111016709124,
        0.1025593839914895,
        0.10199924024066985,
        0.05562697915292242,
    ),
    "covariate_linked_error": (
        0.10637160315806576,
        0.10360980285723148,
        0.08505952855817404,
        0.0813497957535703,
    ),
    "covariate_linked_error_low": (
        0.01116934367706483,
        -0.0062646531440402175,
        0.021991673661951275,
        0.015777603038759677,
    ),
    "covariate_linked_error_high": (
        0.21527820345502952,
        0.21527820345502952,
        0.2423424236409281,
        0.2423424236409281,
    ),
    "covariate_linked_error_raw": (
        0.15469677608463908,
        0.15068027553332392,
        0.12754126560674728,
        0.12197876102928026,
    ),
    "covariate_linked_error_standardized": (
        0.10637160315806576,
        0.10360980285723148,
        0.08505952855817404,
        0.0813497957535703,
    ),
    "covariate": ("log_n_chars", "log_n_chars", "log_n_chars", "log_n_chars"),
    "covariate_scale": ("standardized", "standardized", "standardized", "standardized"),
    "noise": (0.10367087932714245, 0.10367087932714243, 0.08507615832769314, 0.08507615832769311),
    "ceiling": (NAN, NAN, NAN, NAN),
    "share_of_ceiling": (NAN, NAN, NAN, NAN),
    "n_boot": (50, 50, 50, 50),
    "n_undefined_max": (0, 0, 0, 0),
}
RUN_TYPES = ("int64", "Int64", "object", "float64")
# A float pinned with at least PINNED_DECIMALS decimals is compared to twelve significant digits.
RELATIVE_TOLERANCE = 1e-12
PINNED_DECIMALS = 13


def complete_inputs(run_type: str) -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """The long scores of alpha and beta, the gold frame and the table, runs in ``run_type``."""
    rows = [
        (text, run, attribute, value + shift)
        for text, _, runs, _ in TEXTS
        for run, value in enumerate(runs, start=1)
        for attribute, shift in (("alpha", 0.0), ("beta", 0.5))
        if not (attribute == "beta" and text == "t07")
    ]
    scores = pd.DataFrame(rows, columns=["text_id", "run_number", "attribute_id", "score"])
    scores["run_number"] = scores["run_number"].astype(run_type)
    gold = pd.DataFrame(
        [{"text_id": text, GOLD: value} for text, _, _, value in TEXTS if value is not None]
    )
    texts = pd.DataFrame(
        {
            "text_id": [text for text, _, _, _ in TEXTS],
            "text": [" ".join(["word"] * (3 + 2 * index)) for index in range(len(TEXTS))],
            "group_id": [group for _, group, _, _ in TEXTS],
        }
    )
    return scores, gold, TextTable.from_frame(texts, group_id="group_id")


def panel_rows(run_type: str, unit: str, frame: str) -> pd.DataFrame:
    """The panel of alpha and the construct on a route (attributes or levels)."""
    scores, gold, table = complete_inputs(run_type)
    routed = (
        scores
        if frame == "attributes"
        else on_route(scores[scores["attribute_id"] == "alpha"], frame)
    )
    settings = PanelSettings(n_boot=N_BOOT, seed=3, unit=unit)  # type: ignore[arg-type]
    targets = {"alpha": GOLD, "construct": GOLD}
    return validity_panel(routed, gold, table, targets=targets, settings=settings).frame


def decimals(value: float) -> int:
    """Count the decimals of the shortest form of a finite float, also one in scientific notation.

    The shortest form is the one in which a pin is written; ``1e-13`` has thirteen decimals.
    """
    exponent = Decimal(repr(value)).as_tuple().exponent
    assert isinstance(exponent, int), value
    return max(0, -exponent)


def same_value(found: object, expected: object) -> bool:
    """Equal texts, counts and short floats, both missing, or long floats to twelve digits."""
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(found, float) and math.isnan(found)
    if isinstance(expected, float) and decimals(expected) >= PINNED_DECIMALS:
        return bool(found == pytest.approx(expected, rel=RELATIVE_TOLERANCE, abs=0.0))
    return bool(found == expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, 1), (0.5, 1), (1e20, 0), (0.0000000000001, 13), (0.0000001234567890123, 19)],
)
def test_decimals_counts_the_decimals_of_the_shortest_form_also_in_scientific_notation(
    value: float, expected: int
) -> None:
    # repr writes 0.0000000000001 as 1e-13 and 0.0000001234567890123 as 1.234567890123e-07.
    assert decimals(value) == expected


def test_only_the_two_pinned_values_of_one_are_short() -> None:
    floats = [
        value
        for column in PINNED.values()
        for value in column
        if isinstance(value, float) and not math.isnan(value)
    ]
    assert len(floats) == 96
    assert [value for value in floats if decimals(value) < PINNED_DECIMALS] == [1.0, 1.0]


@pytest.mark.parametrize("unit", ["text", "group"])
@pytest.mark.parametrize("frame", ["attributes", "levels"])
def test_complete_inputs_keep_every_value(frame, unit):
    table = panel_rows("int64", unit, frame)
    assert tuple(table.columns) == PANEL_COLUMNS and tuple(PINNED) == PANEL_COLUMNS
    assert table["target"].tolist() == ["alpha", "construct"]
    for row in table.itertuples(index=False):
        pinned_target = row.target if frame == "attributes" else "alpha"
        position = ROWS.index((unit, pinned_target))
        for column in PANEL_COLUMNS:
            expected = row.target if column == "target" else PINNED[column][position]
            found = getattr(row, column)
            assert same_value(found, expected), (row.target, column, found, expected)


@pytest.mark.parametrize("unit", ["text", "group"])
@pytest.mark.parametrize("frame", ["attributes", "levels"])
def test_the_type_of_complete_run_numbers_changes_no_value(frame, unit):
    # Integers, nullable integers, Python integers as objects and floats name the same runs.
    first = panel_rows(RUN_TYPES[0], unit, frame)
    for run_type in RUN_TYPES[1:]:
        pd.testing.assert_frame_equal(panel_rows(run_type, unit, frame), first, check_exact=True)
