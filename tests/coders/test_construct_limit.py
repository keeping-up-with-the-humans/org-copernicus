"""`construct_scores` refuses an attribute score above the library's limit before it averages.

A finite score near the largest float can have a sum that is not finite, so the mean of a text's
attribute scores in one run would become infinite or missing, and a missing construct score
counts as a run without a score. Above `MAX_ABSOLUTE_VALUE` (1e150) a score is refused instead;
up to it, sums and squares of the scores stay inside the range of 64-bit floats.
"""

from __future__ import annotations

import math
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from copernicus.coders import MAX_ABSOLUTE_VALUE, construct_scores
from copernicus.errors import ConfigurationError

ABOVE = math.nextafter(1e150, math.inf)


def twelve_texts(first: float) -> pd.DataFrame:
    """Twelve texts with two runs of three attributes; text i scores i, but t00's run 1 ``first``.

    With ``first = 1e308`` the mean of the three scores of t00's run 1 overflows: summed in this
    order it is missing, and t00's run 2, a score of 0, would then stand for the text.
    """
    rows = [
        {"text_id": f"t{index:02d}", "run_number": run, "attribute_id": name, "value": index}
        for name in ("a", "b", "c")
        for index in range(12)
        for run in (1, 2)
    ]
    numeric = pd.DataFrame(rows).astype({"value": "float64"})
    numeric.loc[(numeric["text_id"] == "t00") & (numeric["run_number"] == 1), "value"] = first
    return numeric


def refusal(numeric: pd.DataFrame) -> str:
    """The message with which `construct_scores` refuses the table, with warnings as errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            construct_scores(numeric)
    return str(refused.value)


def test_the_limit_is_1e150() -> None:
    assert MAX_ABSOLUTE_VALUE == 1e150


def test_a_score_above_the_limit_is_refused_before_the_scores_are_averaged() -> None:
    message = refusal(twelve_texts(1e308))
    assert message == (
        "text(s) 't00' have a score above 1e+150 in absolute value (the largest is 1e+308); the "
        "library refuses a score, a gold value or a covariate above that limit, because sums and "
        "squares of larger values can overflow the range of 64-bit floats: divide the column by "
        "one constant before the call"
    )


@pytest.mark.parametrize("value", [1e150, -1e150], ids=["plus", "minus"])
def test_a_score_at_the_limit_is_averaged(value: float) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = construct_scores(twelve_texts(value)).set_index(["text_id", "run_number"])
    assert result.loc[("t00", 1), "construct_score"] == value
    assert result.loc[("t00", 1), "n_attributes_scored"] == 3


@pytest.mark.parametrize("value", [ABOVE, -ABOVE], ids=["plus", "minus"])
def test_the_next_float_above_the_limit_is_refused(value: float) -> None:
    message = refusal(twelve_texts(value))
    assert message.startswith("text(s) 't00' have a score above 1e+150 in absolute value")
    assert "(the largest is 1.0000000000000002e+150)" in message


def test_the_refusal_names_the_first_ten_texts_and_counts_the_others() -> None:
    numeric = twelve_texts(0.0).assign(value=1e200)
    shown = ", ".join(repr(f"t{index:02d}") for index in range(10))
    assert refusal(numeric).startswith(f"text(s) {shown} and 2 more have a score above 1e+150")


def three_rows(values: list[object], dtype: str) -> pd.DataFrame:
    """Text 1 with the attributes x and y in run 1, text 2 with x; the values of ``dtype``."""
    return pd.DataFrame(
        {
            "text_id": [1, 1, 2],
            "run_number": [1, 1, 1],
            "attribute_id": ["x", "y", "x"],
            "value": pd.array(values, dtype=dtype),
        }
    )


def test_a_python_integer_in_an_object_column_is_checked_as_a_float() -> None:
    # The integer 10**200 is exact in an object column and is 1e200 as a float.
    message = refusal(three_rows([10**200, None, 3], "object"))
    assert message.startswith("text(s) '1' have a score above 1e+150 in absolute value")
    assert "(the largest is 1e+200)" in message


@pytest.mark.parametrize(
    "huge",
    [Decimal("1e400"), Decimal("-1e400"), 10**399, -(10**399)],
    ids=["a decimal", "a negative decimal", "an integer of 400 digits", "a negative integer"],
)
def test_a_finite_number_beyond_the_range_of_floats_is_refused(huge: object) -> None:
    # As a float the decimal would be an infinity, which the limit leaves to its own rule, and
    # the integer cannot be turned into a float at all.
    numeric = three_rows([1, None, 3], "object")
    # Assigned as a series: a frame built from the values fails under pandas before version 3.
    numeric["value"] = pd.Series([huge, None, 3], dtype=object)
    message = refusal(numeric)
    assert message.startswith("text(s) '1' have a score above 1e+150 in absolute value")
    assert "(the largest is beyond the range of 64-bit floats)" in message


def test_a_missing_sibling_does_not_hide_a_number_beyond_the_range_of_floats() -> None:
    numeric = pd.DataFrame({"text_id": ["t", "t"], "run_number": 1})
    numeric["value"] = pd.Series([10**399, pd.NaT], dtype=object)
    message = refusal(numeric)
    assert message.startswith("text(s) 't' have a score above 1e+150 in absolute value")


@pytest.mark.parametrize("sign", [1, -1])
def test_an_infinity_of_another_package_keeps_the_rule_of_an_infinity(sign: int) -> None:
    mpmath = pytest.importorskip("mpmath")
    numeric = pd.DataFrame({"text_id": ["t"], "run_number": 1})
    numeric["value"] = pd.Series([sign * mpmath.mpf("inf")], dtype=object)
    assert construct_scores(numeric)["construct_score"].tolist() == [sign * math.inf]
    numeric["value"] = pd.Series([sign * mpmath.mpf("1e400")], dtype=object)
    assert "beyond the range of 64-bit floats" in refusal(numeric)


@pytest.mark.parametrize("kind", ["Float32", "float[pyarrow]"])
@pytest.mark.parametrize(
    "scores", [[3e38, 3e38, -math.inf], [math.inf, -math.inf]], ids=["overflow and -inf", "both"]
)
def test_32_bit_scores_with_an_infinity_keep_the_mean_they_had(
    kind: str, scores: list[float]
) -> None:
    # The mean of such a group is left as the column's own average gives it, without a warning.
    numeric = pd.DataFrame({"text_id": "t", "run_number": 1, "value": pd.array(scores, dtype=kind)})
    before = numeric.groupby(["text_id", "run_number"])["value"].mean().astype("float64")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = construct_scores(numeric)
    assert result["construct_score"].tolist() == pytest.approx(before.tolist(), nan_ok=True)
    assert math.isnan(result["construct_score"].iloc[0])
    assert result["n_attributes_scored"].tolist() == [len(scores)]


def test_an_infinite_decimal_keeps_the_rule_of_an_infinity() -> None:
    result = construct_scores(three_rows([Decimal("Infinity"), None, 3], "object"))
    assert result["construct_score"].tolist() == [math.inf, 3.0]


@pytest.mark.parametrize("kind", ["float32", "Float32", "float[pyarrow]"])
@pytest.mark.parametrize(
    ("scores", "mean"), [([3e38, 3e38], 3e38), ([3e38, 3e38, -3e38], 1e38)], ids=["two", "three"]
)
def test_scores_in_a_column_of_32_bit_floats_whose_sum_overflows_there_have_their_mean(
    kind: str, scores: list[float], mean: float
) -> None:
    # 3e38 + 3e38 is above the largest 32-bit float (about 3.4e38) and far below the limit.
    numeric = pd.DataFrame({"text_id": "t", "run_number": 1, "value": pd.array(scores, dtype=kind)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = construct_scores(numeric)
    assert result["construct_score"].dtype == np.float64
    assert result["construct_score"].tolist() == pytest.approx([mean], rel=1e-6)
    assert result["n_attributes_scored"].tolist() == [len(scores)]


def test_interleaved_rows_of_32_bit_floats_have_their_means() -> None:
    numeric = pd.DataFrame(
        {
            "text_id": ["a", "b", "a", "b", "a", "b"],
            "run_number": 1,
            "value": np.array([2e38] * 6, dtype="float32"),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = construct_scores(numeric)
    assert result["construct_score"].tolist() == pytest.approx([2e38, 2e38], rel=1e-6)


@pytest.mark.parametrize("kind", ["float32", "Float32"])
def test_means_of_32_bit_floats_that_were_finite_are_returned_bit_for_bit(kind: str) -> None:
    values = pd.array([0.1, 0.2, 0.7, None, 1e30, 3.0], dtype=kind)
    numeric = pd.DataFrame({"text_id": list("aaabbc"), "run_number": 1, "value": values})
    before = numeric.groupby(["text_id", "run_number"], sort=True)["value"].mean()
    result = construct_scores(numeric)
    assert result["construct_score"].tolist()[0] == float(before.iloc[0])
    assert result["construct_score"].tolist()[2] == float(before.iloc[2])
    assert math.isnan(result["construct_score"].tolist()[1]) is False


def test_a_nullable_integer_column_is_averaged_as_before() -> None:
    result = construct_scores(three_rows([5, None, 3], "Int64"))
    assert result["construct_score"].tolist() == [5.0, 3.0]
    assert result["n_attributes_scored"].tolist() == [1, 1]


def test_missing_and_infinite_scores_keep_their_rule() -> None:
    # A missing score is left out of the mean, and an infinite one makes the mean infinite.
    numeric = pd.DataFrame(
        {
            "text_id": ["a", "a", "a", "a", "b", "b"],
            "run_number": [1, 1, 2, 2, 1, 1],
            "attribute_id": ["x", "y", "x", "y", "x", "y"],
            "value": [math.inf, 1.0, math.nan, 2.0, -math.inf, 3.0],
        }
    )
    result = construct_scores(numeric)
    assert result["construct_score"].tolist() == [math.inf, 2.0, -math.inf]
    assert result["n_attributes_scored"].tolist() == [2, 1, 2]


def test_a_row_without_a_run_number_is_refused_before_the_limit_is_checked() -> None:
    numeric = twelve_texts(1e308)
    numeric.loc[numeric["text_id"] == "t05", "run_number"] = None
    assert refusal(numeric).startswith("score row(s) of text(s) 't05' have no run number")
