"""The checks of an embedding pass on its windows and vectors, without torch or a model."""

from __future__ import annotations

import re

import numpy as np
import pytest

from copernicus.embeddings._pass_checks import (
    NORM_FLOOR,
    refuse_windows_without_token,
    unit_rows,
)
from copernicus.errors import ConfigurationError

TEXT_REFUSAL = (
    "{count} texts have a vector of length below 1e-12, which the encoder does not scale to unit "
    "length: {named}; leave these texts out or embed them with another encoder"
)
WINDOW_REFUSAL = (
    "{count} texts have a window whose pooled vector has a length below 1e-12, which the "
    "encoder does not scale to unit length: {named}; leave these texts out or embed them with "
    "another encoder"
)


def whole(template: str, count: int, named: str) -> str:
    """The pattern of a whole refusal."""
    return f"^{re.escape(template.format(count=count, named=named))}$"


@pytest.mark.parametrize(("windows", "template"), [(False, TEXT_REFUSAL), (True, WINDOW_REFUSAL)])
def test_rows_shorter_than_the_floor_are_refused_naming_each_text_once(
    windows: bool, template: str
) -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 0.0], [5e-13, 0.0], [NORM_FLOOR, 0.0], [0.0, 2.0]])
    with pytest.raises(ConfigurationError, match=whole(template, 1, "'b'")):
        unit_rows(matrix, ["a", "b", "b", "c", "d"], windows=windows)
    with pytest.raises(ConfigurationError, match=whole(template, 2, "'b', 'x'")):
        unit_rows(matrix, ["a", "b", "x", "c", "d"], windows=windows)


def test_the_refusal_names_the_first_ten_texts_and_counts_the_others() -> None:
    named = ", ".join(f"'t{index}'" for index in range(10)) + " and 2 more"
    with pytest.raises(ConfigurationError, match=whole(TEXT_REFUSAL, 12, named)):
        unit_rows(np.zeros((12, 3)), [f"t{index}" for index in range(12)], windows=False)


def test_a_row_at_the_floor_is_scaled_and_every_row_is_divided_by_its_length() -> None:
    matrix = np.asarray([[NORM_FLOOR, 0.0], [3.0, 4.0], [1.0, 1.0], [-2.5, 0.1]])
    assert np.linalg.norm(matrix[:1], axis=1)[0] == NORM_FLOOR
    expected = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    scaled = unit_rows(matrix, ["a", "b", "c", "d"], windows=False)
    assert scaled.dtype == np.float64 and np.array_equal(scaled, expected)
    assert scaled[0].tolist() == [1.0, 0.0]
    below = np.asarray([[np.nextafter(NORM_FLOOR, 0.0), 0.0]])
    with pytest.raises(ConfigurationError, match=whole(TEXT_REFUSAL, 1, "'a'")):
        unit_rows(below, ["a"], windows=False)


@pytest.mark.parametrize(
    ("pooling", "rule"),
    [
        ("mean", "so mean pooling has no token vector to average there"),
        ("last_token", "so last-token pooling has no last token to take there"),
    ],
)
def test_a_window_without_a_token_is_refused_with_the_rule_of_its_pooling(
    pooling: str, rule: str
) -> None:
    message = (
        f"2 texts have a window that the tokenizer turned into no token, {rule}: 'a', 'b'; leave "
        "these texts out, or change them so that the tokenizer gives every window a token"
    )
    with pytest.raises(ConfigurationError, match=f"^{re.escape(message)}$"):
        refuse_windows_without_token(pooling, ["a", "a", "b"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e200])
@pytest.mark.parametrize(
    ("windows", "what"),
    [(False, "a vector"), (True, "a window whose pooled vector has a length")],
    ids=["text", "window"],
)
def test_a_row_whose_length_is_not_finite_is_refused_naming_its_text(
    value: float, windows: bool, what: str
) -> None:
    # 1e200 is finite, but its square overflows, so the computed length is infinite.
    matrix = np.asarray([[3.0, 4.0], [value, 1.0], [1.0, 0.0], [0.0, value]])
    with pytest.raises(ConfigurationError) as refused:
        unit_rows(matrix, ["a", "b", "c", "b"], windows=windows)
    message = str(refused.value)
    assert message.startswith(f"1 texts have {what}")
    assert "that is not finite" in message and message.count("'b'") == 1
    assert "or the sum of its squares overflows" in message
    assert message.endswith("leave these texts out or embed them with another encoder")


def test_a_sum_of_finite_squares_that_overflows_is_refused() -> None:
    # Each square is finite (6.4e307); their sum is not.
    row = np.full((1, 4), 8e153)
    assert np.isfinite(row**2).all()
    with pytest.raises(ConfigurationError, match="that is not finite"):
        unit_rows(row, ["a"], windows=True)
