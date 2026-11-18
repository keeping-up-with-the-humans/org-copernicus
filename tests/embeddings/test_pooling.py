"""Tests for the pooling functions."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.embed

from copernicus.embeddings.pooling import (  # noqa: E402
    POOLING_FUNCTIONS,
    cls_pool,
    last_token_pool,
    mean_pool,
)
from copernicus.errors import ConfigurationError  # noqa: E402

POOLINGS = [
    pytest.param(mean_pool, "mean pooling", "average", id="mean"),
    pytest.param(cls_pool, "cls pooling", "take", id="cls"),
    pytest.param(last_token_pool, "last-token pooling", "take", id="last_token"),
]


def test_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_cls_pool_takes_the_first_token() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    assert torch.equal(cls_pool(hidden, torch.ones(2, 3, dtype=torch.long)), hidden[:, 0])
    right = torch.tensor([[1, 1, 1], [1, 0, 0]])
    assert torch.equal(cls_pool(hidden, right), hidden[:, 0])


def test_cls_pool_takes_the_first_token_of_a_left_padded_row() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    left = torch.tensor([[1, 1, 1], [0, 0, 1]])
    assert torch.equal(cls_pool(hidden, left), torch.stack([hidden[0, 0], hidden[1, 2]]))


def test_last_token_pool_right_and_left_padding() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    right = torch.tensor([[1, 1, 1], [1, 1, 0]])
    assert torch.equal(last_token_pool(hidden, right), torch.stack([hidden[0, 2], hidden[1, 1]]))
    left = torch.tensor([[1, 1, 1], [0, 1, 1]])
    assert torch.equal(last_token_pool(hidden, left), hidden[:, -1])


def test_the_registry_of_pooling_functions_has_the_three_names() -> None:
    assert POOLING_FUNCTIONS == {"mean": mean_pool, "cls": cls_pool, "last_token": last_token_pool}


@pytest.mark.parametrize(("pool", "name", "verb"), POOLINGS)
@pytest.mark.parametrize(
    "mask",
    [
        pytest.param([[1, 1, 0], [0, 0, 0], [1, 0, 0]], id="right_padded"),
        pytest.param([[0, 1, 1], [0, 0, 0], [1, 1, 1]], id="left_padded"),
    ],
)
def test_a_row_without_an_attended_token_is_refused_by_its_position(
    pool: Callable[..., Any], name: str, verb: str, mask: list[list[int]]
) -> None:
    hidden = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
    expected = (
        f"{name}: 1 rows of the batch have an attention mask of zeros, so there is no attended "
        f"token to {verb}: rows 1 (counted from 0); give every row at least one token whose "
        "mask is 1"
    )
    with pytest.raises(ConfigurationError, match=f"^{re.escape(expected)}$"):
        pool(hidden, torch.tensor(mask))


@pytest.mark.parametrize(("pool", "name", "verb"), POOLINGS)
def test_the_refusal_names_the_first_ten_rows_and_counts_the_others(
    pool: Callable[..., Any], name: str, verb: str
) -> None:
    mask = torch.zeros((14, 2), dtype=torch.long)
    mask[[3, 7], 0] = 1
    shown = "0, 1, 2, 4, 5, 6, 8, 9, 10, 11 and 2 more"
    with pytest.raises(ConfigurationError) as refused:
        pool(torch.ones((14, 2, 3)), mask)
    assert f"{name}: 12 rows of the batch" in str(refused.value)
    assert f"rows {shown} (counted from 0)" in str(refused.value)


@pytest.mark.parametrize(("pool", "name", "verb"), POOLINGS)
def test_a_batch_without_any_token_position_is_refused(
    pool: Callable[..., Any], name: str, verb: str
) -> None:
    expected = f"{name}: 2 rows of the batch have an attention mask of zeros"
    with pytest.raises(ConfigurationError, match=f"^{re.escape(expected)}"):
        pool(torch.ones((2, 0, 3)), torch.zeros((2, 0), dtype=torch.long))


def test_an_ordinary_batch_is_pooled_exactly_as_the_formulas_say() -> None:
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 6.0], [9.0, 9.0]], [[4.0, 8.0], [5.0, 4.0], [6.0, 6.0]]]
    )
    right = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert torch.equal(mean_pool(hidden, right), torch.tensor([[2.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(last_token_pool(hidden, right), torch.stack([hidden[0, 1], hidden[1, 2]]))
    left = torch.tensor([[0, 1, 1], [1, 1, 1]])
    assert torch.equal(last_token_pool(hidden, left), hidden[:, -1])
