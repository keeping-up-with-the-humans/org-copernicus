"""Tests for the pooling functions."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.embed

from copernicus.embeddings.pooling import (  # noqa: E402
    POOLING_FUNCTIONS,
    cls_pool,
    last_token_pool,
    mean_pool,
)


def test_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_cls_pool_takes_the_first_token() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    assert torch.equal(cls_pool(hidden, torch.ones(2, 3, dtype=torch.long)), hidden[:, 0])


def test_last_token_pool_right_and_left_padding() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    right = torch.tensor([[1, 1, 1], [1, 1, 0]])
    assert torch.equal(last_token_pool(hidden, right), torch.stack([hidden[0, 2], hidden[1, 1]]))
    left = torch.tensor([[1, 1, 1], [0, 1, 1]])
    assert torch.equal(last_token_pool(hidden, left), hidden[:, -1])


def test_the_registry_of_pooling_functions_has_the_three_names() -> None:
    assert POOLING_FUNCTIONS == {"mean": mean_pool, "cls": cls_pool, "last_token": last_token_pool}
