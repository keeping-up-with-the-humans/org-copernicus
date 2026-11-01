"""Pooling functions that turn the token vectors of a batch into one vector per row."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["POOLING_FUNCTIONS", "cls_pool", "last_token_pool", "mean_pool"]

MASK_FLOOR = 1e-9


def mean_pool(last_hidden: Tensor, mask: Tensor) -> Tensor:
    """Average the vectors of the attended tokens.

    Args:
        last_hidden: Token vectors, shape ``[batch, tokens, dim]``.
        mask: Attention mask, shape ``[batch, tokens]``; 1 for a real token.

    Returns:
        Pooled vectors, shape ``[batch, dim]``; padding does not contribute.
    """
    expanded = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    total = torch.sum(last_hidden * expanded, dim=1)
    count = torch.clamp(expanded.sum(dim=1), min=MASK_FLOOR)
    return total / count


def cls_pool(last_hidden: Tensor, mask: Tensor) -> Tensor:
    """Take the vector of the first token.

    Args:
        last_hidden: Token vectors, shape ``[batch, tokens, dim]``.
        mask: Attention mask, shape ``[batch, tokens]``; not read.

    Returns:
        Pooled vectors, shape ``[batch, dim]``.
    """
    return last_hidden[:, 0]


def last_token_pool(last_hidden: Tensor, mask: Tensor) -> Tensor:
    """Take the vector of the last attended token, for left or right padded batches.

    Args:
        last_hidden: Token vectors, shape ``[batch, tokens, dim]``.
        mask: Attention mask, shape ``[batch, tokens]``; 1 for a real token.

    Returns:
        Pooled vectors, shape ``[batch, dim]``.
    """
    left_padded = bool(mask[:, -1].sum() == mask.shape[0])
    if left_padded:
        return last_hidden[:, -1]
    last_position = mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[rows, last_position]


POOLING_FUNCTIONS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "mean": mean_pool,
    "cls": cls_pool,
    "last_token": last_token_pool,
}
