"""Pooling functions that turn the token vectors of a batch into one vector per row."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import Tensor

from copernicus.embeddings._pass_checks import listed
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["POOLING_FUNCTIONS", "cls_pool", "last_token_pool", "mean_pool"]

MASK_FLOOR = 1e-9


def _refuse_rows_without_token(mask: Tensor, pooling: str, verb: str) -> None:
    """Refuse a batch in which the attention mask of a row is all zero.

    Args:
        mask: Attention mask, shape ``[batch, tokens]``.
        pooling: The name of the pooling, for the message.
        verb: What the pooling does with the attended tokens, for the message.

    Raises:
        ConfigurationError: If a row has no attended token; the message counts such rows and
            names the positions of the first ten in the batch, counted from 0.
    """
    rows = torch.nonzero((mask == 0).all(dim=1)).flatten().tolist()
    if rows:
        raise ConfigurationError(
            f"{pooling}: {len(rows)} rows of the batch have an attention mask of zeros, so there "
            f"is no attended token to {verb}: rows {listed([str(row) for row in rows])} (counted "
            "from 0); give every row at least one token whose mask is 1"
        )


def mean_pool(last_hidden: Tensor, mask: Tensor) -> Tensor:
    """Average the vectors of the attended tokens.

    Args:
        last_hidden: Token vectors, shape ``[batch, tokens, dim]``.
        mask: Attention mask, shape ``[batch, tokens]``; 1 for a real token.

    Returns:
        Pooled vectors, shape ``[batch, dim]``; padding does not contribute.

    Raises:
        ConfigurationError: If the mask of a row is all zero, so that the row has no attended
            token to average; the message names the positions of the first ten such rows.
    """
    _refuse_rows_without_token(mask, "mean pooling", "average")
    expanded = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    total = torch.sum(last_hidden * expanded, dim=1)
    # A mask of zeros and ones sums to 1 or more in every row left; the floor acts only on a
    # mask of fractional weights whose row sums to less than 1e-9.
    count = torch.clamp(expanded.sum(dim=1), min=MASK_FLOOR)
    return total / count


def cls_pool(last_hidden: Tensor, mask: Tensor) -> Tensor:
    """Take the vector of the first attended token, for left or right padded batches.

    Args:
        last_hidden: Token vectors, shape ``[batch, tokens, dim]``.
        mask: Attention mask, shape ``[batch, tokens]``; 1 for a real token.

    Returns:
        Pooled vectors, shape ``[batch, dim]``: the vector at the first position of every row
        whose mask is not zero, which is position 0 in a batch padded on the right.

    Raises:
        ConfigurationError: If the mask of a row is all zero, so that the row has no attended
            token to take; the message names the positions of the first ten such rows.
    """
    _refuse_rows_without_token(mask, "cls pooling", "take")
    # Of equal values argmax returns the first position, here the first that is attended.
    first_position = (mask != 0).to(torch.long).argmax(dim=1)
    rows = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[rows, first_position]


def last_token_pool(last_hidden: Tensor, mask: Tensor) -> Tensor:
    """Take the vector of the last attended token, for left or right padded batches.

    Args:
        last_hidden: Token vectors, shape ``[batch, tokens, dim]``.
        mask: Attention mask, shape ``[batch, tokens]``; 1 for a real token.

    Returns:
        Pooled vectors, shape ``[batch, dim]``.

    Raises:
        ConfigurationError: If the mask of a row is all zero, so that the row has no attended
            token to take; the message names the positions of the first ten such rows.
    """
    _refuse_rows_without_token(mask, "last-token pooling", "take")
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
