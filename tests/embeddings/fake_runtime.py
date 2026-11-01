"""A torch-backed fake tokenizer and model for the encoder tests; no weights are loaded."""

from __future__ import annotations

from typing import Any

import pytest
from embeddings_support import WordTokenizer

from copernicus.embeddings.encoder import EncoderRuntime

torch = pytest.importorskip("torch")

DIMENSION = 4
VOCABULARY_SIZE = 4096
PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
FAKE_VERSIONS = {
    "resolved_revision": "fake-commit",
    "torch_version": "fake-torch",
    "transformers_version": "fake-transformers",
}


class Batch(dict[str, Any]):
    """A dict of tensors with the ``to`` method of a tokenizer's batch."""

    def to(self, device: Any) -> Batch:
        return Batch({name: value.to(device) for name, value in self.items()})


class BatchTokenizer(WordTokenizer):
    """Word tokenizer that also encodes batches of texts into padded tensors.

    Padding goes on the side named by ``padding_side``; ``wrap=False`` leaves out the two
    special tokens, so the last attended token of a row is its last word.
    """

    def __init__(self, padding_side: str = "right", wrap: bool = True) -> None:
        super().__init__()
        self.padding_side = padding_side
        self.wrap = wrap

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> Batch:
        assert padding and truncation and return_tensors == "pt"
        edges = ([CLS_ID], [SEP_ID]) if self.wrap else ([], [])
        rows = [
            [*edges[0], *self.encode(text, add_special_tokens=False), *edges[1]] for text in texts
        ]
        rows = [row[:max_length] for row in rows]
        width = max(len(row) for row in rows)
        ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            places = (
                slice(0, len(row))
                if self.padding_side == "right"
                else slice(width - len(row), width)
            )
            ids[index, places] = torch.tensor(row)
            mask[index, places] = 1
        return Batch({"input_ids": ids, "attention_mask": mask})


class TableModel:
    """Looks every token id up in a fixed random table; padding vectors are garbage."""

    def __init__(self, fail_after_calls: int | None = None) -> None:
        generator = torch.Generator().manual_seed(3)
        self.table = torch.randn(VOCABULARY_SIZE, DIMENSION, generator=generator)
        self.fail_after_calls = fail_after_calls
        self.batches: list[int] = []

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        if self.fail_after_calls is not None and len(self.batches) >= self.fail_after_calls:
            raise RuntimeError("model killed")
        self.batches.append(len(input_ids))
        hidden = self.table[input_ids] + 100.0 * (attention_mask == 0).unsqueeze(-1)
        return type("Output", (), {"last_hidden_state": hidden})()


def fake_runtime(
    model: TableModel | None = None, pooling: str = "mean", tokenizer: BatchTokenizer | None = None
) -> EncoderRuntime:
    """Return a runtime that embeds with the table model on the CPU."""
    from copernicus.embeddings.pooling import POOLING_FUNCTIONS

    return EncoderRuntime(
        torch=torch,
        tokenizer=tokenizer or BatchTokenizer(),
        model=model or TableModel(),
        pool=POOLING_FUNCTIONS[pooling],
        device=torch.device("cpu"),
        versions=dict(FAKE_VERSIONS),
    )
