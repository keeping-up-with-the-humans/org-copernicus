"""Synthetic inputs and a runner shortcut shared by the baseline runner tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from copernicus.baseline.runner import BaselineResult, BaselineSettings, run_ridge_baseline
from copernicus.data.table import TextTable
from copernicus.embeddings.sets import EmbeddingSet

N_GROUPS = 60
PER_GROUP = 4
DIM = 20
GOLD = "gold"


def make_set(ids: list[str], vectors: np.ndarray, encoder: str) -> EmbeddingSet:
    n = len(ids)
    return EmbeddingSet(
        ids=np.asarray(ids),
        vectors=vectors.astype(np.float32),
        chunk_vectors=vectors.astype(np.float32),
        chunk_owner=np.arange(n, dtype=np.int64),
        chunk_weight=np.ones(n, dtype=np.int64),
        encoder=encoder,
        metadata=None,
        sidecar={},
    )


class World:
    """Synthetic texts in groups with a planted linear signal in two embedding blocks."""

    def __init__(
        self,
        n_groups: int = N_GROUPS,
        noise: float = 1.0,
        group_sizes: Sequence[int] | None = None,
    ) -> None:
        rng = np.random.default_rng(0)
        sizes = list(group_sizes) if group_sizes is not None else [PER_GROUP] * n_groups
        n = sum(sizes)
        self.ids = [f"t{i}" for i in range(n)]
        self.groups = [f"g{group}" for group, size in enumerate(sizes) for _ in range(size)]
        first = rng.normal(size=(n, DIM)).astype(np.float32).astype(np.float64)
        second = rng.normal(size=(n, DIM)).astype(np.float32).astype(np.float64)
        beta = rng.normal(size=DIM)
        self.signal = first @ beta
        self.gold = self.signal + noise * rng.normal(size=n)
        self.embeddings = {
            "alpha": make_set(self.ids, first, "alpha"),
            "beta": make_set(self.ids, second, "beta"),
        }
        self.table = TextTable.from_frame(
            pd.DataFrame(
                {
                    "text_id": self.ids,
                    "text": [f"text number {i}" for i in range(n)],
                    "group_id": self.groups,
                }
            )
        )
        self.frame = pd.DataFrame({"text_id": self.ids, GOLD: self.gold})
        self.score = pd.Series(self.gold + 0.5 * rng.normal(size=n), index=self.ids)


def run(
    world: World,
    settings: BaselineSettings | None = None,
    *,
    embeddings: dict[str, EmbeddingSet] | None = None,
    gold: pd.DataFrame | None = None,
    table: TextTable | None = None,
    **kwargs: Any,
) -> BaselineResult:
    options: dict[str, Any] = {"gold_column": GOLD, "unit_id": "text_id", **kwargs}
    return run_ridge_baseline(
        world.embeddings if embeddings is None else embeddings,
        world.frame if gold is None else gold,
        world.table if table is None else table,
        BaselineSettings(sizes=(50, "full")) if settings is None else settings,
        **options,
    )
