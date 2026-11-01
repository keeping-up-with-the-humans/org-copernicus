"""Hand-built score, gold and table frames for the validity panel tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus.data import TextTable

N_TEXTS = 40
GROUP_SIZE = 4
ATTRIBUTES = ("alpha", "beta")


@dataclass(frozen=True)
class World:
    """Long scores laid out as ``export_scores`` writes them, a gold frame and a text table."""

    scores: pd.DataFrame
    gold: pd.DataFrame
    table: TextTable


def text_ids(n_texts: int = N_TEXTS) -> list[str]:
    """Return zero-padded string ids."""
    return [f"s{index:03d}" for index in range(n_texts)]


def make_table(n_texts: int = N_TEXTS) -> TextTable:
    """Return a table whose texts differ in length and whose groups hold four texts."""
    ids = text_ids(n_texts)
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "text": [" ".join(["word"] * (3 + (index * 7) % 11)) for index in range(n_texts)],
            "group_id": [f"g{index // GROUP_SIZE:02d}" for index in range(n_texts)],
        }
    )
    return TextTable.from_frame(frame)


def make_world(seed: int = 0, n_runs: int = 2, n_texts: int = N_TEXTS) -> World:
    """Return scores that track the gold values, each attribute with its own gold column."""
    rng = np.random.default_rng(seed)
    ids = text_ids(n_texts)
    gold = pd.DataFrame({"text_id": ids})
    rows = []
    for attribute in ATTRIBUTES:
        gold[f"gold_{attribute}"] = rng.integers(1, 6, size=n_texts)
        for run in range(1, n_runs + 1):
            noise = rng.integers(-1, 2, size=n_texts)
            values = np.clip(gold[f"gold_{attribute}"].to_numpy() + noise, 1, 5)
            rows.append(
                pd.DataFrame(
                    {
                        "text_id": ids,
                        "run_number": run,
                        "attribute_id": attribute,
                        "score": pd.array(values, dtype="Int64"),
                    }
                )
            )
    gold["gold_construct"] = gold[[f"gold_{a}" for a in ATTRIBUTES]].mean(axis=1)
    scores = pd.concat(rows, ignore_index=True)
    return World(scores=scores, gold=gold, table=make_table(n_texts))


def similarity_frame(world: World, seed: int = 1) -> pd.DataFrame:
    """Return a similarity-style frame: one score per text and target, no run number."""
    rng = np.random.default_rng(seed)
    parts = []
    for level, target_id, column in (
        ("construct", "construct", "gold_construct"),
        ("attribute", "alpha__rollup", "gold_alpha"),
        ("attribute", "beta__rollup", "gold_beta"),
    ):
        gold = world.gold[column].to_numpy(dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "text_id": world.gold["text_id"],
                    "target_level": level,
                    "target_id": target_id,
                    "encoder": "toy-encoder",
                    "score": gold + rng.normal(scale=0.8, size=gold.size),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
