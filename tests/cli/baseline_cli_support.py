"""A synthetic world on disk and the arguments of ``baseline ridge`` on it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from baseline_support import GOLD, World
from similarity_support import SIDECAR


def save_world(world: World, folder: Path) -> dict[str, Path]:
    """Write the embedding sets, the texts and the gold of a synthetic world."""
    stems: dict[str, Path] = {}
    for name, embedding_set in world.embeddings.items():
        stems[name] = folder / f"emb_{name}"
        replace(embedding_set, sidecar=dict(SIDECAR)).save(stems[name])
    world.table.frame.to_csv(folder / "texts.csv", index=False)
    world.frame.to_csv(folder / "gold.csv", index=False)
    return stems


def baseline_args(stems: dict[str, Path], folder: Path, out: Path, *extra: str) -> list[str]:
    """Build the arguments of ``baseline ridge`` on a saved synthetic world."""
    embeddings = [arg for name, stem in stems.items() for arg in ("--embeddings", f"{name}={stem}")]
    return [
        *("baseline", "ridge", *embeddings, "--texts", str(folder / "texts.csv")),
        *("--gold", str(folder / "gold.csv"), "--gold-column", GOLD, "--unit-id", "text_id"),
        *("--group-column", "group_id", "--out", str(out), *extra),
    ]
