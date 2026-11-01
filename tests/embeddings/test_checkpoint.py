"""Tests of the checkpoint of an embedding pass, without a model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from copernicus.embeddings import EncoderSpec
from copernicus.embeddings.checkpoint import (
    checkpoint_file,
    fragments_fingerprint,
    read_checkpoint,
    write_checkpoint,
)

SPEC = EncoderSpec("fake", "org/fake", "mean", 6)
FRAGMENTS = ["a b", "c d e"]
COMMIT = "0123abc"


def write_two_fragments(checkpoint: Path, fingerprint: str) -> None:
    """Write a checkpoint of two finished fragments."""
    raw = np.arange(8, dtype=np.float64).reshape(2, 4)
    write_checkpoint(checkpoint, fingerprint, np.array([0, 1]), raw, np.array([2.0, 3.0]))


def test_a_checkpoint_round_trips(tmp_path: Path) -> None:
    fingerprint = fragments_fingerprint(SPEC, FRAGMENTS, resolved_revision=COMMIT)
    write_two_fragments(tmp_path / "state", fingerprint)
    saved = read_checkpoint(tmp_path / "state", fingerprint)
    assert saved is not None
    finished, raw, weight = saved
    assert finished.tolist() == [0, 1] and raw.shape == (2, 4) and weight.tolist() == [2.0, 3.0]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.npz"]


def test_a_failed_write_keeps_the_earlier_checkpoint_and_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprint = fragments_fingerprint(SPEC, FRAGMENTS, resolved_revision=COMMIT)
    write_two_fragments(tmp_path / "state", fingerprint)
    before = checkpoint_file(tmp_path / "state").read_bytes()

    def broken(handle: object, **arrays: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("copernicus.embeddings.checkpoint.np.savez", broken)
    with pytest.raises(OSError, match="disk full"):
        write_two_fragments(tmp_path / "state", fingerprint)
    assert checkpoint_file(tmp_path / "state").read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.npz"]


@pytest.mark.parametrize(
    "changed",
    [
        replace(SPEC, padding_side="left"),
        replace(SPEC, trust_remote_code=True, revision="0123456789abcdef0123456789abcdef01234567"),
        replace(SPEC, dtype="bfloat16"),
        replace(SPEC, pooling="cls"),
        replace(SPEC, revision="0123abcd"),
    ],
    ids=["padding_side", "trust_remote_code", "dtype", "pooling", "revision"],
)
def test_every_setting_that_changes_the_vectors_changes_the_fingerprint(
    changed: EncoderSpec,
) -> None:
    fingerprint = fragments_fingerprint(SPEC, FRAGMENTS, resolved_revision=COMMIT)
    assert fragments_fingerprint(changed, FRAGMENTS, resolved_revision=COMMIT) != fingerprint
    assert fragments_fingerprint(SPEC, list(FRAGMENTS), resolved_revision=COMMIT) == fingerprint


def test_the_commit_the_weights_were_loaded_from_changes_the_fingerprint() -> None:
    fingerprint = fragments_fingerprint(SPEC, FRAGMENTS, resolved_revision=COMMIT)
    moved = fragments_fingerprint(SPEC, FRAGMENTS, resolved_revision="4567def")
    unknown = fragments_fingerprint(SPEC, FRAGMENTS, resolved_revision="unknown")
    assert len({fingerprint, moved, unknown}) == 3
