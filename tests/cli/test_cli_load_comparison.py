"""Tests that the similarity command refuses, warns about and records the two loads it scores."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from similarity_support import VERSIONS, make_references, make_texts, texts_along_first_axis
from typer.testing import CliRunner

from copernicus.cli.app import app
from copernicus.rubric.loader import load_rubric, toy_rubric_path

ONE_ENVIRONMENT = "embed both sides in one environment"


def save_pair(folder: Path, references_versions: dict[str, str]) -> tuple[Path, Path]:
    """Save texts and references of one load, with some versions of the references replaced."""
    texts_stem, references_stem = folder / "texts", folder / "references"
    make_texts(texts_along_first_axis(4)).save(texts_stem)
    refs = make_references(load_rubric(toy_rubric_path()))
    replace(refs, sidecar={**refs.sidecar, **references_versions}).save(references_stem)
    return texts_stem, references_stem


def score(runner: CliRunner, stems: tuple[Path, Path], out: Path) -> tuple[int, str]:
    """Run ``similarity score`` on the toy rubric.

    Returns:
        The exit code of the command and what it wrote to standard error.
    """
    result = runner.invoke(
        app,
        [
            *("similarity", "score", "--texts-embeddings", str(stems[0])),
            *("--references-embeddings", str(stems[1]), "--rubric", str(toy_rubric_path())),
            *("--out", str(out)),
        ],
    )
    return result.exit_code, result.stderr


def description(out: Path) -> dict[str, Any]:
    """Read the description written beside the scores."""
    written: dict[str, Any] = json.loads(
        out.with_name(out.name + ".json").read_text(encoding="utf-8")
    )
    return written


def test_the_description_of_the_scores_records_the_comparison_of_the_two_loads(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "scores.parquet"
    exit_code, stderr = score(runner, save_pair(tmp_path, {}), out)
    assert exit_code == 0, stderr
    assert "WARNING" not in stderr
    assert description(out)["load_comparison"] == {
        "texts": VERSIONS,
        "references": VERSIONS,
        "warnings": [],
    }


def test_another_release_of_transformers_is_scored_with_the_warning_shown_and_recorded(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "scores.parquet"
    exit_code, stderr = score(runner, save_pair(tmp_path, {"transformers_version": "4.57.3"}), out)
    assert exit_code == 0, stderr
    flat = " ".join(stderr.split())
    assert flat.count("WARNING copernicus.similarity.poles:") == 1
    assert "transformers 4.57.6 and 4.57.3" in flat
    comparison = description(out)["load_comparison"]
    assert comparison["references"] == {**VERSIONS, "transformers_version": "4.57.3"}
    (warning,) = comparison["warnings"]
    assert "transformers 4.57.6 and 4.57.3" in warning and ONE_ENVIRONMENT in warning


def test_references_of_another_major_version_of_transformers_are_refused(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "scores.parquet"
    exit_code, stderr = score(runner, save_pair(tmp_path, {"transformers_version": "5.12.1"}), out)
    assert exit_code == 1
    assert stderr.startswith("error:")
    flat = " ".join(stderr.split())
    assert "transformers 4.57.6 and 5.12.1" in flat and flat.endswith(ONE_ENVIRONMENT)
    assert not out.exists()
