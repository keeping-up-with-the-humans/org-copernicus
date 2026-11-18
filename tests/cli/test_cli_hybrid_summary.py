"""What ``copernicus hybrid score`` writes and prints of the summary of its evidence texts.

The command builds the library's encoder from its options; these tests give it the model-free
encoder of the hybrid tests instead, with references made for it, so they run without torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cli_support import summary_value
from hybrid_support import STUB_DIMENSION, StubEncoder, unit_references
from typer.testing import CliRunner, Result

from copernicus.assets import load_toy_cues, toy_corpus_path
from copernicus.cli.app import app
from copernicus.coders import CodingSettings, code_texts
from copernicus.data import TextTable
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.loader import load_rubric, toy_rubric_path

N_TEXTS = 6


def strict_json(path: Path) -> Any:
    """Read a JSON file, refusing NaN and infinity, which are not JSON values."""

    def refuse(constant: str) -> Any:
        raise ValueError(f"{path.name} holds {constant}, which is not a JSON value")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Six toy texts, their two-stage store and references made for the stand-in encoder."""
    folder = tmp_path_factory.mktemp("hybrid_summary")
    rubric = load_rubric(toy_rubric_path())
    texts = folder / "texts.csv"
    corpus = TextTable.read(toy_corpus_path(), group_id="group_id")
    corpus.frame.head(N_TEXTS).to_csv(texts, index=False)
    table = TextTable.read(texts, group_id="group_id")
    store = folder / "run.sqlite"
    backend = FakeBackend(cue_responder(rubric, load_toy_cues()))
    code_texts(table, rubric, backend, store, CodingSettings(runs=1))
    references = folder / "references_stub"
    unit_references(rubric, "stub", STUB_DIMENSION).save(references)
    return {"texts": texts, "store": store, "references": references}


def score(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], out: Path, *extra: str
) -> Result:
    """Run ``hybrid score`` on the inputs with the stand-in encoder."""
    monkeypatch.setattr("copernicus.cli._embeddings.make_encoder", lambda *args: StubEncoder())
    args = [
        *("hybrid", "score", "--store", str(inputs["store"]), "--texts", str(inputs["texts"])),
        *("--rubric", str(toy_rubric_path()), "--group-column", "group_id"),
        *("--references-embeddings", str(inputs["references"]), "--out", str(out), *extra),
    ]
    return CliRunner().invoke(app, args)


def test_a_mean_of_no_evidence_text_is_written_as_null_and_printed_as_none(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = score(monkeypatch, inputs, tmp_path / "hybrid.parquet", "--min-words", "1000")
    assert result.exit_code == 0, result.stderr
    summary = strict_json(tmp_path / "hybrid.parquet.json")["summary"]
    assert (summary["n_short"], summary["n_evidence"]) == (N_TEXTS, 0)
    assert summary["mean_words_evidence"] is None
    assert summary_value(result.stdout, "mean_words_evidence") == "none"
    assert summary_value(result.stdout, "n_evidence") == "0"


def test_the_mean_of_no_evidence_text_of_each_attribute_is_written_as_null(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "hybrid.parquet"
    result = score(monkeypatch, inputs, out, "--granularity", "attribute", "--min-words", "1000")
    assert result.exit_code == 0, result.stderr
    cells = strict_json(tmp_path / "hybrid.parquet.json")["summary"]["by_attribute"]
    assert len(cells) == 3
    for cell in cells.values():
        assert (cell["n_evidence"], cell["mean_words_evidence"]) == (0, None)
    assert "mean_words_evidence" not in result.stdout


def test_a_mean_of_evidence_texts_is_written_as_a_number_and_printed_to_one_decimal(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = score(monkeypatch, inputs, tmp_path / "hybrid.parquet", "--min-words", "1")
    assert result.exit_code == 0, result.stderr
    summary = strict_json(tmp_path / "hybrid.parquet.json")["summary"]
    mean = summary["mean_words_evidence"]
    assert summary["n_evidence"] > 0 and isinstance(mean, float) and mean > 0
    assert summary_value(result.stdout, "mean_words_evidence") == f"{mean:.1f}"
