"""Tests of the ``embed``, ``similarity``, and ``hybrid`` commands."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cli_support import summary_value
from embedding_support import (
    TEST_ENCODER,
    references_args,
    save_hand_built_sets,
)
from embeddings_support import needs_minilm
from similarity_support import SIDECAR, make_references, make_texts, texts_along_first_axis
from typer.testing import CliRunner

from copernicus.assets import toy_corpus_path
from copernicus.cli.app import app
from copernicus.data import TextTable
from copernicus.embeddings import EmbeddingSet
from copernicus.rubric.loader import load_rubric, toy_rubric_path
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    PAPER_GRID,
    ReferenceEmbeddings,
    SimilaritySettings,
    score_similarity,
    score_similarity_many,
)
from copernicus.similarity.scorer import OUTPUT_COLUMNS

RUBRIC = str(toy_rubric_path())
N_REFERENCE_TEXTS = 240


def score_args(texts_stem: Path, references_stem: Path, out: Path, *extra: str) -> list[str]:
    """Build the arguments of ``similarity score`` on the toy rubric."""
    return [
        *("similarity", "score", "--texts-embeddings", str(texts_stem)),
        *("--references-embeddings", str(references_stem), "--rubric", RUBRIC),
        *("--out", str(out), *extra),
    ]


@pytest.fixture(scope="module")
def minilm_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Embed six toy texts and the placeholder reference texts with MiniLM through the CLI."""
    folder = tmp_path_factory.mktemp("minilm")
    runner = CliRunner()
    table = TextTable.read(toy_corpus_path(), group_id="group_id")
    texts = folder / "texts.csv"
    table.frame.head(6).to_csv(texts, index=False)
    references = folder / "references.parquet"
    generated = runner.invoke(app, references_args(references))
    assert generated.exit_code == 0, generated.stderr
    texts_stem, references_stem = folder / "texts_minilm", folder / "references_minilm"
    embed_texts = [
        *("embed", "texts", "--texts", str(texts), "--encoder", "minilm"),
        *("--out", str(texts_stem), "--group-column", "group_id", "--device", "cpu"),
    ]
    embed_refs = [
        *("embed", "references", "--references", str(references), "--encoder", "minilm"),
        *("--out", str(references_stem), "--device", "cpu"),
    ]
    for args in (embed_texts, embed_refs):
        done = runner.invoke(app, args)
        assert done.exit_code == 0, done.stderr
    return {
        "texts": texts,
        "texts_stem": texts_stem,
        "references": references,
        "references_stem": references_stem,
    }


@needs_minilm
def test_embed_texts_writes_the_set_and_prints_the_counts(
    minilm_files: dict[str, Path],
) -> None:
    stem = minilm_files["texts_stem"]
    loaded = EmbeddingSet.load(stem, expect_encoder="minilm")
    assert len(loaded) == 6 and loaded.vectors.shape == (6, 384)
    assert loaded.metadata is not None and "group_id" in loaded.metadata.columns


@needs_minilm
def test_embed_texts_reports_paths_and_counts(
    runner: CliRunner, minilm_files: dict[str, Path], tmp_path: Path
) -> None:
    stem = tmp_path / "again"
    args = [
        *("embed", "texts", "--texts", str(minilm_files["texts"]), "--encoder", "minilm"),
        *("--out", str(stem), "--device", "cpu", "--batch-size", "4"),
        *("--checkpoint", str(tmp_path / "progress"), "--checkpoint-every", "2"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "again.npz").is_file() and (tmp_path / "again.json").is_file()
    assert summary_value(result.stdout, "texts") == "6"
    assert int(summary_value(result.stdout, "chunks")) >= 6
    assert summary_value(result.stdout, "array file") == str(tmp_path / "again.npz")
    assert summary_value(result.stdout, "description file") == str(tmp_path / "again.json")
    assert not (tmp_path / "progress.npz").exists()


@needs_minilm
def test_embed_references_writes_both_files(minilm_files: dict[str, Path]) -> None:
    stem = minilm_files["references_stem"]
    assert stem.with_name(stem.name + ".npz").is_file()
    assert stem.with_name(stem.name + ".json").is_file()
    loaded = ReferenceEmbeddings.load(stem, expect_encoder="minilm")
    assert len(loaded) == N_REFERENCE_TEXTS


def test_embed_texts_with_a_missing_text_column_is_an_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    args = [
        *("embed", "texts", "--texts", str(texts_csv), "--encoder", "minilm"),
        *("--out", str(tmp_path / "set"), "--text-column", "absent"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "absent" in result.stderr


def test_embed_texts_with_a_missing_output_folder_is_an_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    args = [
        *("embed", "texts", "--texts", str(texts_csv), "--encoder", "minilm"),
        *("--out", str(tmp_path / "absent" / "set")),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "absent" in result.stderr


def test_embed_references_without_the_description_file_is_an_error(
    runner: CliRunner, tmp_path: Path
) -> None:
    table = tmp_path / "references.parquet"
    pd.DataFrame({"text": ["a"]}).to_parquet(table)
    args = [
        *("embed", "references", "--references", str(table), "--encoder", "minilm"),
        *("--out", str(tmp_path / "set")),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "does not exist" in result.stderr


def test_similarity_score_writes_a_parquet_with_the_expected_columns(
    runner: CliRunner, tmp_path: Path
) -> None:
    texts_stem, references_stem = save_hand_built_sets(tmp_path)
    out = tmp_path / "scores.parquet"
    result = runner.invoke(app, score_args(texts_stem, references_stem, out))
    assert result.exit_code == 0, result.stderr
    frame = pd.read_parquet(out)
    assert list(frame.columns) == list(OUTPUT_COLUMNS)
    rubric = load_rubric(toy_rubric_path())
    expected = score_similarity(
        make_texts(texts_along_first_axis(4)), make_references(rubric), rubric, SimilaritySettings()
    )
    assert len(frame) == len(expected)
    assert np.allclose(frame["score"], expected["score"], atol=1e-6)
    assert summary_value(result.stdout, "rows") == str(len(frame))
    assert summary_value(result.stdout, "file") == str(out)
    sidecar = tmp_path / "scores.parquet.json"
    assert summary_value(result.stdout, "description file") == str(sidecar)
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    assert description["coder"] == "similarity"
    assert description["settings"] == [asdict(SimilaritySettings())]
    assert (description["rubric_id"], description["rubric_hash"]) == (
        rubric.rubric_id,
        rubric.content_hash(),
    )
    assert description["n_rows"] == len(frame)
    for side in ("texts", "references"):
        assert description[side]["encoder"] == TEST_ENCODER
        assert {key: description[side][key] for key in SIDECAR} == SIDECAR
    assert description["texts"]["n_texts"] == 4
    assert description["references"]["rubric_hash"] == rubric.content_hash()


def test_similarity_score_options_select_the_setting_and_csv_is_written(
    runner: CliRunner, tmp_path: Path
) -> None:
    texts_stem, references_stem = save_hand_built_sets(tmp_path)
    out = tmp_path / "scores.csv"
    options = ["--polarity", "positive", "--reduction", "max", "--writer-pooling", "max"]
    options += ["--long-text", "mean_chunk", "--target-level", "attribute"]
    result = runner.invoke(app, score_args(texts_stem, references_stem, out, *options))
    assert result.exit_code == 0, result.stderr
    frame = pd.read_csv(out)
    assert set(frame["polarity"]) == {"positive"} and set(frame["reduction"]) == {"max"}
    assert set(frame["writer_pooling"]) == {"max"} and set(frame["long_text"]) == {"mean_chunk"}
    assert set(frame["target_level"]) == {"attribute"}


def test_similarity_score_grid_paper_scores_every_setting(
    runner: CliRunner, tmp_path: Path
) -> None:
    texts_stem, references_stem = save_hand_built_sets(tmp_path)
    out = tmp_path / "grid.parquet"
    result = runner.invoke(app, score_args(texts_stem, references_stem, out, "--grid", "paper"))
    assert result.exit_code == 0, result.stderr
    frame = pd.read_parquet(out)
    rubric = load_rubric(toy_rubric_path())
    reference = score_similarity_many(
        make_texts(texts_along_first_axis(4)), make_references(rubric), rubric, PAPER_GRID
    )
    assert len(frame) == len(reference) == 144 * (len(reference) // 144)
    settings = ["polarity", "reduction", "writer_pooling", "long_text", "target_level"]
    assert len(frame[settings].drop_duplicates()) == 144
    assert "8" in result.stderr and "PAPER_KEYS" in result.stderr
    description = json.loads((tmp_path / "grid.parquet.json").read_text(encoding="utf-8"))
    assert description["settings"] == [asdict(each) for each in PAPER_GRID]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--polarity", "sideways"], "polarity"),
        (["--grid", "other"], "paper"),
        (["--grid", "paper", "--polarity", "net"], "--grid"),
    ],
)
def test_similarity_score_refuses_bad_settings(
    runner: CliRunner, tmp_path: Path, extra: list[str], message: str
) -> None:
    texts_stem, references_stem = save_hand_built_sets(tmp_path)
    out = tmp_path / "scores.parquet"
    result = runner.invoke(app, score_args(texts_stem, references_stem, out, *extra))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and message in result.stderr
    assert not out.exists()


def test_similarity_score_refuses_a_wrong_suffix_and_mismatched_encoders(
    runner: CliRunner, tmp_path: Path
) -> None:
    texts_stem, references_stem = save_hand_built_sets(tmp_path)
    result = runner.invoke(app, score_args(texts_stem, references_stem, tmp_path / "scores.txt"))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and ".parquet" in result.stderr
    rubric = load_rubric(toy_rubric_path())
    other = tmp_path / "other_refs"
    make_references(rubric, encoder="other-encoder").save(other)
    result = runner.invoke(app, score_args(texts_stem, other, tmp_path / "scores.parquet"))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and TEST_ENCODER in result.stderr


def test_similarity_score_refuses_references_of_the_rubric_before_its_rewrite(
    runner: CliRunner, tmp_path: Path, rewritten_toy_rubric: Rubric
) -> None:
    texts_stem, references_stem = save_hand_built_sets(tmp_path)
    rubric_file = tmp_path / "rewritten.json"
    rubric_file.write_text(json.dumps(rewritten_toy_rubric.to_dict()), encoding="utf-8")
    args = score_args(texts_stem, references_stem, tmp_path / "scores.parquet")
    args[args.index("--rubric") + 1] = str(rubric_file)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")
    flat = " ".join(result.stderr.split())
    assert f"but the rubric now has hash {rewritten_toy_rubric.content_hash()}" in flat
    assert not (tmp_path / "scores.parquet").exists()


@pytest.mark.parametrize(
    ("command", "defaults"),
    [
        ("similarity", ("net.", "pca.", "mean.", "max_chunk.", "construct.")),
        ("hybrid", ("net.", "pca.", "pooled.", "pooled_embedding.", "construct.")),
    ],
)
def test_the_setting_options_name_the_defaults_of_their_coder(
    runner: CliRunner, command: str, defaults: tuple[str, ...]
) -> None:
    result = runner.invoke(app, [command, "score", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.stderr
    flat = " ".join(result.stdout.replace("│", " ").split())
    for default in defaults:
        assert f"default {default}" in flat, default


@pytest.fixture(scope="module")
def coded_store(tmp_path_factory: pytest.TempPathFactory, minilm_files: dict[str, Path]) -> Path:
    """Code the six texts with the two-stage coder, through the command line."""
    store = tmp_path_factory.mktemp("store") / "run.sqlite"
    args = [
        *("code", "two-stage", "--texts", str(minilm_files["texts"]), "--rubric", RUBRIC),
        *("--out", str(store), "--backend", "fake", "--runs", "1", "--group-column", "group_id"),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.stderr
    return store


def hybrid_args(minilm_files: dict[str, Path], store: Path, out: Path, *extra: str) -> list[str]:
    """Build the arguments of ``hybrid score`` on the toy rubric."""
    return [
        *("hybrid", "score", "--store", str(store), "--texts", str(minilm_files["texts"])),
        *("--rubric", RUBRIC, "--encoder", "minilm", "--device", "cpu"),
        *("--references-embeddings", str(minilm_files["references_stem"])),
        *("--group-column", "group_id", "--out", str(out), *extra),
    ]


@needs_minilm
@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_hybrid_score_runs_at_both_granularities(
    runner: CliRunner,
    minilm_files: dict[str, Path],
    coded_store: Path,
    tmp_path: Path,
    granularity: str,
) -> None:
    out = tmp_path / "hybrid.parquet"
    args = hybrid_args(minilm_files, coded_store, out, "--granularity", granularity)
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    frame = pd.read_parquet(out)
    assert set(frame["method"]) == {f"hybrid_{granularity}"}
    assert "fallback" in frame.columns and frame["score"].notna().all()
    assert set(frame["writer_pooling"]) == {"pooled"}
    assert set(frame["long_text"]) == {"pooled_embedding"}
    assert summary_value(result.stdout, "n_texts") == "6"
    assert summary_value(result.stdout, "rows") == str(len(frame))
    assert "n_passages_embedded" in result.stdout
    description = json.loads((tmp_path / "hybrid.parquet.json").read_text(encoding="utf-8"))
    assert description["coder"] == "hybrid" and description["granularity"] == granularity
    assert (description["run_number"], description["min_words"]) == (1, 40)
    assert description["summary"]["n_texts"] == 6 and description["n_rows"] == len(frame)
    assert description["store"]["method"] == "two_stage"
    assert description["texts"]["revision"] == "unpinned"
    assert len(description["texts"]["resolved_revision"]) == 40
    assert description["references"]["encoder"] == "minilm"


@needs_minilm
def test_hybrid_score_options_and_csv_output(
    runner: CliRunner, minilm_files: dict[str, Path], coded_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "hybrid.csv"
    options = ["--polarity", "positive", "--reduction", "mean", "--min-words", "5"]
    options += ["--run-number", "1", "--writer-pooling", "max", "--long-text", "first_window"]
    result = runner.invoke(app, hybrid_args(minilm_files, coded_store, out, *options))
    assert result.exit_code == 0, result.stderr
    frame = pd.read_csv(out, keep_default_na=False)
    assert set(frame["polarity"]) == {"positive"} and set(frame["long_text"]) == {"first_window"}


@needs_minilm
def test_hybrid_score_takes_a_checkpoint_and_removes_it_when_done(
    runner: CliRunner, minilm_files: dict[str, Path], coded_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "hybrid.parquet"
    options = ["--checkpoint", str(tmp_path / "state"), "--checkpoint-every", "2"]
    result = runner.invoke(app, hybrid_args(minilm_files, coded_store, out, *options))
    assert result.exit_code == 0, result.stderr
    assert out.is_file() and not (tmp_path / "state.npz").exists()


@needs_minilm
def test_hybrid_score_refuses_a_wrong_suffix_and_another_encoders_references(
    runner: CliRunner, minilm_files: dict[str, Path], coded_store: Path, tmp_path: Path
) -> None:
    result = runner.invoke(app, hybrid_args(minilm_files, coded_store, tmp_path / "hybrid.xlsx"))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and ".parquet" in result.stderr
    _, other = save_hand_built_sets(tmp_path)
    args = hybrid_args(minilm_files, coded_store, tmp_path / "hybrid.parquet")
    args[args.index("--references-embeddings") + 1] = str(other)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "minilm" in result.stderr
