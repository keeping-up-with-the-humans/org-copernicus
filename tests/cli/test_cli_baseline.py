"""Tests of the ``baseline ridge`` command."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from baseline_support import GOLD, World
from cli_support import option_defaults, summary_value
from embeddings_support import needs_minilm
from similarity_support import SIDECAR
from typer.testing import CliRunner

from copernicus._version import __version__
from copernicus.assets import toy_corpus_path
from copernicus.baseline import BaselineSettings
from copernicus.cli.app import app

TOY = toy_corpus_path()
FLOOR = "10"


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


@pytest.fixture(scope="module")
def saved_world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Path]]:
    folder = tmp_path_factory.mktemp("world")
    return folder, save_world(World(), folder)


def test_baseline_on_a_synthetic_set_writes_the_four_tables(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    out = tmp_path / "results.csv"
    args = baseline_args(
        stems,
        folder,
        out,
        *("--sizes", "50,full", "--seeds", "42,43", "--k-folds", "3"),
        *("--encoder-set", "alpha", "--encoder-set", "stacked"),
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    results = pd.read_csv(out)
    assert set(results["encoder_set"]) == {"alpha", "stacked"}
    assert set(results["size"].astype(str)) == {"50", "full"}
    assert results["spearman_abs"].between(0, 1).all()
    for suffix in ("oof", "skipped", "curves"):
        assert (tmp_path / f"results_{suffix}.csv").is_file()
    oof = pd.read_csv(tmp_path / "results_oof.csv")
    assert set(oof["size"].astype(str)) == {"full"}
    sidecar = tmp_path / "results.csv.json"
    assert summary_value(result.stdout, "description file") == str(sidecar)
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    settings = BaselineSettings(
        k_folds=3, seeds=(42, 43), sizes=(50, "full"), encoder_sets=("alpha", "stacked")
    )
    assert description["analysis"] == "NLP-ML baseline"
    assert description["settings"] == json.loads(json.dumps(asdict(settings)))
    assert (description["gold_column"], description["unit_id"]) == (GOLD, "text_id")
    assert description["binary"] is False
    assert description["inputs"] == {
        "texts": str(folder / "texts.csv"),
        "gold": str(folder / "gold.csv"),
        "coder_scores": None,
        "coder_score_column": None,
    }
    for name in ("alpha", "beta"):
        embedded = description["embeddings"][name]
        assert embedded["stem"] == str(stems[name]) and embedded["encoder"] == name
        assert {key: embedded[key] for key in SIDECAR} == SIDECAR
        assert embedded["n_texts"] == 240
    assert description["tables"] == {
        name: str(tmp_path / f"results_{name}.csv") for name in ("oof", "skipped", "curves")
    }
    assert description["library_version"] == __version__
    curves = pd.read_csv(tmp_path / "results_curves.csv")
    assert len(curves) == 4
    assert result.stdout.count("rank validity") == len(curves)
    assert summary_value(result.stdout, "skipped") == "0"


def test_baseline_parquet_tables_round_trip_with_an_integer_size_and_full(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    out = tmp_path / "results.parquet"
    args = baseline_args(
        {"alpha": stems["alpha"]},
        folder,
        out,
        "--sizes",
        "50,full",
        "--seeds",
        "42",
        "--k-folds",
        "3",
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    results = pd.read_parquet(out)
    assert list(results["size"]) == ["50", "full"]
    assert results["size_units"].dtype == "Int64"
    assert results["size_units"].iloc[0] == 50 and pd.isna(results["size_units"].iloc[1])
    oof = pd.read_parquet(tmp_path / "results_oof.parquet")
    assert set(oof["size"]) == {"full"} and oof["size_units"].isna().all()
    assert (oof["alpha"] > 0).all() and oof["at_grid_edge"].dtype == bool
    skipped = pd.read_parquet(tmp_path / "results_skipped.parquet")
    assert skipped.empty and list(skipped.columns) == ["seed", "size", "size_units", "reason"]
    curves = pd.read_parquet(tmp_path / "results_curves.parquet")
    assert list(curves["size"]) == ["50", "full"]
    assert curves["size_units"].dtype == "Int64"
    assert np.isfinite(curves[["rank_validity", "spearman"]].to_numpy()).all()


def test_baseline_defaults_are_the_settings_defaults() -> None:
    settings = BaselineSettings()
    defaults = option_defaults("baseline", "ridge")
    assert defaults["k_folds"] == settings.k_folds
    assert defaults["floor_units"] == settings.floor_units
    assert defaults["seeds"] == ",".join(str(seed) for seed in settings.seeds)
    assert defaults["sizes"] == ",".join(str(size) for size in settings.sizes)
    assert defaults["binary"] is None


def test_baseline_records_skipped_sizes_and_adds_the_coder_score(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    world = World()
    scores = pd.DataFrame({"text_id": world.ids, "score": world.score.to_numpy()})
    scores_path = tmp_path / "coder.csv"
    scores.to_csv(scores_path, index=False)
    out = tmp_path / "results.parquet"
    args = baseline_args(
        {"alpha": stems["alpha"]},
        folder,
        out,
        *("--sizes", "500,full", "--seeds", "42", "--k-folds", "3"),
        *("--coder-scores", str(scores_path), "--no-binary"),
    )
    skipped_only = runner.invoke(app, args)
    assert skipped_only.exit_code == 0, skipped_only.stderr
    skipped = pd.read_parquet(tmp_path / "results_skipped.parquet")
    assert list(skipped["size"]) == ["500"] and list(skipped["size_units"]) == [500]
    assert skipped.loc[0, "reason"].startswith("quota_exceeds_half_units")
    assert summary_value(skipped_only.stdout, "skipped") == "1"
    results = pd.read_parquet(out)
    assert set(results["variant"]) == {"embeddings_only", "embeddings_plus_score"}


def test_baseline_with_binary_set_scores_a_gold_coded_one_and_two(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    world = World()
    coded = np.where(world.gold > np.median(world.gold), 2, 1)
    gold_path = tmp_path / "binary_gold.csv"
    pd.DataFrame({"text_id": world.ids, GOLD: coded}).to_csv(gold_path, index=False)
    extra = ("--sizes", "full", "--seeds", "42", "--k-folds", "3", "--binary")
    out = tmp_path / "binary.csv"
    args = baseline_args({"alpha": stems["alpha"]}, folder, out, *extra)
    args[args.index(str(folder / "gold.csv"))] = str(gold_path)
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert (pd.read_csv(out)["auc"] > 0.8).all()
    description = json.loads((tmp_path / "binary.csv.json").read_text(encoding="utf-8"))
    assert description["binary"] is True and description["settings"]["binary"] is True
    continuous = baseline_args({"alpha": stems["alpha"]}, folder, tmp_path / "c.csv", *extra)
    refused = runner.invoke(app, continuous)
    assert refused.exit_code == 1 and "binary is set" in refused.stderr
    assert not (tmp_path / "c.csv").exists()


def test_baseline_refuses_a_coder_score_column_that_is_not_numeric(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    labels = pd.DataFrame({"text_id": World().ids, "rating": "high"})
    scores_path = tmp_path / "labels.csv"
    labels.to_csv(scores_path, index=False)
    extra = ("--sizes", "full", "--coder-scores", str(scores_path), "--coder-score-column")
    args = baseline_args({"alpha": stems["alpha"]}, folder, tmp_path / "r.csv", *extra, "rating")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "coder score column 'rating'" in result.stderr and "numeric" in result.stderr


def test_baseline_with_every_size_skipped_still_writes_the_tables(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    out = tmp_path / "results.csv"
    args = baseline_args(
        {"alpha": stems["alpha"]}, folder, out, "--sizes", "500", "--seeds", "42", "--k-folds", "3"
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert summary_value(result.stdout, "results") == "0"
    assert summary_value(result.stdout, "skipped") == "1"
    assert len(pd.read_csv(tmp_path / "results_skipped.csv")) == 1
    assert "rank validity" not in result.stdout


def test_baseline_refuses_an_embeddings_value_without_equals(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    args = baseline_args(stems, folder, tmp_path / "r.csv")
    args[args.index(f"alpha={stems['alpha']}")] = str(stems["alpha"])
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "NAME=STEM" in result.stderr


def test_baseline_refuses_an_unknown_encoder_set(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    out = tmp_path / "r.csv"
    result = runner.invoke(app, baseline_args(stems, folder, out, "--encoder-set", "nonesuch"))
    assert result.exit_code == 1
    assert "nonesuch" in result.stderr
    assert not out.exists()


def test_baseline_refuses_a_bad_size_and_a_bad_output_suffix(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    bad_size = runner.invoke(
        app, baseline_args(stems, folder, tmp_path / "r.csv", "--sizes", "50,most")
    )
    assert bad_size.exit_code == 1 and "most" in bad_size.stderr
    bad_suffix = runner.invoke(app, baseline_args(stems, folder, tmp_path / "r.txt"))
    assert bad_suffix.exit_code == 1 and ".csv or .parquet" in bad_suffix.stderr


def test_baseline_refuses_an_encoder_that_differs_from_the_name(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    args = baseline_args({"wrong": stems["alpha"]}, folder, tmp_path / "r.csv")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "expected wrong" in result.stderr


@pytest.fixture(scope="module")
def minilm_toy_stem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Embed the 54 toy texts with MiniLM through the command line."""
    stem = tmp_path_factory.mktemp("minilm_toy") / "toy_minilm"
    args = [
        *("embed", "texts", "--texts", str(TOY), "--encoder", "minilm", "--out", str(stem)),
        *("--group-column", "group_id", "--device", "cpu"),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.stderr
    return stem


@needs_minilm
def test_baseline_on_the_toy_corpus_with_minilm(
    runner: CliRunner, minilm_toy_stem: Path, tmp_path: Path
) -> None:
    out = tmp_path / "results.csv"
    args = [
        *("baseline", "ridge", "--embeddings", f"minilm={minilm_toy_stem}"),
        *("--texts", str(TOY), "--gold", str(TOY), "--group-column", "group_id"),
        *("--gold-column", "gold_constructiveness", "--sizes", "50,full"),
        *("--floor-units", FLOOR, "--out", str(out)),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    results = pd.read_csv(out)
    full = results[results["size"].astype(str) == "full"]
    assert len(full) == len(BaselineSettings().seeds)
    assert np.isfinite(full["spearman_abs"]).all()
    skipped = pd.read_csv(tmp_path / "results_skipped.csv")
    assert set(skipped["size"].astype(str)) == {"50"}
    assert skipped["reason"].str.startswith("quota_exceeds_half_units").all()
    curves = pd.read_csv(tmp_path / "results_curves.csv")
    assert list(curves["size"].astype(str)) == ["full"]
