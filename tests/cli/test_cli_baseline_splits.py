"""The splits options of ``baseline ridge``: reading folds and draws, writing the ones used."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from baseline_cli_support import baseline_args, save_world
from baseline_support import GOLD, World, run
from cli_support import option_defaults, summary_value
from splits_support import draw_table, fold_table, keyed_folds
from typer.testing import CliRunner

from copernicus.baseline import BaselineSettings, BaselineSplits, read_splits, write_splits
from copernicus.cli.app import app

SETTINGS = BaselineSettings(sizes=(50, "full"), seeds=(42,), k_folds=3, encoder_sets=("alpha",))
OPTIONS = ("--sizes", "50,full", "--seeds", "42", "--k-folds", "3", "--encoder-set", "alpha")
TABLES = ("", "_oof", "_skipped", "_curves")


@pytest.fixture(scope="module")
def saved_world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Path]]:
    folder = tmp_path_factory.mktemp("world")
    return folder, save_world(World(), folder)


def description(out: Path) -> dict[str, object]:
    return json.loads(out.with_name(out.name + ".json").read_text(encoding="utf-8"))


def assert_same_files(first: Path, second: Path) -> None:
    """The four tables of two runs hold the same values, the floating-point ones in every bit."""
    for suffix in TABLES:
        mine = pd.read_parquet(first.with_name(f"{first.stem}{suffix}.parquet"))
        other = pd.read_parquet(second.with_name(f"{second.stem}{suffix}.parquet"))
        pd.testing.assert_frame_equal(mine, other, check_exact=True)


def test_the_defaults_neither_read_nor_write_splits() -> None:
    defaults = option_defaults("baseline", "ridge")
    assert defaults["splits"] is None and defaults["write_splits"] is None


def test_written_splits_are_the_ones_the_run_used_and_give_the_run_again(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    stems = {"alpha": stems["alpha"]}
    first, splits = tmp_path / "first.parquet", tmp_path / "splits.json"
    written = runner.invoke(
        app, baseline_args(stems, folder, first, *OPTIONS, "--write-splits", str(splits))
    )
    assert written.exit_code == 0, written.stderr
    assert summary_value(written.stdout, "splits file") == str(splits)
    expected = run(World(), SETTINGS).splits
    used = read_splits(splits)
    pd.testing.assert_frame_equal(used.folds, expected.folds)
    assert used.draws is not None and expected.draws is not None
    pd.testing.assert_frame_equal(used.draws, expected.draws)
    record = description(first)
    assert record["supplied_splits"] == {"file": None, "folds": False, "draws": False}
    assert record["written_splits"] == str(splits)
    second = tmp_path / "second.parquet"
    again = runner.invoke(
        app, baseline_args(stems, folder, second, *OPTIONS, "--splits", str(splits))
    )
    assert again.exit_code == 0, again.stderr
    assert_same_files(first, second)
    record = description(second)
    assert record["supplied_splits"] == {"file": str(splits), "folds": True, "draws": True}
    assert record["written_splits"] is None


def test_a_file_of_folds_alone_leaves_the_draws_to_the_generators(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    world = World()
    folds = keyed_folds(world, (42,), k=3, gold_column="another key")
    path = tmp_path / "folds.json"
    write_splits(BaselineSplits(folds=fold_table(world, folds)), path)
    out, used = tmp_path / "r.parquet", tmp_path / "used.json"
    args = baseline_args({"alpha": stems["alpha"]}, folder, out, *OPTIONS, "--splits", str(path))
    result = runner.invoke(app, [*args, "--write-splits", str(used)])
    assert result.exit_code == 0, result.stderr
    assert description(out)["supplied_splits"] == {"file": str(path), "folds": True, "draws": False}
    oof = pd.read_parquet(tmp_path / "r_oof.parquet").set_index("text_id")
    labels = fold_table(world, folds).set_index("text_id")["fold"]
    assert (oof.loc[world.ids, "fold"].to_numpy() == labels.loc[world.ids].to_numpy()).all()
    python = run(world, SETTINGS, splits=BaselineSplits(folds=fold_table(world, folds)))
    assert python.splits.draws is not None
    pd.testing.assert_frame_equal(read_splits(used).draws, python.splits.draws)


def test_splits_that_do_not_fit_the_run_are_refused_before_a_table_is_written(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    path = tmp_path / "one_seed.json"
    write_splits(BaselineSplits(folds=fold_table(World(), keyed_folds(World(), (42,), k=3))), path)
    out = tmp_path / "r.csv"
    options = ("--sizes", "full", "--seeds", "42,43", "--k-folds", "3", "--splits", str(path))
    result = runner.invoke(app, baseline_args({"alpha": stems["alpha"]}, folder, out, *options))
    assert result.exit_code == 1
    assert (
        "error: the seeds of the splits' folds (42) are not the seeds of the settings"
        in result.stderr
    )
    assert not out.exists()


def test_a_file_without_splits_is_refused_with_its_name(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"analysis": "NLP-ML baseline"}), encoding="utf-8")
    out = tmp_path / "r.csv"
    args = baseline_args({"alpha": stems["alpha"]}, folder, out, *OPTIONS, "--splits", str(path))
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert f"error: {path} holds no splits" in result.stderr
    assert not out.exists()


def test_writing_splits_into_a_missing_folder_is_refused_before_any_work(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    out, target = tmp_path / "r.csv", tmp_path / "missing" / "splits.json"
    args = baseline_args(
        {"alpha": stems["alpha"]}, folder, out, *OPTIONS, "--write-splits", str(target)
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert f"folder of the --write-splits file does not exist: {target.parent}" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    "name", ["r.csv", "r.csv.json", "r_oof.csv", "r_skipped.csv", "r_curves.csv"]
)
def test_writing_splits_over_an_output_of_the_run_is_refused(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path, name: str
) -> None:
    folder, stems = saved_world
    out = tmp_path / "r.csv"
    args = baseline_args(
        {"alpha": stems["alpha"]}, folder, out, *OPTIONS, "--write-splits", str(tmp_path / name)
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "--write-splits must name a file other than the results" in result.stderr
    assert not out.exists()


def test_the_gold_column_can_be_renamed_when_the_splits_are_given(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    stems = {"alpha": stems["alpha"]}
    first, splits = tmp_path / "first.parquet", tmp_path / "splits.json"
    written = runner.invoke(
        app, baseline_args(stems, folder, first, *OPTIONS, "--write-splits", str(splits))
    )
    assert written.exit_code == 0, written.stderr
    # Only the header changes: the lines of values stay as they are, digit for digit.
    renamed = tmp_path / "renamed_gold.csv"
    header, values = (folder / "gold.csv").read_text(encoding="utf-8").split("\n", 1)
    assert header.split(",").count(GOLD) == 1
    renamed_header = ",".join("other" if name == GOLD else name for name in header.split(","))
    renamed.write_text(f"{renamed_header}\n{values}", encoding="utf-8")
    second = tmp_path / "second.parquet"
    args = baseline_args(stems, folder, second, *OPTIONS, "--splits", str(splits))
    args[args.index(str(folder / "gold.csv"))] = str(renamed)
    args[args.index(GOLD)] = "other"
    again = runner.invoke(app, args)
    assert again.exit_code == 0, again.stderr
    assert_same_files(first, second)
    assert np.isfinite(pd.read_parquet(second)["spearman"]).all()


def test_an_all_skipped_run_round_trips_through_the_command_line(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    stems = {"alpha": stems["alpha"]}
    options = ("--sizes", "500", "--seeds", "42", "--k-folds", "3", "--encoder-set", "alpha")
    first, splits = tmp_path / "first.parquet", tmp_path / "splits.json"
    written = runner.invoke(
        app, baseline_args(stems, folder, first, *options, "--write-splits", str(splits))
    )
    assert written.exit_code == 0, written.stderr
    stored = read_splits(splits)
    assert stored.draws is not None and stored.draws.empty
    second = tmp_path / "second.parquet"
    again = runner.invoke(
        app, baseline_args(stems, folder, second, *options, "--splits", str(splits))
    )
    assert again.exit_code == 0, again.stderr
    assert_same_files(first, second)
    supplied = {"file": str(splits), "folds": True, "draws": True}
    assert description(second)["supplied_splits"] == supplied


def test_the_description_says_supplied_for_a_draws_table_that_the_run_ignores(
    runner: CliRunner, saved_world: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    folder, stems = saved_world
    world = World()
    folds = keyed_folds(world, (42,), k=3)
    pools = [train for train, _ in folds[42]]
    ignored = draw_table(world, {(42, "500"): pools})
    path = tmp_path / "ignored.json"
    write_splits(BaselineSplits(folds=fold_table(world, folds), draws=ignored), path)
    out = tmp_path / "r.parquet"
    options = ("--sizes", "500,full", "--seeds", "42", "--k-folds", "3", "--splits", str(path))
    result = runner.invoke(app, baseline_args({"alpha": stems["alpha"]}, folder, out, *options))
    assert result.exit_code == 0, result.stderr
    assert list(pd.read_parquet(out)["size"]) == ["full"]
    assert description(out)["supplied_splits"] == {"file": str(path), "folds": True, "draws": True}
