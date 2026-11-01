"""The draw file of ``corrections sample`` binds a coded sample to the texts file of an estimate."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from corrections_cli_support import (
    N_DRAWN,
    data_args,
    estimate_args,
    library_data,
    write_corpus,
    write_drawn_gold,
)
from typer.testing import CliRunner

from copernicus.cli._embedding_options import read_table
from copernicus.cli.app import app
from copernicus.corrections import attach_draw, draw_human_sample, mean_ppi, read_draw

DIFFERENT_TABLE = "the draw was made on a different table than the data (different texts or groups)"
Regroup = Callable[[pd.DataFrame, str], pd.DataFrame]


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The synthetic battery corpus as a texts, a scores and a gold file."""
    return write_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="module")
def drawn_gold(files: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The coded sample: the sample command's texts with their gold values and ``pi``."""
    return write_drawn_gold(files, tmp_path_factory.mktemp("drawn"))


@pytest.fixture(scope="module")
def draw_file(drawn_gold: Path) -> Path:
    """The draw file the sample command wrote beside the drawn texts."""
    return drawn_gold.parent / "sample.csv.json"


def _estimate(
    runner: CliRunner, files: dict[str, Path], gold: Path, out: Path, *extra: str
) -> tuple[int, str, Any]:
    """Run ``corrections estimate --kind mean``; the exit code, stderr and the written JSON."""
    result = runner.invoke(app, estimate_args(files, gold, out, "--kind", "mean", *extra))
    written = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    return result.exit_code, " ".join(result.stderr.split()), written


def _renamed(texts: pd.DataFrame, drawn_group: str) -> pd.DataFrame:
    return texts.assign(group_id="person " + texts["group_id"])


def _split(texts: pd.DataFrame, drawn_group: str) -> pd.DataFrame:
    members = texts.index[texts["group_id"] == drawn_group]
    regrouped = texts.copy()
    regrouped.loc[members[: members.size // 2], "group_id"] = drawn_group + " part"
    return regrouped


def test_the_sample_command_writes_a_draw_file_that_reads_back(
    files: dict[str, Path], draw_file: Path
) -> None:
    loaded = read_draw(draw_file)
    table = read_table(files["texts"], "text_id", "text", "group_id")
    direct = draw_human_sample(table, n=N_DRAWN, seed=11)
    for name in ("ids", "groups", "pi", "m", "n_groups", "n_units", "quota", "seed"):
        assert np.array_equal(getattr(loaded, name), getattr(direct, name)), name
    assert loaded.population_hash == direct.population_hash


def test_an_estimate_with_the_draw_file_takes_pi_from_the_draw(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, draw_file: Path, tmp_path: Path
) -> None:
    draw = ("--draw", str(draw_file))
    code, stderr, bound = _estimate(runner, files, drawn_gold, tmp_path / "a.json", *draw)
    assert code == 0, stderr
    data = library_data(files, drawn_gold)
    direct = mean_ppi(attach_draw(data, pd.Series(data.gold, index=data.ids), read_draw(draw_file)))
    assert bound["estimate"]["point"] == pytest.approx(direct.point)
    assert bound["estimate"]["se"] == pytest.approx(direct.se)
    assert bound["estimate"]["n_labelled"] == N_DRAWN
    assert bound["inputs"]["draw"] == str(draw_file)
    pi = ("--pi-column", "pi")
    code, stderr, unbound = _estimate(runner, files, drawn_gold, tmp_path / "b.json", *pi)
    assert code == 0 and unbound["estimate"]["point"] == pytest.approx(direct.point), stderr
    assert unbound["inputs"]["draw"] is None
    code, stderr, both = _estimate(runner, files, drawn_gold, tmp_path / "c.json", *pi, *draw)
    assert code == 0 and both["estimate"]["point"] == pytest.approx(direct.point), stderr


@pytest.mark.parametrize("regroup", [_renamed, _split], ids=["renamed groups", "split group"])
def test_a_texts_file_regrouped_after_the_draw_is_refused_with_the_draw_file(
    runner: CliRunner,
    files: dict[str, Path],
    drawn_gold: Path,
    draw_file: Path,
    tmp_path: Path,
    regroup: Regroup,
) -> None:
    drawn_group = str(read_draw(draw_file).groups[0])
    regrouped = {**files, "texts": tmp_path / "regrouped.csv"}
    texts = pd.read_csv(files["texts"], dtype={"group_id": str})
    regroup(texts, drawn_group).to_csv(regrouped["texts"], index=False)
    out = tmp_path / "e.json"
    code, stderr, _ = _estimate(runner, regrouped, drawn_gold, out, "--draw", str(draw_file))
    assert code == 1 and DIFFERENT_TABLE in stderr and not out.exists()
    # Without the draw file the coded texts still cover whole groups, so the estimate runs.
    code, stderr, _ = _estimate(runner, regrouped, drawn_gold, out, "--pi-column", "pi")
    assert code == 0, stderr


def test_a_pi_column_that_disagrees_with_the_draw_is_refused(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, draw_file: Path, tmp_path: Path
) -> None:
    coded = pd.read_csv(drawn_gold)
    halved = tmp_path / "halved.csv"
    coded.assign(pi=coded["pi"] / 2).to_csv(halved, index=False)
    draw = ("--draw", str(draw_file))
    for extra in (draw, ("--pi-column", "pi", *draw)):
        out = tmp_path / "e.json"
        code, stderr, _ = _estimate(runner, files, halved, out, *extra)
        assert code == 1 and "column 'pi' of the gold file" in stderr and not out.exists()
        assert repr(str(coded["text_id"].iloc[0])) in stderr and "the draw's pi" in stderr
    without_pi = tmp_path / "without_pi.csv"
    coded.drop(columns=["pi"]).to_csv(without_pi, index=False)
    code, stderr, _ = _estimate(runner, files, without_pi, tmp_path / "w.json", *draw)
    assert code == 0, stderr
    named = ("--pi-column", "pi", *draw)
    code, stderr, _ = _estimate(runner, files, without_pi, tmp_path / "n.json", *named)
    assert code == 1 and "--pi-column 'pi' is not a column of the gold file" in stderr


def test_the_log_counts_the_coded_texts_outside_the_draw_that_are_set_aside(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, draw_file: Path, tmp_path: Path
) -> None:
    coded = pd.read_csv(drawn_gold)
    gold = pd.read_csv(files["gold"])
    others = gold[~gold["text_id"].isin(coded["text_id"])].head(2)
    wider = tmp_path / "wider.csv"
    pd.concat([coded, others]).to_csv(wider, index=False)
    draw = ("--draw", str(draw_file), "--verbose")
    code, stderr, written = _estimate(runner, files, wider, tmp_path / "w.json", *draw)
    assert code == 0, stderr
    assert written["estimate"]["n_labelled"] == N_DRAWN
    logged = f"bound the {N_DRAWN} drawn texts to the draw in {draw_file}; coded texts outside"
    assert f"{logged} the draw, set aside: 2" in stderr
    code, stderr, _ = _estimate(runner, files, drawn_gold, tmp_path / "d.json", *draw)
    assert code == 0 and f"{logged} the draw, set aside: 0" in stderr


def test_a_moment_estimate_binds_too_and_the_help_names_the_draw_file(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, draw_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "m.json"
    args = [*data_args(files, drawn_gold), "--out", str(out), "--draw", str(draw_file)]
    moments = ("--kind", "moment-subtraction", "--outcome", "n_words", "--n-boot", "50")
    result = runner.invoke(app, ["corrections", "estimate", *args, *moments])
    assert result.exit_code == 0, result.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["estimate"]["n_labelled"] == N_DRAWN
    for command in ("estimate", "sample"):
        shown = runner.invoke(app, ["corrections", command, "--help"], env={"COLUMNS": "200"})
        assert "--draw" in shown.stdout and "binds" in shown.stdout, command
