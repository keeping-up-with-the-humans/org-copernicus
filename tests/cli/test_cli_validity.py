"""Tests of the ``validity panel`` command."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import typer
from cli_support import option_defaults, summary_value
from typer.core import TyperGroup
from typer.testing import CliRunner

from copernicus._version import __version__
from copernicus.assets import toy_corpus_path
from copernicus.cli import _validity
from copernicus.cli._embedding_options import read_table
from copernicus.cli._inputs import read_gold
from copernicus.cli.app import app
from copernicus.coders import export_scores
from copernicus.rubric.loader import toy_rubric_path
from copernicus.validity import PanelSettings, validity_panel

TOY = toy_corpus_path()
TARGETS = {
    "construct": "gold_constructiveness",
    "specificity": "gold_specificity",
    "tone": "gold_tone",
    "reasoning": "gold_reasoning",
}
N_TOY_TEXTS = 54


def target_args(targets: dict[str, str] = TARGETS) -> list[str]:
    """Build repeated ``--target KEY=COLUMN`` options."""
    return [arg for key, column in targets.items() for arg in ("--target", f"{key}={column}")]


@pytest.fixture(scope="module")
def toy_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Code the 54 toy texts through the command line with the fake backend."""
    store = tmp_path_factory.mktemp("panel") / "run.sqlite"
    args = [
        *("code", "two-stage", "--texts", str(TOY), "--rubric", str(toy_rubric_path())),
        *("--out", str(store), "--backend", "fake", "--runs", "2", "--group-column", "group_id"),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.stderr
    return store


def panel_args(source: list[str], out: Path, *extra: str) -> list[str]:
    """Build the arguments of ``validity panel`` against the toy corpus."""
    return [
        *("validity", "panel", *source, "--texts", str(TOY), "--gold", str(TOY)),
        *("--group-column", "group_id", *target_args(), "--out", str(out), *extra),
    ]


def test_panel_from_a_store_writes_four_rows_and_the_report(
    runner: CliRunner, toy_store: Path, tmp_path: Path
) -> None:
    out, report = tmp_path / "panel.csv", tmp_path / "report.txt"
    args = panel_args(["--store", str(toy_store)], out, "--n-boot", "50", "--report", str(report))
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    frame = pd.read_csv(out)
    assert list(frame["target"]) == list(TARGETS)
    assert "rank validity" in result.stdout
    assert report.read_text(encoding="utf-8").strip() in result.stdout
    sidecar = tmp_path / "panel.csv.json"
    assert summary_value(result.stdout, "description file") == str(sidecar)
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    assert description["analysis"] == "validity panel"
    assert description["settings"] == {
        "covariate": None,
        "covariate_scale": "standardized",
        "n_boot": 50,
        "seed": 42,
        "direction": "positive",
        "reliability": None,
    }
    assert description["targets"] == TARGETS
    assert description["inputs"] == {"store": str(toy_store), "texts": str(TOY), "gold": str(TOY)}
    assert description["text_columns"] == {
        "text_id": "text_id",
        "text": "text",
        "group_id": "group_id",
    }
    assert (description["n_rows"], description["n_used"]) == (4, N_TOY_TEXTS)
    assert description["library_version"] == __version__


def test_the_report_file_goes_through_the_atomic_writer(
    runner: CliRunner, toy_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[tuple[Path, str]] = []
    monkeypatch.setattr(_validity, "write_report", lambda path, text: written.append((path, text)))
    out, report = tmp_path / "panel.csv", tmp_path / "report.txt"
    args = panel_args(["--store", str(toy_store)], out, "--n-boot", "20", "--report", str(report))
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    ((path, text),) = written
    assert path == report and text.startswith("Validity panel:") and text.endswith("\n")


def test_panel_options_reach_the_settings(
    runner: CliRunner, toy_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "panel.parquet"
    extra = ["--n-boot", "20", "--seed", "7", "--direction", "negative", "--reliability", "0.81"]
    result = runner.invoke(
        app, panel_args(["--store", str(toy_store)], out, *extra, "--covariate-scale", "raw")
    )
    assert result.exit_code == 0, result.stderr
    settings = PanelSettings(
        n_boot=20, seed=7, direction="negative", reliability=0.81, covariate_scale="raw"
    )
    scores, gold = export_scores(toy_store), read_gold(TOY)
    table = read_table(TOY, "text_id", "text", "group_id")
    direct = validity_panel(scores, gold, table, targets=TARGETS, settings=settings)
    pd.testing.assert_frame_equal(pd.read_parquet(out), direct.frame)
    default_seed = validity_panel(
        scores, gold, table, targets=TARGETS, settings=replace(settings, seed=42)
    )
    assert not default_seed.frame["rank_validity_low"].equals(direct.frame["rank_validity_low"])
    description = json.loads((tmp_path / "panel.parquet.json").read_text(encoding="utf-8"))
    assert (description["settings"]["seed"], description["settings"]["direction"]) == (
        7,
        "negative",
    )


def test_panel_reliability_per_gold_column(
    runner: CliRunner, toy_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "panel.csv"
    extra = ["--n-boot", "20", "--reliability", "gold_tone=0.81"]
    result = runner.invoke(
        app,
        panel_args(
            ["--store", str(toy_store)], out, *extra, "--reliability", "gold_reasoning=0.64"
        ),
    )
    assert result.exit_code == 0, result.stderr
    frame = pd.read_csv(out).set_index("target")
    assert frame.loc["tone", "ceiling"] == pytest.approx(0.9)
    assert frame.loc["reasoning", "ceiling"] == pytest.approx(0.8)
    assert frame.loc[["construct", "specificity"], "ceiling"].isna().all()
    description = json.loads((tmp_path / "panel.csv.json").read_text(encoding="utf-8"))
    assert description["settings"]["reliability"] == {"gold_tone": 0.81, "gold_reasoning": 0.64}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["0.8", "gold_tone=0.7"], "not both"),
        (["0.8", "0.9"], "one number"),
        (["high"], "number"),
        (["gold_tone=high"], "number"),
        (["gold_tone=0.8", "gold_tone=0.7"], "twice"),
    ],
)
def test_panel_refuses_a_malformed_reliability(
    runner: CliRunner, toy_store: Path, tmp_path: Path, values: list[str], message: str
) -> None:
    extra = [arg for value in values for arg in ("--reliability", value)]
    out = tmp_path / "p.csv"
    result = runner.invoke(app, panel_args(["--store", str(toy_store)], out, *extra))
    assert result.exit_code == 1
    assert "--reliability" in result.stderr and message in result.stderr
    assert not out.exists()


def test_panel_from_a_similarity_style_frame(runner: CliRunner, tmp_path: Path) -> None:
    gold = pd.read_csv(TOY)
    rng = np.random.default_rng(3)
    scores = pd.DataFrame(
        {
            "text_id": gold["text_id"],
            "target_level": "construct",
            "target_id": "construct",
            "score": gold["gold_constructiveness"] + rng.normal(scale=0.5, size=len(gold)),
        }
    )
    path, out = tmp_path / "similarity.csv", tmp_path / "panel.csv"
    scores.to_csv(path, index=False)
    args = [
        *("validity", "panel", "--scores", str(path), "--texts", str(TOY), "--gold", str(TOY)),
        *("--group-column", "group_id", "--target", "construct=gold_constructiveness"),
        *("--n-boot", "30", "--out", str(out)),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    frame = pd.read_csv(out)
    assert len(frame) == 1 and frame.loc[0, "status"] == "ok"
    assert frame.loc[0, "n_used"] == N_TOY_TEXTS
    assert frame.loc[0, "rank_validity"] > 0.5
    assert "rank validity" in result.stdout


def test_panel_defaults_are_the_settings_defaults() -> None:
    settings = PanelSettings()
    defaults = option_defaults("validity", "panel")
    assert defaults["n_boot"] == settings.n_boot and defaults["seed"] == settings.seed
    assert defaults["covariate_scale"] == settings.covariate_scale
    assert defaults["direction"] == settings.direction
    assert defaults["covariate"] is None and defaults["reliability"] is None


def test_panel_refuses_store_and_scores_together(
    runner: CliRunner, toy_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "panel.csv"
    result = runner.invoke(app, panel_args(["--store", str(toy_store), "--scores", str(TOY)], out))
    assert result.exit_code == 1
    assert "exactly one of --store and --scores" in result.stderr
    assert not out.exists()


def test_panel_refuses_neither_store_nor_scores(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, panel_args([], tmp_path / "panel.csv"))
    assert result.exit_code == 1
    assert "exactly one of --store and --scores" in result.stderr


def test_panel_refuses_a_target_without_equals(
    runner: CliRunner, toy_store: Path, tmp_path: Path
) -> None:
    args = [
        *("validity", "panel", "--store", str(toy_store), "--texts", str(TOY)),
        *("--gold", str(TOY), "--target", "construct", "--out", str(tmp_path / "p.csv")),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "KEY=COLUMN" in result.stderr


def test_panel_refuses_a_missing_target(runner: CliRunner, toy_store: Path, tmp_path: Path) -> None:
    args = [
        *("validity", "panel", "--store", str(toy_store), "--texts", str(TOY)),
        *("--gold", str(TOY), "--out", str(tmp_path / "p.csv")),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "--target" in result.stderr


def test_panel_refuses_a_bad_output_suffix_before_any_work(
    runner: CliRunner, toy_store: Path, tmp_path: Path
) -> None:
    result = runner.invoke(app, panel_args(["--store", str(toy_store)], tmp_path / "panel.xlsx"))
    assert result.exit_code == 1
    assert ".csv or .parquet" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"), [("--reliability", "1.5"), ("--seed", "-1"), ("--direction", "up")]
)
def test_panel_refuses_a_setting_out_of_range(
    runner: CliRunner, toy_store: Path, tmp_path: Path, option: str, value: str
) -> None:
    out = tmp_path / "p.csv"
    result = runner.invoke(app, panel_args(["--store", str(toy_store)], out, option, value))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and option.lstrip("-") in result.stderr
    assert not out.exists()


@pytest.mark.parametrize(("group", "command"), [("validity", "panel"), ("baseline", "ridge")])
def test_the_command_help_keeps_each_paragraph_on_one_line(
    runner: CliRunner, group: str, command: str
) -> None:
    root = typer.main.get_command(app)
    assert isinstance(root, TyperGroup)
    group_command = root.commands[group]
    assert isinstance(group_command, TyperGroup)
    paragraphs = inspect.cleandoc(group_command.commands[command].help or "").split("\n\n")
    result = runner.invoke(app, [group, command, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.stderr
    lines = {" ".join(line.replace("│", " ").split()) for line in result.stdout.splitlines()}
    assert len(paragraphs) >= 3
    for paragraph in paragraphs:
        assert "\n" not in paragraph, paragraph
        assert " ".join(paragraph.split()) in lines
