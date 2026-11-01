"""Tests of the ``corrections`` commands sample, plan, battery and list."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cli_support import option_defaults, summary_value
from corrections_cli_support import data_args, library_data, write_corpus
from corrections_support import BATTERY_SCORES
from typer.testing import CliRunner

from copernicus._version import __version__
from copernicus.cli._correction_inputs import parse_families, parse_sizes
from copernicus.cli.app import app
from copernicus.corrections import REGISTRY, BatterySettings, population_hash_of, run_battery
from copernicus.errors import SettingsError


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The synthetic battery corpus as a texts, a scores and a gold file."""
    return write_corpus(tmp_path_factory.mktemp("corpus"))


def test_plan_prints_the_effective_size_factor(runner: CliRunner) -> None:
    args = ["corrections", "plan", "--rho", "0.7", "--n-human", "100", "--n-machine", "1000"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert summary_value(result.stdout, "factor") == "1.79"
    assert summary_value(result.stdout, "limit") == "1.96"
    assert summary_value(result.stdout, "uncorrected") == "2.04"


def test_plan_over_lists_writes_the_table(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "plan.csv"
    args = ["corrections", "plan", "--rho", "0.5", "--n-human", "50,100", "--n-machine", "500,1000"]
    result = runner.invoke(app, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.stderr
    table = pd.read_csv(out)
    assert list(table["n_human"]) == [50, 50, 100, 100]
    assert list(table["n_machine"]) == [500, 1000, 500, 1000]
    refused = runner.invoke(app, args)
    assert refused.exit_code == 1 and "--out" in refused.stderr
    not_counts = runner.invoke(app, [*args[:5], "50,many", *args[6:]])
    assert not_counts.exit_code == 1 and "--n-human expects integers" in not_counts.stderr


def test_sample_writes_the_drawn_texts_with_pi(
    runner: CliRunner, files: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "sample.csv"
    args = [
        *("corrections", "sample", "--texts", str(files["texts"]), "--group-column", "group_id"),
        *("--n", "100", "--seed", "3", "--out", str(out)),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    sample = pd.read_csv(out)
    assert list(sample.columns) == ["text_id", "group_id", "pi"]
    assert len(sample) == 100 and sample["group_id"].nunique() == 25
    assert np.allclose(sample["pi"], 25 / 150)
    assert summary_value(result.stdout, "groups drawn") == "25 of 150"
    description = json.loads((tmp_path / "sample.csv.json").read_text(encoding="utf-8"))
    assert description["analysis"] == "human sample"
    assert description["draw"]["m"] == 25 and description["draw"]["seed"] == 3
    assert description["settings"] == {"n": 100, "share": None, "seed": 3, "unit_id": None}
    texts = pd.read_csv(files["texts"], dtype=str)
    table_hash = population_hash_of(texts["text_id"], texts["group_id"])
    assert description["draw"]["population_hash"] == table_hash
    both = runner.invoke(app, [*args, "--share", "0.1"])
    assert both.exit_code == 1 and "exactly one of n and share" in both.stderr


def test_battery_writes_its_four_tables_and_the_description(
    runner: CliRunner, files: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "battery.csv"
    args = [
        *("corrections", "battery", *data_args(files)),
        *("--families", "regression_calibration:linear", "--families", "supervised_stacking:ridge"),
        *("--families", "multiple_imputation:model_draws:m_draws=3"),
        *("--sizes", "100", "--schemes", "kfold5", "--seeds", "42", "--out", str(out)),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    results = pd.read_csv(out)
    settings = BatterySettings(
        families=[
            ("regression_calibration", "linear"),
            ("supervised_stacking", "ridge"),
            ("multiple_imputation", "model_draws", {"m_draws": 3}),
        ],
        sizes=(100,),
        schemes=("kfold5",),
        seeds=(42,),
    )
    direct = run_battery(library_data(files), settings)
    assert list(results["variant"]) == list(direct.results["variant"])
    assert np.allclose(results["gain_rank_validity"], direct.results["gain_rank_validity"])
    for label in ("skipped", "raw", "summary"):
        assert (tmp_path / f"battery_{label}.csv").is_file()
    summary = pd.read_csv(tmp_path / "battery_summary.csv")
    assert list(summary["variant"]) == ["linear", "ridge", "model_draws(m_draws=3)"]
    sidecar = tmp_path / "battery.csv.json"
    assert summary_value(result.stdout, "description file") == str(sidecar)
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    assert description["analysis"] == "correction battery"
    assert description["settings"]["sizes"] == [100]
    assert description["settings"]["entries"] == [
        "regression_calibration:linear",
        "supervised_stacking:ridge",
        "multiple_imputation:model_draws(m_draws=3)",
    ]
    assert description["inputs"]["score_columns"] == list(BATTERY_SCORES)
    assert description["tables"]["summary"] == str(tmp_path / "battery_summary.csv")
    assert description["library_version"] == __version__
    assert summary_value(result.stdout, "results") == str(len(results))


def test_battery_refuses_a_fractional_parameter_before_reading_the_data(
    runner: CliRunner, files: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "battery.csv"
    args = [
        *("corrections", "battery", *data_args(files)),
        *("--families", "regression_calibration:linear"),
        *("--families", "multiple_imputation:pmm:k_donors=2.5"),
        *("--sizes", "100", "--schemes", "kfold5", "--seeds", "42", "--out", str(out)),
    ]
    args[args.index("--gold-column") + 1] = "absent_gold"
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "k_donors must be an integer of at least 1; got 2.5" in result.stderr
    assert "absent_gold" not in result.stderr and not out.exists()


def test_the_default_families_leave_out_the_embedding_family_without_embeddings(
    runner: CliRunner, files: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "battery.csv"
    args = [
        *("corrections", "battery", *data_args(files)),
        *("--sizes", "100", "--schemes", "holdout", "--seeds", "42", "--out", str(out)),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    entries = json.loads((tmp_path / "battery.csv.json").read_text())["settings"]["entries"]
    expected = [
        entry.key for entry in BatterySettings().entries if entry.family != "embedding_augmented"
    ]
    assert entries == expected and len(entries) == len(BatterySettings().entries) - 3
    assert "embedding_augmented" not in set(pd.read_csv(out)["family"])


def test_battery_defaults_are_the_settings_defaults() -> None:
    settings = BatterySettings()
    defaults = option_defaults("corrections", "battery")
    assert defaults["sizes"] == "50,100,250,500,1000,0.01,0.05,0.1,0.25"
    assert defaults["schemes"] == ",".join(settings.schemes)
    assert defaults["seeds"] == ",".join(str(seed) for seed in settings.seeds)
    assert defaults["covariate_scale"] == settings.covariate_scale
    assert defaults["families"] is None


def test_battery_refuses_a_bad_family_score_columns_and_output(
    runner: CliRunner, files: dict[str, Path], tmp_path: Path
) -> None:
    base = ["corrections", "battery", *data_args(files), "--sizes", "100", "--schemes", "kfold5"]
    out = ["--out", str(tmp_path / "r.csv")]
    bad_family = runner.invoke(app, [*base, "--families", "linear", *out])
    assert bad_family.exit_code == 1 and "FAMILY:VARIANT" in bad_family.stderr
    no_scores = [*base, *out]
    no_scores[no_scores.index("--score-columns") + 1] = " , "
    empty = runner.invoke(app, no_scores)
    assert empty.exit_code == 1 and "--score-columns must name" in empty.stderr
    bad_suffix = runner.invoke(app, [*base, "--out", str(tmp_path / "r.txt")])
    assert bad_suffix.exit_code == 1 and ".csv or .parquet" in bad_suffix.stderr
    assert not (tmp_path / "r.csv").exists()


def test_list_shows_every_registered_family(runner: CliRunner) -> None:
    result = runner.invoke(app, ["corrections", "list"])
    assert result.exit_code == 0, result.stderr
    assert len(REGISTRY) == 15
    for family in REGISTRY:
        assert family in result.stdout
    assert "15 families" in result.stdout


def test_sizes_and_families_parse_into_settings_values() -> None:
    assert parse_sizes("50, 0.05,5e-2") == (50, 0.05, 0.05)
    assert parse_families(["regression_calibration:linear", "a:b:encoder=minilm:m_draws=3"]) == [
        ("regression_calibration", "linear"),
        ("a", "b", {"encoder": "minilm", "m_draws": 3}),
    ]
    with pytest.raises(SettingsError, match="'most'"):
        parse_sizes("50,most")
    with pytest.raises(SettingsError, match="NAME=VALUE"):
        parse_families(["a:b:encoder"])
