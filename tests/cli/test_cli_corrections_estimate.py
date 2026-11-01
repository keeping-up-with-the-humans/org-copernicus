"""Tests of the ``corrections estimate`` command on a drawn human sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cli_support import summary_value
from corrections_cli_support import (
    N_DRAWN,
    estimate_args,
    library_data,
    write_corpus,
    write_drawn_gold,
)
from typer.testing import CliRunner

from copernicus._version import __version__
from copernicus.cli.app import app
from copernicus.corrections import (
    group_difference_ppi,
    long_short_split,
    mean_ppi,
    moment_subtraction,
    ols_outcome_ppi,
    ols_predictor_ppi,
)


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The synthetic battery corpus as a texts, a scores and a gold file."""
    return write_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="module")
def drawn_gold(files: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Gold of a drawn human sample: the sample command's texts with their values and pi."""
    return write_drawn_gold(files, tmp_path_factory.mktemp("drawn"))


def run_estimate(
    runner: CliRunner, files: dict[str, Path], gold: Path, out: Path, *extra: str
) -> dict[str, object]:
    """Run the command, require success, and return the written JSON."""
    result = runner.invoke(app, estimate_args(files, gold, out, *extra))
    assert result.exit_code == 0, result.stderr
    written: dict[str, object] = json.loads(out.read_text(encoding="utf-8"))
    return written


def test_estimate_of_a_mean_on_a_drawn_sample(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    out = tmp_path / "mean.json"
    result = runner.invoke(
        app, estimate_args(files, drawn_gold, out, "--kind", "mean", "--pi-column", "pi")
    )
    assert result.exit_code == 0, result.stderr
    written = json.loads(out.read_text(encoding="utf-8"))
    direct = mean_ppi(library_data(files, drawn_gold, pi_column="pi"))
    assert written["estimate"]["point"] == pytest.approx(direct.point)
    assert written["estimate"]["se"] == pytest.approx(direct.se)
    assert written["estimate"]["n_labelled"] == N_DRAWN and written["estimate"]["n_total"] == 600
    assert written["estimate"]["details"]["prediction"] == "score_a"
    assert written["kind"] == "mean" and written["library_version"] == __version__
    assert summary_value(result.stdout, "method") == "ppi_mean"
    assert f"{direct.point:.4g}" in summary_value(result.stdout, "estimate")


def test_estimate_of_an_outcome_regression_with_a_formula(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    extra = ("--kind", "ols-outcome", "--formula", "1 + log_chars", "--pi-column", "pi")
    written = run_estimate(runner, files, drawn_gold, tmp_path / "ols.json", *extra, "--cluster")
    data = library_data(files, drawn_gold, pi_column="pi")
    direct = ols_outcome_ppi(data, "1 + log_chars", cluster=True)
    estimate = written["estimate"]
    assert isinstance(estimate, dict)
    assert estimate["point"] == pytest.approx(direct.point)
    assert estimate["details"]["columns"] == ["Intercept", "log_chars"]
    assert estimate["details"]["coefficients"]["log_chars"] == pytest.approx(direct.point)
    assert written["options"] == {
        "formula": "1 + log_chars",
        "cluster": True,
        "pi_column": "pi",
        "covariate": "log_chars",
    }


def test_estimate_reports_a_coefficient_by_name_or_by_position(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    extra = ("--kind", "ols-outcome", "--formula", "1 + log_chars", "--pi-column", "pi")
    by_name = run_estimate(
        runner, files, drawn_gold, tmp_path / "n.json", *extra, "--coefficient", "Intercept"
    )
    by_position = run_estimate(
        runner, files, drawn_gold, tmp_path / "p.json", *extra, "--coefficient", "0"
    )
    data = library_data(files, drawn_gold, pi_column="pi")
    direct = ols_outcome_ppi(data, "1 + log_chars", coefficient="Intercept")
    for written in (by_name, by_position):
        estimate = written["estimate"]
        assert isinstance(estimate, dict)
        assert estimate["point"] == pytest.approx(direct.point)
        assert estimate["details"]["coefficient"] == "Intercept"


def test_estimate_of_differences_by_length_and_by_a_flag_column(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    long_short = ("--kind", "difference", "--long-short", "--pi-column", "pi", "--power-tuning")
    written = run_estimate(runner, files, drawn_gold, tmp_path / "d.json", *long_short)
    data = library_data(files, drawn_gold, pi_column="pi", extra_columns=("flag",))
    direct = group_difference_ppi(data, long_short_split(data), power_tuning=True)
    estimate = written["estimate"]
    assert isinstance(estimate, dict)
    assert estimate["point"] == pytest.approx(direct.point)
    assert estimate["method"] == "ppi_difference_tuned"
    by_flag = ("--kind", "difference", "--group-column-split", "flag", "--pi-column", "pi")
    flagged = run_estimate(runner, files, drawn_gold, tmp_path / "f.json", *by_flag)
    mask = data.frame["flag"].to_numpy(dtype=bool)
    expected = group_difference_ppi(data, mask)
    estimate = flagged["estimate"]
    assert isinstance(estimate, dict)
    assert estimate["point"] == pytest.approx(expected.point)
    assert estimate["details"]["n_group"] == int(mask.sum())
    by_half = ("--kind", "difference", "--group-column-split", "half", "--pi-column", "pi")
    halved = run_estimate(runner, files, drawn_gold, tmp_path / "h.json", *by_half)
    swapped = halved["estimate"]
    assert isinstance(swapped, dict)
    assert swapped["point"] == pytest.approx(-expected.point)
    assert swapped["details"]["n_group"] == int((~mask).sum())


def test_estimate_of_a_predictor_slope_with_covariates_and_the_ludwig_variant(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    extra = (
        *("--kind", "ols-predictor", "--outcome", "n_words", "--covariates", "1 + log_chars"),
        *("--variant", "ludwig", "--pi-column", "pi", "--prediction-column", "score_c"),
    )
    written = run_estimate(runner, files, drawn_gold, tmp_path / "ludwig.json", *extra)
    data = library_data(files, drawn_gold, pi_column="pi", extra_columns=("n_words",))
    prediction = pd.Series(data.frame["score_c"].to_numpy(), index=data.ids, name="score_c")
    direct = ols_predictor_ppi(
        data, "n_words", "1 + log_chars", prediction=prediction, variant="ludwig"
    )
    estimate = written["estimate"]
    assert isinstance(estimate, dict)
    assert estimate["point"] == pytest.approx(direct.point)
    assert estimate["method"] == "ludwig_ols_predictor"
    assert estimate["details"]["columns"] == ["Intercept", "construct", "log_chars"]


def test_estimate_of_a_predictor_slope_by_moment_subtraction_and_by_oriv(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    moments = (
        *("--kind", "moment-subtraction", "--outcome", "n_words"),
        *("--prediction-column", "score_c", "--n-boot", "50", "--seed", "3"),
    )
    written = run_estimate(runner, files, drawn_gold, tmp_path / "m.json", *moments)
    data = library_data(files, drawn_gold, extra_columns=("n_words",))
    direct = moment_subtraction(data, "n_words", score_column="score_c", n_boot=50, seed=3)
    estimate = written["estimate"]
    assert isinstance(estimate, dict)
    assert estimate["point"] == pytest.approx(direct.point)
    assert estimate["se"] == pytest.approx(direct.se)
    assert estimate["details"]["score_column"] == "score_c"
    instrumented = (
        *("--kind", "oriv", "--outcome", "n_words", "--score-a", "score_a"),
        *("--score-b", "score_b", "--n-boot", "50", "--seed", "4"),
    )
    written = run_estimate(runner, files, drawn_gold, tmp_path / "o.json", *instrumented)
    estimate = written["estimate"]
    assert isinstance(estimate, dict)
    assert estimate["method"] == "oriv" and estimate["details"]["n_boot"] == 50
    assert np.isfinite(estimate["point"]) and estimate["se"] > 0


def test_estimate_clusters_by_group_unless_told_not_to(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    mean = ("--kind", "mean", "--pi-column", "pi")
    automatic = run_estimate(runner, files, drawn_gold, tmp_path / "a.json", *mean)
    independent = run_estimate(
        runner, files, drawn_gold, tmp_path / "i.json", *mean, "--no-cluster"
    )
    data = library_data(files, drawn_gold, pi_column="pi", extra_columns=("n_words",))
    for written, cluster in ((automatic, None), (independent, False)):
        estimate = written["estimate"]
        assert isinstance(estimate, dict)
        expected = mean_ppi(data, cluster=cluster)
        assert estimate["se"] == pytest.approx(expected.se)
        assert estimate["details"]["clustered"] is (cluster is None)
    assert "cluster" not in automatic["options"]  # type: ignore[operator]
    assert independent["options"]["cluster"] is False  # type: ignore[index]
    split = long_short_split(data)
    for kind, direct in (
        (("--kind", "difference", "--long-short"), group_difference_ppi(data, split, cluster=True)),
        (("--kind", "ols-predictor", "--outcome", "n_words"), ols_predictor_ppi(data, "n_words")),
        (
            ("--kind", "ols-outcome", "--formula", "1 + log_chars"),
            ols_outcome_ppi(data, "1 + log_chars", cluster=True),
        ),
    ):
        written = run_estimate(
            runner, files, drawn_gold, tmp_path / "k.json", *kind, "--pi-column", "pi", "--cluster"
        )
        estimate = written["estimate"]
        assert isinstance(estimate, dict) and estimate["details"]["clustered"] is True
        assert estimate["se"] == pytest.approx(direct.se)


def test_estimate_refuses_a_clustered_difference_with_one_coded_group_on_a_side(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    texts = pd.read_csv(files["texts"])
    coded = texts["text_id"].isin(pd.read_csv(drawn_gold)["text_id"])
    drawn = sorted(texts.loc[coded, "group_id"].unique())
    undrawn = sorted(set(texts["group_id"]) - set(drawn))
    texts["rare"] = texts["group_id"].isin([drawn[0], *undrawn[:3]])
    rare_files = {**files, "texts": tmp_path / "texts_rare.csv"}
    texts.to_csv(rare_files["texts"], index=False)
    split = ("--kind", "difference", "--group-column-split", "rare", "--pi-column", "pi")
    out = tmp_path / "d.json"
    result = runner.invoke(app, estimate_args(rare_files, drawn_gold, out, *split))
    assert result.exit_code == 1 and not out.exists()
    assert f"where the grouping is True all lie in the group {drawn[0]!r}" in result.stderr
    assert "draw more groups" in result.stderr and "--no-cluster" in result.stderr
    independent = run_estimate(
        runner, rare_files, drawn_gold, tmp_path / "i.json", *split, "--no-cluster"
    )
    assert independent["estimate"]["details"]["clustered"] is False  # type: ignore[index]


def test_estimate_passes_the_weak_instrument_floor_to_oriv(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    instrumented = (
        *("--kind", "oriv", "--outcome", "n_words", "--score-a", "score_a"),
        *("--score-b", "score_b", "--n-boot", "50", "--weak-instrument-floor", "0.3"),
    )
    written = run_estimate(runner, files, drawn_gold, tmp_path / "o.json", *instrumented)
    estimate = written["estimate"]
    assert isinstance(estimate, dict)
    assert estimate["details"]["weak_instrument_floor"] == 0.3
    assert written["options"]["weak_instrument_floor"] == 0.3  # type: ignore[index]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--kind", "ols-outcome"), "--formula"),
        (("--kind", "mean", "--weak-instrument-floor", "0.3"), "--weak-instrument-floor does"),
        (("--kind", "simex", "--outcome", "n_words", "--no-cluster"), "--cluster does not apply"),
        (("--kind", "ols-predictor"), "--outcome"),
        (("--kind", "oriv", "--outcome", "n_words", "--score-b", "score_b"), "--score-a"),
        (("--kind", "difference"), "--group-column-split or --long-short"),
        (("--kind", "mean", "--formula", "1"), "--formula does not apply"),
        (("--kind", "simex", "--outcome", "n_words", "--cluster"), "--cluster does not apply"),
        (("--kind", "median"), "--kind"),
        (("--kind", "ols-predictor", "--outcome", "n_words", "--variant", "x"), "ppi or ludwig"),
        (
            ("--kind", "ols-predictor", "--outcome", "n_words", "--coefficient", "construct"),
            "is a position",
        ),
        (
            ("--kind", "mean", "--prediction-column", "nonesuch", "--pi-column", "pi"),
            "--prediction-column 'nonesuch'",
        ),
        (
            ("--kind", "difference", "--group-column-split", "level", "--pi-column", "pi"),
            "boolean or 0/1",
        ),
    ],
)
def test_estimate_refuses_missing_stray_and_unusable_options(
    runner: CliRunner,
    files: dict[str, Path],
    drawn_gold: Path,
    tmp_path: Path,
    extra: tuple[str, ...],
    message: str,
) -> None:
    out = tmp_path / "e.json"
    result = runner.invoke(app, estimate_args(files, drawn_gold, out, *extra))
    assert result.exit_code == 1
    assert message in result.stderr
    assert not out.exists()


def test_estimate_refuses_an_output_that_is_not_json_or_has_no_folder(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    not_json = runner.invoke(
        app, estimate_args(files, drawn_gold, tmp_path / "e.csv", "--kind", "mean")
    )
    assert not_json.exit_code == 1 and ".json" in not_json.stderr
    nowhere = tmp_path / "absent" / "e.json"
    no_folder = runner.invoke(app, estimate_args(files, drawn_gold, nowhere, "--kind", "mean"))
    assert (
        no_folder.exit_code == 1 and "folder of the output file does not exist" in no_folder.stderr
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--kind", "mean"), "--kind mean needs --draw or --pi-column"),
        (("--kind", "ols-outcome", "--formula", "1"), "corrections sample"),
        (
            (
                "--kind",
                "ols-predictor",
                "--outcome",
                "n_words",
                "--variant",
                "x",
                "--pi-column",
                "pi",
            ),
            "ppi or ludwig",
        ),
        (
            ("--kind", "ols-predictor", "--outcome", "n_words", "--coefficient", "slope"),
            "is a position",
        ),
    ],
)
def test_estimate_refuses_option_problems_before_reading_any_data(
    runner: CliRunner,
    files: dict[str, Path],
    drawn_gold: Path,
    tmp_path: Path,
    extra: tuple[str, ...],
    message: str,
) -> None:
    args = estimate_args(files, drawn_gold, tmp_path / "e.json", *extra)
    args[args.index("--gold-column") + 1] = "absent_gold"
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert message in result.stderr and "absent_gold" not in result.stderr


def test_estimate_refuses_a_coded_sample_that_leaves_texts_of_a_drawn_group_out(
    runner: CliRunner, files: dict[str, Path], drawn_gold: Path, tmp_path: Path
) -> None:
    coded = pd.read_csv(drawn_gold)
    dropped = str(coded["text_id"].iloc[1])
    gapped = tmp_path / "gapped.csv"
    coded.drop(index=1).to_csv(gapped, index=False)
    out = tmp_path / "e.json"
    result = runner.invoke(
        app, estimate_args(files, gapped, out, "--kind", "mean", "--pi-column", "pi")
    )
    assert result.exit_code == 1 and not out.exists()
    assert "whole groups" in result.stderr and repr(dropped) in result.stderr
    moments = ("--kind", "moment-subtraction", "--outcome", "n_words", "--n-boot", "50")
    written = run_estimate(runner, files, gapped, tmp_path / "m.json", *moments)
    assert written["estimate"]["n_labelled"] == N_DRAWN - 1  # type: ignore[index]
