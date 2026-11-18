"""The bootstrap floor of ``corrections estimate``: ``--min-defined`` and ``--below-floor``."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from cli_support import summary_value
from estimates_support import SPARSE_BOOT, SPARSE_SEED, classical_data, sparse_covariation_data
from typer.testing import CliRunner

from copernicus.cli._embedding_options import read_table
from copernicus.cli._inputs import read_gold
from copernicus.cli.app import app
from copernicus.corrections import CorrectionData, Estimate, moment_subtraction, oriv, simex
from copernicus.data import read_frame

SCORES = ("score_a", "score_b")
KINDS: dict[str, tuple[str, ...]] = {
    "moment-subtraction": ("--outcome", "outcome", "--prediction-column", "score_a"),
    "simex": ("--outcome", "outcome", "--prediction-column", "score_a"),
    "oriv": ("--outcome", "outcome", "--score-a", "score_a", "--score-b", "score_b"),
}
LIBRARY: dict[str, Callable[..., Estimate]] = {
    "moment-subtraction": lambda data, **options: moment_subtraction(
        data, "outcome", score_column="score_a", **options
    ),
    "simex": lambda data, **options: simex(data, "outcome", score_column="score_a", **options),
    "oriv": lambda data, **options: oriv(
        data, "outcome", score_a="score_a", score_b="score_b", **options
    ),
}
BOOT = ("--n-boot", str(SPARSE_BOOT), "--seed", str(SPARSE_SEED))
# The logger of each kind's estimator, which the command shows a warning under.
MODULES = {
    "moment-subtraction": "copernicus.corrections.estimates_classical",
    "simex": "copernicus.corrections.estimates_classical",
    "oriv": "copernicus.corrections.estimates_oriv",
}
# A floor or a rule outside its range: the options, the exit code and the message. The parser
# of the command line refuses a floor that it does not read as an integer, with exit code 2 and
# its own wording, of which every version starts with the words below; whether it reads "10.0"
# as 10 differs between its versions, so no case asks. The command refuses the others with exit
# code 1 and one line "error: <the whole sentence>".
REFUSALS: dict[str, tuple[tuple[str, ...], int, str]] = {
    "floor 1": (
        ("--n-boot", "60", "--min-defined", "1"),
        1,
        "error: min_defined must be an integer of at least 2; got 1",
    ),
    "floor 0": (
        ("--n-boot", "60", "--min-defined", "0"),
        1,
        "error: min_defined must be an integer of at least 2; got 0",
    ),
    "floor above the resamples": (
        ("--n-boot", "60", "--min-defined", "61"),
        1,
        "error: n_boot must be an integer of at least 61; got 60",
    ),
    "floor written true": (
        ("--min-defined", "true"),
        2,
        "Invalid value for '--min-defined':",
    ),
    "floor written 10.5": (
        ("--min-defined", "10.5"),
        2,
        "Invalid value for '--min-defined':",
    ),
    "unknown rule": (
        ("--below-floor", "pont"),
        1,
        "error: --below-floor must be refuse or point; got 'pont'",
    ),
    "resamples below the floor": (
        ("--n-boot", "9", "--min-defined", "10"),
        1,
        "error: n_boot must be an integer of at least 10; got 9",
    ),
}


def write_files(frame: pd.DataFrame, folder: Path) -> dict[str, Path]:
    """The texts of a frame as a texts, a scores and a gold file, rows in the frame's order."""
    paths = {name: folder / f"{name}.csv" for name in ("texts", "scores", "gold")}
    texts = frame[["text_id", "group_id", "outcome"]].assign(text="a text of the corpus")
    texts.to_csv(paths["texts"], index=False)
    frame[["text_id", *SCORES]].to_csv(paths["scores"], index=False)
    frame.loc[frame["gold"].notna(), ["text_id", "gold"]].to_csv(paths["gold"], index=False)
    return paths


@pytest.fixture(scope="module")
def sparse_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """``sparse_covariation_data``, on which most resamples are undefined, as files."""
    return write_files(sparse_covariation_data(0).frame, tmp_path_factory.mktemp("sparse"))


@pytest.fixture(scope="module")
def defined_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Texts on which every resample of the three estimators is defined, as files."""
    frame = classical_data(3, n_texts=400, n_labelled=60).frame
    return write_files(frame, tmp_path_factory.mktemp("defined"))


@pytest.fixture
def unreadable_files(tmp_path: Path) -> dict[str, Path]:
    """A texts, a scores and a gold file that exist and fail as soon as they are read."""
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("texts", "scores", "gold")}
    for path in paths.values():
        path.write_text("this line is not JSON\n", encoding="utf-8")
    return paths


def command_args(files: dict[str, Path], out: Path, *extra: str) -> list[str]:
    """The arguments of ``corrections estimate`` on a texts, a scores and a gold file."""
    return [
        *("corrections", "estimate", "--scores", str(files["scores"])),
        *("--texts", str(files["texts"]), "--gold", str(files["gold"]), "--gold-column", "gold"),
        *("--score-columns", ",".join(SCORES), "--group-column", "group_id"),
        *("--out", str(out), *extra),
    ]


def files_data(files: dict[str, Path]) -> CorrectionData:
    """The correction data the command builds from a texts, a scores and a gold file.

    The scores are read by the library's reader, as the command reads them: the default parser
    of a pandas before 3.1 does not round every one of their numbers correctly, which the
    bootstrap of ``simex`` carries into the tenth digit of its standard error.
    """
    return CorrectionData.from_scores(
        read_frame(files["scores"], protected_columns=("text_id",)),
        read_table(files["texts"], "text_id", "text", "group_id"),
        read_gold(files["gold"]),
        gold_column="gold",
        score_columns=SCORES,
        extra_columns=("outcome",),
    )


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_by_default_the_command_refuses_too_few_defined_resamples(
    runner: CliRunner, sparse_files: dict[str, Path], tmp_path: Path, kind: str
) -> None:
    out = tmp_path / "refused.json"
    result = runner.invoke(
        app, command_args(sparse_files, out, "--kind", kind, *KINDS[kind], *BOOT)
    )
    assert result.exit_code == 1 and not out.exists()
    assert (
        f"of {SPARSE_BOOT} bootstrap resamples have a defined estimate; the standard error needs "
        "at least 50" in result.stderr
    )


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_below_the_floor_the_command_keeps_the_point_and_writes_no_interval(
    runner: CliRunner, sparse_files: dict[str, Path], tmp_path: Path, kind: str
) -> None:
    out = tmp_path / "kept.json"
    extra = ("--kind", kind, *KINDS[kind], *BOOT, "--below-floor", "point")
    result = runner.invoke(app, command_args(sparse_files, out, *extra))
    assert result.exit_code == 0, result.stderr
    written = json.loads(out.read_text(encoding="utf-8"))
    direct = LIBRARY[kind](
        files_data(sparse_files), n_boot=SPARSE_BOOT, seed=SPARSE_SEED, below_floor="point"
    )
    estimate = written["estimate"]
    assert estimate["point"] == pytest.approx(direct.point, rel=1e-12)
    assert (estimate["se"], estimate["ci_low"], estimate["ci_high"]) == (None, None, None)
    defined = direct.details["n_boot_defined"]
    details = estimate["details"]
    assert (details["n_boot_defined"], details["min_defined"], details["below_floor"]) == (
        defined,
        50,
        "point",
    )
    assert details["se_status"] == "too_few_defined"
    assert written["options"]["below_floor"] == "point"
    warned = f"WARNING {MODULES[kind]}: {direct.method}: only {defined} of {SPARSE_BOOT} "
    assert warned in result.stderr
    assert summary_value(result.stdout, "estimate") == f"{direct.point:.4g} (no standard error)"
    assert summary_value(result.stdout, "95% interval") == "none"


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_the_command_passes_the_floor_to_each_bootstrapped_kind(
    runner: CliRunner, sparse_files: dict[str, Path], tmp_path: Path, kind: str
) -> None:
    out = tmp_path / "floor.json"
    extra = ("--kind", kind, *KINDS[kind], *BOOT, "--min-defined", "10")
    result = runner.invoke(app, command_args(sparse_files, out, *extra))
    assert result.exit_code == 0, result.stderr
    written = json.loads(out.read_text(encoding="utf-8"))
    direct = LIBRARY[kind](
        files_data(sparse_files), n_boot=SPARSE_BOOT, seed=SPARSE_SEED, min_defined=10
    )
    estimate = written["estimate"]
    assert estimate["point"] == pytest.approx(direct.point, rel=1e-12)
    assert estimate["se"] == pytest.approx(direct.se, rel=1e-12)
    details = estimate["details"]
    assert (details["min_defined"], details["below_floor"], details["se_status"]) == (
        10,
        "refuse",
        "ok",
    )
    assert written["options"]["min_defined"] == 10 and "below_floor" not in written["options"]
    assert summary_value(result.stdout, "estimate") == f"{direct.point:.4g} (SE {direct.se:.4g})"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            ("--kind", "simex", "--outcome", "outcome", "--below-floor", "pont"),
            "--below-floor must be refuse or point; got 'pont'",
        ),
        (("--kind", "mean", "--min-defined", "10"), "--min-defined does not apply to --kind mean"),
        (
            ("--kind", "ols-predictor", "--outcome", "outcome", "--below-floor", "point"),
            "--below-floor does not apply to --kind ols-predictor",
        ),
    ],
)
def test_the_command_refuses_floor_options_it_cannot_use_before_reading_any_data(
    runner: CliRunner,
    unreadable_files: dict[str, Path],
    tmp_path: Path,
    extra: tuple[str, ...],
    message: str,
) -> None:
    out = tmp_path / "e.json"
    result = runner.invoke(app, command_args(unreadable_files, out, *extra))
    assert result.exit_code == 1 and not out.exists()
    assert f"error: {message}" in result.stderr.splitlines()
    assert "cannot read" not in result.stderr


def test_the_unreadable_files_fail_as_soon_as_the_command_reads_them(
    runner: CliRunner, unreadable_files: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "e.json"
    extra = ("--kind", "simex", *KINDS["simex"], "--below-floor", "point")
    result = runner.invoke(app, command_args(unreadable_files, out, *extra))
    assert result.exit_code == 1 and not out.exists()
    assert "error: cannot read '" in result.stderr
    assert "line 1 is not valid JSON" in result.stderr


def test_the_command_reports_a_floor_above_the_resamples_and_names_both_options(
    runner: CliRunner, sparse_files: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "e.json"
    extra = ("--kind", "oriv", *KINDS["oriv"], "--n-boot", "20", "--min-defined", "30")
    result = runner.invoke(app, command_args(sparse_files, out, *extra))
    assert result.exit_code == 1 and not out.exists()
    assert "n_boot must be an integer of at least 30; got 20" in result.stderr
    shown = runner.invoke(app, ["corrections", "estimate", "--help"], env={"COLUMNS": "200"})
    assert "--min-defined" in shown.stdout and "--below-floor" in shown.stdout


@pytest.mark.parametrize("case", sorted(REFUSALS))
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_bootstrapped_kind_refuses_a_floor_or_a_rule_outside_its_range(
    runner: CliRunner, sparse_files: dict[str, Path], tmp_path: Path, kind: str, case: str
) -> None:
    extra, code, message = REFUSALS[case]
    out = tmp_path / "refused.json"
    arguments = command_args(sparse_files, out, "--kind", kind, *KINDS[kind], *extra)
    result = runner.invoke(app, arguments, env={"COLUMNS": "200"})
    assert result.exit_code == code
    assert not out.exists()
    if code == 1:
        assert message in result.stderr.splitlines()
    else:
        assert message in " ".join(result.stderr.replace("│", " ").split())


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_bootstrapped_kind_takes_ten_resamples_at_a_floor_of_ten(
    runner: CliRunner, defined_files: dict[str, Path], tmp_path: Path, kind: str
) -> None:
    out = tmp_path / "ten.json"
    options = ("--n-boot", "10", "--min-defined", "10", "--seed", str(SPARSE_SEED))
    result = runner.invoke(
        app, command_args(defined_files, out, "--kind", kind, *KINDS[kind], *options)
    )
    assert result.exit_code == 0, result.stderr
    estimate = json.loads(out.read_text(encoding="utf-8"))["estimate"]
    assert all(math.isfinite(estimate[field]) for field in ("point", "se", "ci_low", "ci_high"))
    assert estimate["ci_low"] < estimate["point"] < estimate["ci_high"]
    details = estimate["details"]
    assert (details["n_boot"], details["n_boot_defined"], details["min_defined"]) == (10, 10, 10)
    assert details["se_status"] == "ok"
    direct = LIBRARY[kind](files_data(defined_files), n_boot=10, min_defined=10, seed=SPARSE_SEED)
    assert estimate["se"] == pytest.approx(direct.se, rel=1e-12)
