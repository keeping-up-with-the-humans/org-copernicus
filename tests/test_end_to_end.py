"""The end-to-end example runs the whole pipeline on the toy corpus and writes its results."""

from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
from embeddings_support import MINILM_AVAILABLE, SKIP_REASON

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = LIBRARY_ROOT / "examples" / "end_to_end.py"
WALL_TIME_LIMIT_SECONDS = 120.0
CODERS = ("two_stage", "one_stage", "similarity", "hybrid")

pytestmark = [
    pytest.mark.embed,
    pytest.mark.slow,
    pytest.mark.skipif(not MINILM_AVAILABLE, reason=SKIP_REASON),
]


@dataclass(frozen=True)
class Example:
    """One run of the example.

    Attributes:
        module: The example, loaded as a module.
        out_dir: The folder the run wrote to.
        summary: The dictionary that ``run`` returned.
        seconds: Wall time of ``run``.
    """

    module: ModuleType
    out_dir: Path
    summary: dict[str, Any]
    seconds: float


def load_example() -> ModuleType:
    """Load ``examples/end_to_end.py`` as a module; ``end_to_end_steps`` must be importable."""
    specification = importlib.util.spec_from_file_location("end_to_end_example", EXAMPLE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(EXAMPLE.parent))
        specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example(tmp_path_factory: pytest.TempPathFactory) -> Example:
    """Run the example once and time it."""
    module = load_example()
    out_dir = tmp_path_factory.mktemp("end_to_end")
    started = time.perf_counter()
    summary = module.run(out_dir)
    return Example(module, out_dir, summary, time.perf_counter() - started)


def test_the_run_finishes_within_the_wall_time_limit(example: Example) -> None:
    assert example.seconds < WALL_TIME_LIMIT_SECONDS


def test_every_coder_is_validated_against_the_gold_value(example: Example) -> None:
    assert example.summary["two_stage"]["rank_validity"] >= 0.5
    for coder in CODERS:
        record = example.summary[coder]
        assert record["n_used"] == example.summary["n_texts"]
        low, value, high = (
            record["rank_validity_low"],
            record["rank_validity"],
            record["rank_validity_high"],
        )
        assert 0.0 <= low <= high <= 1.0 and math.isfinite(value)


def test_every_listed_file_exists_and_holds_rows(example: Example) -> None:
    names = example.module.CSV_FILES
    assert len(names) == len(set(names))
    for name in (
        *names,
        "summary.json",
        "validity_report.txt",
        "two_stage.sqlite",
        "one_stage.sqlite",
    ):
        assert (example.out_dir / name).stat().st_size > 0, name
    for name in set(names) - {"battery_skipped.csv"}:
        assert len(pd.read_csv(example.out_dir / name)) > 0, name


def test_summary_json_holds_the_returned_summary(example: Example) -> None:
    written = json.loads((example.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(example.summary))


def test_the_validity_report_names_every_coder(example: Example) -> None:
    report = (example.out_dir / "validity_report.txt").read_text(encoding="utf-8")
    for coder in CODERS:
        assert coder in report


def test_the_human_sample_draws_whole_groups_and_supports_both_estimators(
    example: Example,
) -> None:
    sample = example.summary["human_sample"]
    assert sample["n_texts"] == 18 and sample["n_groups"] == 6
    assert sample["n_texts"] == sample["n_groups"] * example.summary["texts_per_group"]
    difference = example.summary["ppi_difference"]
    assert difference["n_labelled_groups_long"] >= 2 and difference["n_labelled_groups_short"] >= 2
    for name in ("ppi_mean", "ppi_difference"):
        record = example.summary[name]
        assert record["ci_low"] <= record["point"] <= record["ci_high"]
        assert record["se"] > 0.0
    assert example.summary["ppi_mean"]["clustered"] is True


def test_the_cross_fit_calibration_is_scored_against_the_gold(example: Example) -> None:
    cross_fit = example.summary["cross_fit"]
    assert cross_fit["n_texts"] == example.summary["n_texts"]
    assert cross_fit["mae_corrected"] < cross_fit["mae_raw"]


def test_the_baseline_runs_at_both_sizes(example: Example) -> None:
    sizes = [record["size"] for record in example.summary["baseline"]]
    assert sizes == ["20", "full"]


def test_the_battery_scores_all_three_corrections_at_the_one_size(
    example: Example,
) -> None:
    assert example.summary["battery_rows"] >= 3
    results = pd.read_csv(example.out_dir / "battery_results.csv")
    scored = results[~results["diagnostic"]]
    assert set(scored["family"]) == {
        "regression_calibration",
        "supervised_stacking",
        "correction_ensemble",
    }
    assert set(scored["size"].astype(str)) == {"20"}
    assert example.summary["battery_skipped"] == len(
        pd.read_csv(example.out_dir / "battery_skipped.csv")
    )


def test_the_script_prints_the_validity_report_and_where_it_wrote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_example()
    module.main([str(tmp_path / "out"), "--n-boot", "20"])
    printed = capsys.readouterr().out
    assert "rank validity" in printed and "summary.json" in printed
    assert (tmp_path / "out" / "summary.json").is_file()
