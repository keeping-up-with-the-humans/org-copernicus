"""The size planner: the effective-size factor and the plan table."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import copernicus
from copernicus import corrections
from copernicus.corrections import planning
from copernicus.corrections.planning import SizePlan, plan_sample_size, plan_table
from copernicus.errors import SettingsError


def test_the_planner_is_exported_from_the_package_and_the_top_level() -> None:
    assert corrections.SizePlan is SizePlan
    assert corrections.plan_sample_size is plan_sample_size
    assert corrections.plan_table is plan_table
    assert copernicus.plan_sample_size is plan_sample_size
    assert planning.__all__ == ["SizePlan", "plan_sample_size", "plan_table"]


def test_plan_sample_size_formulas() -> None:
    plan = plan_sample_size(0.70, 100, 1000)
    assert isinstance(plan, SizePlan)
    assert plan.factor == pytest.approx(1.79, abs=0.01)
    assert plan.limit == pytest.approx(1.96, abs=0.01)
    assert plan.uncorrected_equivalent == pytest.approx(1 / 0.49)
    assert plan.factor == pytest.approx(1 / (0.51 + 0.49 * 100 / 1000))
    assert (plan.rho, plan.n_human, plan.n_machine) == (0.70, 100, 1000)
    assert plan_sample_size(0.5, 50, 50).factor == pytest.approx(1.0)


def test_the_docstrings_name_the_correlation_and_what_each_number_means() -> None:
    def flat(text: str | None) -> str:
        return " ".join((text or "").split())

    plan, sample = flat(SizePlan.__doc__), flat(plan_sample_size.__doc__)
    assert "rho: Pearson correlation" in plan
    assert "upper limit of the multiplier" in plan
    assert "machine-scored texts that, used without correction" in plan
    assert "human texts per machine-scored text" not in plan
    for text in (flat(planning.__doc__), sample, flat(plan_table.__doc__)):
        assert "Pearson correlation" in text
    assert "partial correlation" in sample and "large-sample" in sample


DOCS = Path(__file__).resolve().parents[2] / "docs"
ILLUSTRATED = ("tutorials/sample-size.md", "methods/corrections/index.md")
DEFAULT_WEIGHT_WORTH = "n_human / (1 + (1 - n_human / n_machine) * (c^2 - 2 * rho * c))"
NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
ILLUSTRATION = re.compile(
    rf"With `rho = {NUMBER}`, `c = {NUMBER}`, `n_human = {NUMBER}` and `n_machine = {NUMBER}`, "
    rf".*? the plan promises an effective size of {NUMBER} texts and the default weight is "
    rf"worth {NUMBER}, from `([^`]+)`"
)


def default_weight_worth(rho: float, c: float, n_human: int, n_machine: int) -> float:
    """The large-sample effective size of the prediction-powered mean with the weight 1.

    For independent texts and the mean of the population they come from, the variance of
    ``mean(y_lab) + mean(s_all) - mean(s_lab)`` is ``var(y) / n * (1 + (1 - n / N) * (c^2 - 2
    rho c))``, with ``c`` the standard deviation of the score over that of the human value.
    """
    return n_human / (1.0 + (1.0 - n_human / n_machine) * (c * c - 2.0 * rho * c))


@pytest.mark.parametrize("page", ILLUSTRATED)
def test_the_pages_illustrate_the_default_weight_with_numbers_from_the_formulas(page: str) -> None:
    text = " ".join((DOCS / page).read_text(encoding="utf-8").split())
    found = ILLUSTRATION.search(text)
    assert found is not None, f"{page}: no illustration of the default weight"
    rho, c = float(found[1]), float(found[2])
    n_human, n_machine = int(found[3]), int(found[4])
    assert found[7] == DEFAULT_WEIGHT_WORTH
    assert round(plan_sample_size(rho, n_human, n_machine).factor * n_human) == int(found[5])
    assert round(default_weight_worth(rho, c, n_human, n_machine)) == int(found[6])


def test_plan_sample_size_refuses_out_of_range_settings() -> None:
    for rho in (0.0, 1.0, -0.2, 1.5, float("nan")):
        with pytest.raises(SettingsError, match="rho"):
            plan_sample_size(rho, 100, 1000)
    with pytest.raises(SettingsError, match="n_human"):
        plan_sample_size(0.7, 1, 1000)
    with pytest.raises(SettingsError, match="n_machine"):
        plan_sample_size(0.7, 100, 99)
    for bad in (100.0, True, "100"):
        with pytest.raises(SettingsError, match="n_human"):
            plan_sample_size(0.7, bad, 1000)  # type: ignore[arg-type]
        with pytest.raises(SettingsError, match="n_machine"):
            plan_sample_size(0.7, 100, bad)  # type: ignore[arg-type]
    assert plan_sample_size(0.7, np.int64(100), np.int32(1000)).n_human == 100


def test_plan_table_has_one_row_per_pair_in_the_given_order() -> None:
    table = plan_table(0.7, [200, 50], [1000, 500, 5000])
    assert list(table.columns) == [
        "rho",
        "n_human",
        "n_machine",
        "factor",
        "limit",
        "uncorrected_equivalent",
    ]
    assert len(table) == 6
    assert table["n_human"].tolist() == [200, 200, 200, 50, 50, 50]
    assert table["n_machine"].tolist() == [1000, 500, 5000, 1000, 500, 5000]
    row = table.iloc[0]
    assert row["factor"] == pytest.approx(plan_sample_size(0.7, 200, 1000).factor)
    assert table["limit"].nunique() == 1


def test_plan_table_refuses_empty_and_inconsistent_grids() -> None:
    with pytest.raises(SettingsError):
        plan_table(0.7, [], [100])
    with pytest.raises(SettingsError, match="n_machine"):
        plan_table(0.7, [200], [100])
