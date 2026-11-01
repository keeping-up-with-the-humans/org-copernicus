"""The size planner: the effective-size factor and the plan table."""

from __future__ import annotations

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
