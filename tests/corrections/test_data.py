"""CorrectionData construction and validation."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from corrections_support import N_ROWS, make_data, make_frame

from copernicus.corrections.data import CorrectionData
from copernicus.errors import ConfigurationError


def test_properties_from_nan_gold() -> None:
    data = make_data()
    assert data.n == N_ROWS
    assert data.labelled_mask.dtype == bool
    assert data.n_labelled == 8
    assert data.labelled_ids.tolist() == [f"t{i:02d}" for i in range(N_ROWS) if i % 3 != 0]
    assert np.isnan(data.gold).sum() == 4
    assert data.ids.tolist()[:2] == ["t00", "t01"]
    assert data.groups.tolist()[:4] == ["g0", "g0", "g0", "g1"]
    assert list(data.scores.columns) == ["good", "poor"]
    assert data.covariate.shape == (N_ROWS,)
    assert data.pi is None
    assert data.embeddings is None


def test_gold_classes_and_binary_flag() -> None:
    frame = make_frame()
    assert not make_data(frame).is_binary_gold
    frame["gold"] = np.where(np.arange(N_ROWS) % 2 == 0, 1.0, 0.0)
    frame.loc[[0, 5], "gold"] = np.nan
    data = make_data(frame)
    assert data.is_binary_gold
    assert data.gold_classes.tolist() == [0.0, 1.0]


def test_input_frame_is_not_modified_and_ids_become_strings() -> None:
    frame = make_frame()
    frame["text_id"] = np.arange(N_ROWS)
    frame.index = frame.index + 100
    data = make_data(frame)
    assert frame["text_id"].dtype.kind == "i"
    assert data.ids.tolist()[:3] == ["0", "1", "2"]
    assert data.frame.index.tolist() == list(range(N_ROWS))


def test_missing_required_column_is_named() -> None:
    frame = make_frame().drop(columns=["poor"])
    with pytest.raises(ConfigurationError, match="poor"):
        make_data(frame)
    with pytest.raises(ConfigurationError, match="group_id"):
        make_data(make_frame().drop(columns=["group_id"]))
    with pytest.raises(ConfigurationError, match="log_chars"):
        make_data(make_frame().drop(columns=["log_chars"]))
    with pytest.raises(ConfigurationError, match="gold"):
        make_data(make_frame().drop(columns=["gold"]))


def test_non_unique_and_missing_ids_are_listed() -> None:
    frame = make_frame()
    frame.loc[5, "text_id"] = "t01"
    with pytest.raises(ConfigurationError, match="t01"):
        make_data(frame)
    frame = make_frame()
    frame.loc[4, "text_id"] = np.nan
    with pytest.raises(ConfigurationError, match="text_id"):
        make_data(frame)


def test_non_finite_score_and_covariate_name_column_and_ids() -> None:
    frame = make_frame()
    frame.loc[2, "good"] = np.nan
    with pytest.raises(ConfigurationError, match=r"good.*t02"):
        make_data(frame)
    frame = make_frame()
    frame.loc[7, "poor"] = np.inf
    with pytest.raises(ConfigurationError, match=r"poor.*t07"):
        make_data(frame)
    frame = make_frame()
    frame.loc[1, "log_chars"] = np.nan
    with pytest.raises(ConfigurationError, match=r"log_chars.*t01"):
        make_data(frame)


def test_error_message_lists_at_most_ten_ids() -> None:
    frame = make_frame(30)
    frame["good"] = np.nan
    with pytest.raises(ConfigurationError) as caught:
        make_data(frame)
    assert "t09" in str(caught.value)
    assert "t10" not in str(caught.value)
    assert "20 more" in str(caught.value)


def test_string_gold_is_refused() -> None:
    frame = make_frame()
    frame["gold"] = ["low", "high"] * (N_ROWS // 2)
    with pytest.raises(ConfigurationError, match="gold"):
        make_data(frame)


def test_infinite_gold_is_refused() -> None:
    frame = make_frame()
    frame.loc[1, "gold"] = np.inf
    with pytest.raises(ConfigurationError, match=r"gold.*t01"):
        make_data(frame)


def test_pi_must_lie_in_zero_one() -> None:
    frame = make_frame()
    frame["pi"] = np.where(np.isnan(frame["gold"]), np.nan, 0.5)
    data = make_data(frame)
    assert data.pi is not None
    assert np.isnan(data.pi).sum() == 4
    for bad in (0.0, 1.5, -0.2):
        frame.loc[1, "pi"] = bad
        with pytest.raises(ConfigurationError, match=r"pi.*t01"):
            make_data(frame)


def test_score_column_declaration_is_checked() -> None:
    with pytest.raises(ConfigurationError, match="score_columns"):
        CorrectionData(frame=make_frame(), score_columns=())
    with pytest.raises(ConfigurationError, match="good"):
        CorrectionData(frame=make_frame(), score_columns=("good", "good"))
    with pytest.raises(ConfigurationError, match="gold"):
        CorrectionData(frame=make_frame(), score_columns=("good", "gold"))


@pytest.mark.parametrize("name", ["text_id", "group_id", "pi"])
def test_reserved_names_are_refused_as_score_gold_or_covariate(name: str) -> None:
    frame = make_frame().assign(pi=0.5)
    with pytest.raises(ConfigurationError, match=name):
        CorrectionData(frame=frame, score_columns=("good", name))
    with pytest.raises(ConfigurationError, match=name):
        CorrectionData(frame=frame, score_columns=("good",), gold_column=name)
    with pytest.raises(ConfigurationError, match=name):
        CorrectionData(frame=frame, score_columns=("good",), covariate_column=name)


def test_importing_corrections_loads_neither_scikit_learn_nor_torch() -> None:
    code = (
        "import sys, copernicus.corrections; "
        "bad = [m for m in ('sklearn', 'torch') if m in sys.modules]; "
        "sys.exit(len(bad))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
