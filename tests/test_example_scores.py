"""The end-to-end example joins the construct scores of its two coders on the text id."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from script_rules import check_script_rules

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "end_to_end.py"


def load_example() -> ModuleType:
    """Load ``examples/end_to_end.py`` as a module; loading it needs no encoder."""
    specification = importlib.util.spec_from_file_location("end_to_end_scores", EXAMPLE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(EXAMPLE.parent))
        specification.loader.exec_module(module)
    return module


def test_the_construct_scores_of_the_two_coders_are_joined_on_the_text_id() -> None:
    example = load_example()
    two_stage = pd.DataFrame({"text_id": ["a", "b", "c"], "construct_score": [1.0, 2.0, 3.0]})
    one_stage = pd.DataFrame({"text_id": ["c", "a", "b"], "construct_score": [30.0, 10.0, 20.0]})
    wide = example.join_construct_scores(two_stage, one_stage)
    assert wide.to_dict("list") == {
        "text_id": ["a", "b", "c"],
        "two_stage": [1.0, 2.0, 3.0],
        "one_stage": [10.0, 20.0, 30.0],
    }


def test_two_exports_of_different_texts_are_refused() -> None:
    example = load_example()
    two_stage = pd.DataFrame({"text_id": ["a", "b"], "construct_score": [1.0, 2.0]})
    one_stage = pd.DataFrame({"text_id": ["a", "z"], "construct_score": [1.0, 2.0]})
    with pytest.raises(ValueError, match="same texts"):
        example.join_construct_scores(two_stage, one_stage)


def test_the_example_and_its_steps_use_only_public_names_and_stay_short() -> None:
    check_script_rules(EXAMPLE.parent, ("end_to_end.py", "end_to_end_steps.py"))


def test_the_example_takes_its_steps_from_the_module_beside_it() -> None:
    example = load_example()
    steps = sys.modules["end_to_end_steps"]
    for name in steps.__all__:
        if name in vars(example):
            assert getattr(example, name) is getattr(steps, name), name
    assert "join_construct_scores" in vars(example)
