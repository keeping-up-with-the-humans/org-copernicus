"""The helpers that moved out of the study and the estimates keep their old names there.

Code that imported them from their old modules, such as ``from copernicus.corrections.study
import _draw``, gets the moved objects themselves.
"""

from __future__ import annotations

import pytest

from copernicus.corrections import _estimate_inputs, _study_draws

pytestmark = pytest.mark.filterwarnings("error")


def test_the_study_keeps_the_old_names_of_its_moved_helpers() -> None:
    from copernicus.corrections.study import (
        _crossfit_se,
        _Draw,
        _draw,
        _estimate,
        _plugged_column,
    )

    assert _Draw is _study_draws.StudyDraw
    assert _draw is _study_draws.study_draw
    assert _estimate is _study_draws.draw_estimate
    assert _plugged_column is _study_draws.plugged_column
    assert _crossfit_se is _study_draws.crossfit_se


def test_the_estimates_keep_the_old_name_of_their_moved_helper() -> None:
    from copernicus.corrections.estimates import _parts

    assert _parts is _estimate_inputs.estimand_parts
