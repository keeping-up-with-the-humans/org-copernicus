"""Whether a coded sample covers whole groups, as the group draw codes them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from estimates_support import clustered_frame

from copernicus.corrections import CorrectionData, attach_draw, draw_groups
from copernicus.corrections._whole_groups import require_whole_groups, uncoded_in_groups
from copernicus.errors import ConfigurationError

pytestmark = pytest.mark.filterwarnings("error")


def _coded(frame: pd.DataFrame, coded: np.ndarray) -> CorrectionData:
    """The texts with gold on the coded ones only."""
    gold = np.where(coded, frame["gold"].to_numpy(), np.nan)
    return CorrectionData(frame=frame.assign(gold=gold), score_columns=("score",))


def test_a_whole_group_draw_passes_and_a_gap_is_named() -> None:
    frame = clustered_frame(np.random.default_rng(1), (3, 4, 2, 5, 3))
    base = _coded(frame, np.zeros(len(frame), dtype=bool))
    draw = draw_groups(base.ids, base.groups, base.ids, 6, seed=3)
    drawn = attach_draw(base, pd.Series(frame["gold"].to_numpy(), index=base.ids), draw)
    require_whole_groups(drawn)
    gap = drawn.labelled_mask.copy()
    first = int(np.flatnonzero(gap)[0])
    gap[first] = False
    gapped = _coded(frame, gap)
    with pytest.raises(ConfigurationError, match=rf"whole groups.*'{drawn.ids[first]}'"):
        require_whole_groups(gapped)
    groups = np.unique(gapped.groups[gapped.labelled_mask])
    assert uncoded_in_groups(gapped, set(gapped.labelled_ids.tolist()), groups) == [
        str(drawn.ids[first])
    ]


def test_texts_drawn_one_by_one_pass_when_each_is_its_own_group() -> None:
    frame = clustered_frame(np.random.default_rng(2), (3, 4, 2, 5, 3))
    coded = np.arange(len(frame)) % 4 == 0
    singles = _coded(frame.assign(group_id=frame["text_id"]), coded)
    require_whole_groups(singles)
    with pytest.raises(ConfigurationError, match="whole groups"):
        require_whole_groups(_coded(frame, coded))


def test_attach_draw_refuses_a_draw_of_the_table_whose_ids_leave_out_a_text() -> None:
    frame = clustered_frame(np.random.default_rng(4), (3, 4, 2, 5, 3))
    base = _coded(frame, np.zeros(len(frame), dtype=bool))
    draw = draw_groups(base.ids, base.groups, base.ids, 6, seed=3)
    partial = draw.__class__(**{**draw.__dict__, "ids": draw.ids[1:]})
    with pytest.raises(ConfigurationError, match="missing from the drawn ids"):
        attach_draw(base, pd.Series(frame["gold"].to_numpy(), index=base.ids), partial)
