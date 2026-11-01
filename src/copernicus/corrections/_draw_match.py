"""The checks that refuse a draw of another table, or a draw that does not describe its table.

``attach_draw`` runs ``check_draw_matches`` before it sets any human value: the fingerprint of
the draw must be that of the data, and the ids, groups and counts of the draw must describe the
data.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections._whole_groups import design_problem, uncoded_in_groups
from copernicus.errors import ConfigurationError

if TYPE_CHECKING:
    # Type-only: sampling imports this module, so a runtime import would be circular.
    from copernicus.corrections.data import CorrectionData
    from copernicus.corrections.sampling import SampleDraw

__all__ = ["DIFFERENT_TABLE", "NOT_DESCRIBED", "check_draw_matches"]

DIFFERENT_TABLE = (
    "the draw was made on a different table than the data (different texts or groups): draw on "
    "the table the correction data are built from, then attach that draw"
)
NOT_DESCRIBED = "the draw does not describe the table it was made on"


def check_draw_matches(data: CorrectionData, draw: SampleDraw, drawn: list[str]) -> None:
    """Refuse a draw of another table, then a draw whose fields do not describe its table.

    The fingerprint decides first whether the data are the draw's table. On that table a draw
    of ``draw_groups`` passes every further check; they refuse a draw edited or built by hand
    whose ids, groups, counts or ``pi`` contradict the table.
    """
    if data.population_hash != draw.population_hash:
        raise ConfigurationError(f"{DIFFERENT_TABLE}; {_table_counts(data, draw, drawn)}")
    group_of = dict(zip(data.ids.tolist(), data.groups.tolist(), strict=True))
    drawn_groups = {str(group) for group in draw.groups}
    repeated = sorted(text_id for text_id, count in Counter(drawn).items() if count > 1)
    problems = {
        "drawn ids repeat": repeated,
        "drawn texts not in the data": sorted(set(drawn) - group_of.keys()),
        "drawn groups not in the data": sorted(drawn_groups - set(group_of.values())),
        "drawn texts whose group in the data is not a drawn group": [
            text_id
            for text_id in drawn
            if text_id in group_of and group_of[text_id] not in drawn_groups
        ],
        "texts of drawn groups missing from the drawn ids": uncoded_in_groups(
            data, set(drawn), np.array(sorted(drawn_groups))
        ),
    }
    for cause, ids in problems.items():
        if ids:
            raise ConfigurationError(f"{NOT_DESCRIBED}: {cause}: {list_first_ten(ids)}")
    problem = design_problem(draw, len(set(group_of.values())), "the data")
    if problem:
        raise ConfigurationError(f"{NOT_DESCRIBED}: {problem}")


def _table_counts(data: CorrectionData, draw: SampleDraw, drawn: list[str]) -> str:
    """How the table of a refused draw and the data differ: their counts, and the texts lacking.

    The draw records the units and groups of its table, the data hold their texts and groups,
    and the drawn texts that the data lack are named, the first ten in id order.
    """
    n_groups = int(np.unique(data.groups).size)
    counts = (
        f"the draw records n_units = {draw.n_units} and n_groups = {draw.n_groups}, the data "
        f"hold {_counted(data.n, 'text')} in {_counted(n_groups, 'group')}"
    )
    absent = sorted(set(drawn) - set(data.ids.tolist()))
    return f"{counts}; drawn texts not in the data: {list_first_ten(absent)}" if absent else counts


def _counted(number: int, noun: str) -> str:
    """A number with its noun, in the plural unless the number is 1."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"
