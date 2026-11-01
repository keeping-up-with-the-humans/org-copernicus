"""Whether a human-coded sample covers whole groups, as the group draw codes them.

``draw_human_sample`` draws whole groups and gives every text of a drawn group the inclusion
probability ``m / G``. That probability is only right when every text of a drawn group is coded:
a drawn group with an uncoded text holds a sample of its own texts, so its texts had a smaller
chance than ``m / G``. ``attach_draw`` refuses such gaps against the draw; ``require_whole_groups``
checks coded data that come without the draw, such as a coded file read by the command line.
``design_problem`` checks that a draw's counts describe its drawn groups (``m`` of them out of
``n_groups``, ``pi = m / n_groups``), for ``attach_draw`` and for ``read_draw`` alike.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.data import CorrectionData
from copernicus.errors import ConfigurationError

if TYPE_CHECKING:
    # Type-only: sampling imports this module, so a runtime import would be circular.
    from copernicus.corrections.sampling import SampleDraw

logger = logging.getLogger(__name__)

__all__ = ["PI_TOLERANCE", "design_problem", "require_whole_groups", "uncoded_in_groups"]

PI_TOLERANCE = 1e-12
"""The relative tolerance within which a recorded ``pi`` must equal ``m / n_groups``."""


def design_problem(draw: SampleDraw, n_groups: int, against: str) -> str:
    """Why a draw's ``m``, ``n_groups`` or ``pi`` contradict its drawn groups, or ``""``.

    Args:
        draw: The draw.
        n_groups: The number of groups the draw must record: the data's for ``attach_draw``,
            its own for ``read_draw``; at least 1.
        against: Where ``n_groups`` comes from, for the message (``the data``).

    Returns:
        ``""`` when ``m`` is the number of distinct drawn groups, ``n_groups`` the given number
        and ``pi`` equals ``m / n_groups``; otherwise the recorded and the expected values.
    """
    m = len({str(group) for group in draw.groups})
    if (draw.m, draw.n_groups) == (m, n_groups) and math.isclose(
        draw.pi, m / n_groups, rel_tol=PI_TOLERANCE
    ):
        return ""
    return (
        f"it records m = {draw.m}, n_groups = {draw.n_groups} and pi = {draw.pi!r}; its drawn "
        f"groups and {against} give m = {m}, n_groups = {n_groups} and pi = {m / n_groups!r}"
    )


def uncoded_in_groups(data: CorrectionData, coded: set[str], groups: np.ndarray) -> list[str]:
    """The texts of the given groups that are not among the coded texts.

    Args:
        data: The correction data.
        coded: Ids of the coded texts.
        groups: The groups to check.

    Returns:
        The ids of the uncoded texts of those groups, in data order.
    """
    in_group = np.isin(data.groups, groups)
    return [text_id for text_id in data.ids[in_group].tolist() if text_id not in coded]


def require_whole_groups(data: CorrectionData) -> None:
    """Refuse labelled texts that do not cover whole groups.

    Args:
        data: The correction data; its labelled texts are the coded sample.

    Raises:
        ConfigurationError: If a group with a labelled text also holds a text without a gold
            value (the message names the first ten such texts).
    """
    labelled = data.labelled_mask
    left_out = uncoded_in_groups(data, set(data.labelled_ids.tolist()), data.groups[labelled])
    if left_out:
        raise ConfigurationError(
            "the coded texts must cover whole groups, as the group draw of draw_human_sample "
            "(corrections sample) codes them, so that pi = m / G holds; groups with a coded text "
            f"also hold texts without a gold value: {list_first_ten(left_out)} (code them, or "
            "give every text its own group when the texts were drawn one by one)"
        )
    logger.debug("the %d coded texts cover whole groups", int(labelled.sum()))
