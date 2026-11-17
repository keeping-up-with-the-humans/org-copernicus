"""The level orders of the one-stage prompt in the identity of a run store.

The one-stage prompt lists the level descriptions of each marker in the marker's level order, the
order in which the rubric file lists them (for a marker built in code the order it is given, by
default from 5 down to 1), and `copernicus.rubric.model.Rubric.content_hash` does not see that
order. A one-stage store therefore records it under ``level_orders``, every marker of every
attribute, but only when
the levels of some marker are not listed from 5 down to 1: an earlier version of the library
listed every marker's levels from 5 down to 1 and recorded no order, so a one-stage store of a
rubric listed so keeps the identity it had, and every other store is told apart. The two-stage
prompts do not depend on the order, and their stores record none.
"""

from __future__ import annotations

import json
import logging

from copernicus.rubric.model import DEFAULT_LEVEL_ORDER, Rubric

logger = logging.getLogger(__name__)

__all__ = [
    "LEVEL_ORDERS_DIFFER",
    "LEVEL_ORDERS_KEY",
    "LEVEL_ORDERS_MERGE",
    "LEVEL_ORDERS_MISSING",
    "level_order_settings",
    "level_orders_reason",
]

LEVEL_ORDERS_KEY = "level_orders"
"""The run_info key under which a one-stage store records the level order of every marker."""
LEVEL_ORDERS_MISSING = (
    "the store was coded with the level descriptions of every marker listed from level 5 down "
    "to level 1 in the one-stage prompt, as an earlier version of the library listed them for "
    "every rubric; this version lists them in the order of the rubric file, and this rubric "
    "lists the levels of some marker in another order, so the prompt of the one-stage coder has "
    "changed for this rubric: code the texts into a new store, or list the levels of every "
    "marker from 5 down to 1 in the rubric file to continue the old coding (the content hash of "
    "the rubric stays the same)"
)
"""Why a resume refuses a store without ``level_orders`` for this rubric, and what to do."""
LEVEL_ORDERS_DIFFER = (
    "the one-stage prompt lists the level descriptions of each marker in the order of the rubric "
    "file (a marker built in code lists them in its level_order, by default from 5 down to 1), "
    "and this rubric lists the levels in another order than the rubric the store was coded with, "
    "so the prompt of the one-stage coder has changed: code the texts into a new store, or list "
    "the levels in the rubric file in the order that the store records under level_orders (the "
    "content hash of the rubric stays the same)"
)
"""Why a resume refuses a store whose ``level_orders`` the rubric does not give, and what to do."""
LEVEL_ORDERS_MERGE = (
    "the stores were coded with one-stage prompts that list the level descriptions of the "
    "markers in different orders: a store records the order of the rubric file under "
    "level_orders, and none when every marker's levels were listed from 5 down to 1, as an "
    "earlier version of the library listed them for every rubric; code the texts of one store "
    "again with the level order of the other, then merge"
)
"""Why a merge refuses stores that differ in ``level_orders``, and what to do."""


def level_order_settings(rubric: Rubric) -> dict[str, str]:
    """Build the entry of the level orders in the ``settings`` of a one-stage `RunIdentity`.

    Args:
        rubric: The rubric of the run.

    Returns:
        An empty mapping when every marker of the rubric lists its levels from 5 down to 1;
        otherwise ``{"level_orders": ...}``, a JSON object of each marker id, in rubric order,
        to the marker's level order.
    """
    orders = {
        marker.marker_id: marker.level_order
        for attribute in rubric.attributes
        for marker in attribute.markers
    }
    if all(order == DEFAULT_LEVEL_ORDER for order in orders.values()):
        return {}
    return {LEVEL_ORDERS_KEY: json.dumps({key: list(order) for key, order in orders.items()})}


def level_orders_reason(key: str, reason: str) -> str:
    """Return the reason to append to a refusal on a ``run_info`` key.

    Args:
        key: The key the refusal names.
        reason: Why a refusal on ``level_orders`` is made, and what to do.

    Returns:
        ``": <reason>"`` for ``level_orders``, and the empty string for any other key, whose
        refusal names the key and the values alone.
    """
    return f": {reason}" if key == LEVEL_ORDERS_KEY else ""
