"""The draw file: a human-sample draw written to JSON and read back.

``corrections sample`` writes its draw into the ``draw`` block of the description beside the
drawn texts (``<out>.json``), and ``write_draw`` writes the same block for a draw made in
Python. ``read_draw`` reads the block back into a ``SampleDraw``, so the draw can bind the coded
sample to its table later, through ``attach_draw`` or ``corrections estimate --draw``. Writer
and reader use one list of fields, ``DRAW_FIELDS``, the fields of ``SampleDraw``. A file that
lacks a field, as one written by an earlier version without the drawn ids, the groups or the
fingerprint of the table, is refused with "draw again". A file whose counts contradict its
groups is refused too; the part of that message which gives the recorded and the expected
counts is the one ``attach_draw`` gives (``design_problem``), after a prefix of its own.
"""

from __future__ import annotations

import dataclasses
import logging
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from copernicus._io import as_unicode_1d, json_ready, read_sidecar, write_json
from copernicus._version import __version__
from copernicus.corrections._fingerprint import POPULATION_HASH_PATTERN
from copernicus.corrections._messages import list_first_ten
from copernicus.corrections._whole_groups import design_problem
from copernicus.corrections.sampling import SampleDraw
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["DRAW_FIELDS", "DRAW_KEY", "draw_record", "read_draw", "write_draw"]

DRAW_KEY = "draw"
"""The key of the draw's block in a draw file."""
DRAW_FIELDS = tuple(field.name for field in dataclasses.fields(SampleDraw))
"""The fields of the draw block: the fields of ``SampleDraw``, in its order."""
ANALYSIS = "human sample"
EARLIER = "it was made by an earlier version: draw again"
COUNT_FLOORS = {"m": 1, "n_groups": 1, "n_units": 1, "quota": 1, "seed": 0}


def draw_record(draw: SampleDraw) -> dict[str, Any]:
    """The draw block of a draw file: every field of ``DRAW_FIELDS`` as a JSON value.

    Args:
        draw: The draw.

    Returns:
        The fields in the order of ``DRAW_FIELDS``, the ids and groups as lists of strings.
    """
    return {name: json_ready(getattr(draw, name)) for name in DRAW_FIELDS}


def write_draw(draw: SampleDraw, path: str | Path) -> None:
    """Write a draw to a JSON file that ``read_draw`` reads back.

    The file holds the draw block under ``draw``, as the description of ``corrections sample``
    does, with the library version beside it.

    Args:
        draw: The draw.
        path: The file to write, a ``str`` or a ``Path``; it is replaced atomically.
    """
    target = Path(path)
    write_json(
        target, {"analysis": ANALYSIS, DRAW_KEY: draw_record(draw), "library_version": __version__}
    )
    logger.info("wrote the draw of %d texts to %s", draw.ids.size, target)


def read_draw(path: str | Path) -> SampleDraw:
    """Read the draw of a draw file, as ``corrections sample`` and ``write_draw`` write it.

    Args:
        path: The draw file, a ``str`` or a ``Path``, such as the ``<out>.json`` of
            ``corrections sample``.

    Returns:
        The draw, equal in every field to the draw that was written.

    Raises:
        ConfigurationError: Naming the file, if it is not a JSON object with a draw block, the
            block lacks a field of ``DRAW_FIELDS`` (as a file of an earlier version without
            the drawn ids, the groups or ``population_hash``: draw again) or holds another,
            ``population_hash`` is not 64 lowercase hexadecimal characters (draw again), a
            field has the wrong type or range, or ``m``, ``n_groups`` and ``pi`` contradict
            the drawn groups, as ``attach_draw`` refuses them.
    """
    source = Path(path)
    content = read_sidecar(source)
    if DRAW_KEY not in content or not isinstance(content[DRAW_KEY], Mapping):
        raise ConfigurationError(
            f"{source} holds no draw: write it with corrections sample or write_draw"
        )
    block = content[DRAW_KEY]
    missing = [name for name in DRAW_FIELDS if name not in block]
    if missing:
        raise ConfigurationError(f"{source}: the draw lacks {list_first_ten(missing)}; {EARLIER}")
    unknown = sorted(str(name) for name in block if name not in DRAW_FIELDS)
    if unknown:
        raise ConfigurationError(
            f"{source}: the draw holds fields that no version writes: {list_first_ten(unknown)}"
        )
    fingerprint = block["population_hash"]
    if not isinstance(fingerprint, str) or POPULATION_HASH_PATTERN.fullmatch(fingerprint) is None:
        raise ConfigurationError(
            f"{source}: the draw's population_hash is not 64 lowercase hexadecimal characters "
            f"(got {fingerprint!r}); {EARLIER}"
        )
    draw = SampleDraw(
        ids=_labels(block, "ids", source),
        groups=_labels(block, "groups", source),
        pi=_probability(block, source),
        m=_count(block, "m", source),
        n_groups=_count(block, "n_groups", source),
        n_units=_count(block, "n_units", source),
        quota=_count(block, "quota", source),
        seed=_count(block, "seed", source),
        population_hash=fingerprint,
    )
    problem = design_problem(draw, draw.n_groups, "its own n_groups")
    if problem:
        raise ConfigurationError(f"{source}: the draw's fields contradict each other: {problem}")
    logger.info("read the draw of %d texts in %d groups from %s", draw.ids.size, draw.m, source)
    return draw


def _labels(block: Mapping[str, Any], name: str, source: Path) -> np.ndarray:
    """The drawn ids or groups: a non-empty list of strings or integers, as strings."""
    values = block[name]
    if not isinstance(values, list) or not values:
        raise ConfigurationError(
            f"{source}: the draw's {name} must be a non-empty list of strings; got {values!r}"
        )
    return as_unicode_1d(values, f"{source}: the draw's {name}")


def _count(block: Mapping[str, Any], name: str, source: Path) -> int:
    """A count of the draw block: an integer, not a boolean, of at least its floor."""
    value, floor = block[name], COUNT_FLOORS[name]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < floor:
        raise ConfigurationError(
            f"{source}: the draw's {name} must be an integer of at least {floor}; got {value!r}"
        )
    return int(value)


def _probability(block: Mapping[str, Any], source: Path) -> float:
    """The draw's ``pi``: a number in ``(0, 1]``."""
    value = block["pi"]
    refusal = f"{source}: the draw's pi must be a number in (0, 1]; got {value!r}"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigurationError(refusal)
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise ConfigurationError(refusal)
    return number
