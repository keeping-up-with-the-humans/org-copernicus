"""Package versions, read with the parser of the ``packaging`` library.

A version is read whole, as the package index reads it, without importing the package it
describes: an epoch (``1!5.12.1``), a leading ``v``, surrounding spaces and the labels of a
pre-release, post-release, development release or local build are parts of a version, and a
string with anything else, such as ``4.57.3oops`` or ``4..5``, is not a version.
"""

from __future__ import annotations

import logging
import re

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

__all__ = ["is_below", "is_release_number", "major_line", "read_version", "release_text"]

RELEASE_NUMBER = re.compile(r"\d+(?:\.\d+)*")


def is_release_number(value: object) -> bool:
    """Tell whether a value is a release number: decimal integers joined by dots.

    Args:
        value: The value as given; non-strings are not release numbers.

    Returns:
        True for a string such as ``"5"`` or ``"4.56"``; false for one with a space, a letter,
        a suffix such as ``rc1``, or an empty part between two dots.
    """
    return isinstance(value, str) and RELEASE_NUMBER.fullmatch(value) is not None


def read_version(text: object) -> Version | None:
    """Read a version as the ``packaging`` library reads a package version.

    Args:
        text: The version as a package reports it, such as ``transformers.__version__``, or
            as a sidecar records it.

    Returns:
        The version, or ``None`` for a value that is not a string or that the parser cannot
        read; the caller refuses such a value.
    """
    if not isinstance(text, str):
        return None
    try:
        return Version(text)
    except InvalidVersion:
        return None


def major_line(version: Version) -> tuple[int, int]:
    """Return the line of a version: its epoch and the first of its release numbers.

    An epoch starts a new numbering, so versions of two epochs are on two lines even when their
    release numbers agree. A pre-release, post-release, development release or local build is
    on the line of its release numbers: ``5.0.0rc1`` is on the line of ``5.12.1``.

    Args:
        version: A version that `read_version` returned.

    Returns:
        ``(epoch, major)``, such as ``(0, 5)`` for ``5.12.1`` and ``(1, 4)`` for ``1!4.57.6``.
    """
    return version.epoch, version.release[0]


def is_below(version: Version, bound: Version) -> bool:
    """Tell whether a version comes before a bound: by its epoch, then by its release numbers.

    The release numbers are compared with the shorter padded with zeros, and the labels after
    them are left out, so a pre-release or development release of the bound's release counts
    as that release: ``5.0.0rc1`` is not below ``5``. A version of a later epoch is never below
    a bound of an earlier one, as in the ordering of the package index.

    Args:
        version: The version, such as the installed transformers.
        bound: The bound, such as the ``transformers_below`` of an encoder.

    Returns:
        True when ``version`` comes before ``bound``: ``4.57.6`` before ``5``, while ``5``,
        ``5.0.0rc1`` and ``1!4.57.6`` do not.
    """
    width = max(len(version.release), len(bound.release))
    numbers = version.release + (0,) * (width - len(version.release))
    limit = bound.release + (0,) * (width - len(bound.release))
    return (version.epoch, numbers) < (bound.epoch, limit)


def release_text(numbers: tuple[int, ...]) -> str:
    """Return release numbers as a version string, such as ``"4.56"`` for ``(4, 56)``."""
    return ".".join(str(number) for number in numbers)
