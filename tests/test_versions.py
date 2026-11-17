"""The version parser, and the comparisons that decide whether an encoder may be loaded.

`copernicus._versions` imports neither torch nor transformers, so these tests run where neither
is installed, as in the continuous integration. The tests of an encoder's own load, which use
stand-in libraries, need torch and carry the embed marker.
"""

from __future__ import annotations

import pytest
from packaging.version import Version

from copernicus._versions import (
    is_below,
    is_release_number,
    major_line,
    read_version,
    release_text,
)


def read(text: str) -> Version:
    """Read a version that the parser accepts."""
    version = read_version(text)
    assert version is not None, text
    return version


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4.57.6", "4.57.6"),
        ("v4.57.3", "4.57.3"),
        (" 4.57.3 ", "4.57.3"),
        ("4.57.3+review.local", "4.57.3+review.local"),
        ("4.57.3.post1", "4.57.3.post1"),
        ("5.0.0rc1", "5.0.0rc1"),
        ("1!5.12.1", "1!5.12.1"),
    ],
)
def test_a_version_is_read_whole_as_the_package_index_reads_it(text: str, expected: str) -> None:
    assert read_version(text) == Version(expected)


@pytest.mark.parametrize(
    "value",
    ["4.57.3oops", "4..5", "4.garbage", "4.57.3 rc1", "", "unknown", None, 4.57, b"4.57.6"],
)
def test_a_value_that_is_not_a_version_string_reads_as_none(value: object) -> None:
    assert read_version(value) is None


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("5.12.1", (0, 5)),
        ("5.0.0rc1", (0, 5)),
        ("5.12.1+review.local", (0, 5)),
        ("4.57.6", (0, 4)),
        ("0!4.57.6", (0, 4)),
        ("1!4.57.6", (1, 4)),
    ],
)
def test_the_line_of_a_version_is_its_epoch_and_its_first_release_number(
    text: str, line: tuple[int, int]
) -> None:
    assert major_line(read(text)) == line


@pytest.mark.parametrize(
    ("installed", "bound", "below"),
    [
        ("4.56.0", "5", True),
        ("4.99.99", "5", True),
        ("4.57.0.dev0", "5", True),
        ("0!4.57.6", "5", True),
        ("5", "5", False),
        ("5.0.0rc1", "5", False),
        ("5.0.0.dev0", "5", False),
        ("5.12.1+review.local", "5", False),
        ("10.0.0", "5", False),
        ("1!4.57.6", "5", False),
        ("1!5.12.1", "5", False),
        ("4.59.2", "4.60", True),
        ("4.60.0", "4.60", False),
        ("4.60", "4.60.1", True),
        ("2.0", "1!0.1", True),
    ],
)
def test_a_version_is_below_a_bound_by_its_epoch_then_its_release_numbers(
    installed: str, bound: str, below: bool
) -> None:
    assert is_below(read(installed), read(bound)) is below


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5", True),
        ("4.56", True),
        ("4.56.0", True),
        ("5.0rc1", False),
        (" 5", False),
        ("4..5", False),
        ("", False),
        (5, False),
        (None, False),
    ],
)
def test_a_release_number_is_decimal_integers_joined_by_dots(value: object, expected: bool) -> None:
    assert is_release_number(value) is expected


def test_release_numbers_are_written_as_a_version_string() -> None:
    assert release_text((4, 56)) == "4.56"
    assert release_text((5,)) == "5"
