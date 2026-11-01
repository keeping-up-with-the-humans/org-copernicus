"""Parsing of option values and the error handling shared by all commands."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import typer

from copernicus.errors import CopernicusError

logger = logging.getLogger(__name__)

__all__ = [
    "cli_errors",
    "parse_backend_options",
    "parse_id_list",
    "parse_range",
    "parse_temperature",
]

AUTO: Literal["auto"] = "auto"
ERROR_EXIT_CODE = 1
BACKEND_OPTION = "--backend-option"


@contextmanager
def cli_errors() -> Iterator[None]:
    """Turn library errors raised in a command body into a message and exit code 1.

    Only :class:`~copernicus.errors.CopernicusError` is handled; the settings objects
    raise :class:`~copernicus.errors.SettingsError`, which is one. Any other exception is
    a bug and keeps its traceback. The message is printed to standard error as
    ``error: <message>``.

    Raises:
        typer.Exit: With code 1, after printing the message.
    """
    try:
        yield
    except CopernicusError as error:
        logger.info("command failed with %s", type(error).__name__)
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=ERROR_EXIT_CODE) from error


def parse_backend_options(values: Sequence[str]) -> dict[str, Any]:
    """Parse repeated ``KEY=JSON`` values into keyword arguments of a backend.

    The messages never repeat a value, so an option cannot leak a secret into the
    terminal output.

    Args:
        values: Each ``KEY=JSON``, with ``KEY`` a Python name and the rest a JSON value.

    Returns:
        The decoded values by key.

    Raises:
        typer.BadParameter: If a value has no ``=``, its key is not a Python name or is
            given twice, or the text after ``=`` is not JSON.
    """
    options: dict[str, Any] = {}
    for value in values:
        key, separator, encoded = value.partition("=")
        if not separator:
            raise typer.BadParameter(
                "expected KEY=JSON; a value has no '='", param_hint=BACKEND_OPTION
            )
        if not key.isidentifier():
            raise typer.BadParameter(
                f"expected KEY=JSON with KEY a Python name, got the key {key!r}",
                param_hint=BACKEND_OPTION,
            )
        if key in options:
            raise typer.BadParameter(f"{key!r} is given twice", param_hint=BACKEND_OPTION)
        try:
            options[key] = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise typer.BadParameter(
                f"the value of {key!r} is not JSON ({error.msg}); write text in double "
                f'quotes, for example {key}="text"',
                param_hint=BACKEND_OPTION,
            ) from error
    return options


def parse_range(value: str) -> tuple[int, int]:
    """Parse ``A:B`` into the half-open text range ``(A, B)``.

    Args:
        value: Two non-negative integers separated by a colon, with ``A <= B``.

    Returns:
        The pair ``(A, B)``.

    Raises:
        typer.BadParameter: If the value is not of that form.
    """
    start_text, separator, end_text = value.partition(":")
    if not separator or not start_text.isdecimal() or not end_text.isdecimal():
        raise typer.BadParameter(
            f"expected A:B with two non-negative integers, got {value!r}", param_hint="--range"
        )
    start, end = int(start_text), int(end_text)
    if start > end:
        raise typer.BadParameter(
            f"the start must not exceed the end, got {value!r}", param_hint="--range"
        )
    return start, end


def parse_temperature(value: str) -> float | Literal["auto"]:
    """Parse ``auto`` or a number into a temperature setting.

    Args:
        value: ``"auto"`` or a decimal number; the range is checked by the settings.

    Returns:
        ``"auto"`` or the number.

    Raises:
        typer.BadParameter: If the value is neither.
    """
    if value == AUTO:
        return AUTO
    try:
        return float(value)
    except ValueError as error:
        raise typer.BadParameter(
            f"expected '{AUTO}' or a number, got {value!r}", param_hint="--temperature"
        ) from error


def parse_id_list(value: str | None) -> tuple[str, ...]:
    """Split a comma-separated list of ids.

    Args:
        value: Ids separated by commas, or ``None``.

    Returns:
        The ids with surrounding spaces removed; empty for ``None`` or an empty string.
    """
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())
