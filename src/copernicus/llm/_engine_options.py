"""Checks of the vLLM engine options: they are recorded as JSON, and the window is an integer."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Any

from copernicus.errors import SettingsError

__all__ = ["context_window", "engine_options_json"]

WINDOW_OPTION = "max_model_len"


def engine_options_json(options: Mapping[str, Any]) -> str:
    """Serialise the engine options as canonical JSON: sorted keys at every level, no spaces.

    Args:
        options: The engine options as the caller gave them.

    Returns:
        The JSON text of the options; ``{}`` for none.

    Raises:
        SettingsError: If an option name, or a key of a mapping nested in an option, is not a
            string, or an option is not a JSON value. The message names the option and the
            type of its value (``float nan`` for a number that is not finite); for a list or
            dict that holds a bad value it says so and names the option.
    """
    for name in options:
        if not isinstance(name, str):
            raise SettingsError(
                f"engine option names must be strings; got {name!r} ({type(name).__name__})"
            )
    for option, value in options.items():
        for key in _non_string_keys(value, frozenset()):
            raise SettingsError(
                f"engine option {option!r} holds a mapping whose key {key!r} "
                f"({type(key).__name__}) is not a string; names must be strings at every level"
            )
    try:
        return json.dumps(dict(options), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        for key, value in options.items():
            try:
                json.dumps(value, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError):
                raise SettingsError(_non_json_message(key, value)) from error
        raise SettingsError(
            f"the engine options cannot be recorded in the run provenance as JSON: {error}"
        ) from error


def context_window(value: object) -> int:
    """Return the context window of the engine options: an integer of at least 1.

    Args:
        value: The ``max_model_len`` of the engine options after the merge.

    Returns:
        The window in tokens.

    Raises:
        SettingsError: If the value is not an ``int`` (a ``bool`` is not) of at least 1.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SettingsError(
            f"engine option {WINDOW_OPTION!r} must be an integer of at least 1, "
            f"got {value!r} ({type(value).__name__})"
        )
    return value


def _non_string_keys(value: object, path: frozenset[int]) -> Iterator[object]:
    """Yield the keys that are not strings of the mappings in a value, at every level.

    Args:
        value: An engine option, or a part of one.
        path: The ids of the containers being walked, so that a container holding itself ends
            the walk (the JSON dump then refuses it).
    """
    if not isinstance(value, (dict, list, tuple)) or id(value) in path:
        return
    inside = path | {id(value)}
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                yield key
            yield from _non_string_keys(item, inside)
    else:
        for item in value:
            yield from _non_string_keys(item, inside)


def _non_json_message(key: str, value: Any) -> str:
    """Describe an engine option whose value is not JSON, for the message of the refusal."""
    if isinstance(value, (dict, list, tuple)):
        problem = f"holds a value that is not JSON ({type(value).__name__})"
    elif isinstance(value, float) and not math.isfinite(value):
        problem = f"is not a JSON value (float {value})"
    else:
        problem = f"is not a JSON value ({type(value).__name__})"
    return (
        f"engine option {key!r} {problem}; engine options are recorded in the run provenance "
        "as JSON, so pass strings, numbers, booleans, None, and lists or dicts of them"
    )
