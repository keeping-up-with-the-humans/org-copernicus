"""The rules that every example script and its support module obey.

A script imports only public names from the library and stays under 400 lines, so that a reader
can take it in at once and copy it without reaching into the internals.
"""

from __future__ import annotations

import ast
from pathlib import Path

__all__ = ["MAX_LINES", "check_script_rules"]

MAX_LINES = 400


def check_script_rules(folder: Path, names: tuple[str, ...]) -> None:
    """Assert that each named file of the folder is short and imports no private library name.

    Args:
        folder: The folder that holds the files.
        names: The file names to check.

    Raises:
        AssertionError: If a file has 400 lines or more, or a ``from copernicus... import``
            statement names a module or an object that starts with an underscore.
    """
    for name in names:
        path = folder / name
        assert len(path.read_text(encoding="utf-8").splitlines()) < MAX_LINES, name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("copernicus"):
                parts = (node.module or "").split(".") + [alias.name for alias in node.names]
                assert not [part for part in parts if part.startswith("_")], (name, node.module)
