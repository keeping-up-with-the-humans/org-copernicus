"""Log output of the command-line application."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

import typer

logger = logging.getLogger(__name__)

__all__ = ["Verbose", "cli_logging"]

PACKAGE_LOGGER = "copernicus"
LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"

Verbose = Annotated[
    bool,
    typer.Option(
        "--verbose", "-v", help="Show progress and other INFO messages on standard error."
    ),
]


class _EchoHandler(logging.Handler):
    """Writes each record to standard error through typer, like the other CLI messages."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write one formatted record."""
        typer.echo(self.format(record), err=True)


@contextmanager
def cli_logging(verbose: bool) -> Iterator[None]:
    """Show the library's log messages on standard error while a command runs.

    Messages at WARNING and above are shown by default, and INFO and above with
    ``verbose``. The handler and the level of the package logger are restored when
    the command ends, so a process that runs several commands keeps its own setup.

    Args:
        verbose: Whether to show INFO messages, such as the progress of a coding run.

    Yields:
        Nothing; the command runs inside the block.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER)
    handler = _EchoHandler(logging.INFO if verbose else logging.WARNING)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    previous_level = package_logger.level
    package_logger.addHandler(handler)
    if verbose:
        package_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(previous_level)
