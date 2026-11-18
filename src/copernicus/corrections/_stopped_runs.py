"""What a battery or a study that stopped with an error does with the report of what it counted.

The run logs the summaries of the cases that its finished calls met (``_open_cases``) on the way
out. ``report_stopped`` makes that report without letting a failure of the report take the place
of the error that stopped the run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ["report_stopped"]


def _described(failure: Exception) -> str:
    """A failure's ``repr``, or its type's name when the ``repr`` fails with an ``Exception``."""
    try:
        return repr(failure)
    except Exception:
        return f"{type(failure).__name__}, whose description could not be formed"


def _noted(error: BaseException, note: str) -> bool:
    """Add a note to an error; False without notes or when adding fails with an ``Exception``."""
    add_note = getattr(error, "add_note", None)
    if add_note is None:
        return False
    try:
        add_note(note)
    except Exception:
        return False
    return True


def report_stopped(error: BaseException, report: Callable[[], None]) -> None:
    """Log what a run counted before ``error`` stopped it; no ``Exception`` takes its place.

    A report that fails with an ``Exception`` must not take the place of the error that stopped
    the run. The failure is attached to that error instead: as a note where exceptions have
    notes (Python 3.11 and later), and otherwise, or when adding the note fails with an
    ``Exception``, as the error's context when it has none. A failure whose ``repr`` fails with
    an ``Exception`` is named by its type, and a report that raises ``error`` itself leaves it
    as it is. A ``KeyboardInterrupt`` or a ``SystemExit`` raised during the report, or while
    its failure is described or noted, is not held back: it propagates with Python's normal
    exception chaining.

    Args:
        error: The error that stopped the run, which the caller raises again.
        report: Logs the summaries of the run, saying that it stopped.
    """
    try:
        report()
    except Exception as failure:
        if failure is error:
            return
        note = f"the warnings counted before this error could not be logged: {_described(failure)}"
        if not _noted(error, note) and error.__context__ is None:
            failure.__context__ = None
            error.__context__ = failure
