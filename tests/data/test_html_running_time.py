"""How the processor time of strip_html grows with the length of four kinds of long line.

Each test compares the processor time of a line of 200,000 characters with that of a line of
20,000. These tests can detect a dominant quadratic part at the measured sizes; they do not
establish linear time for every input.
"""

from __future__ import annotations

import signal
import time
from types import FrameType

import pytest

from copernicus.data import strip_html

SHORT_LENGTH = 20_000
GROWTH = 10
# A rule whose work grows linearly takes about ten times as long on a line ten times as long, a
# rule whose work grows with the square of the length about a hundred times. The bound lies
# between the two, a factor of three from each.
BOUND = 30
SHORT_REPEATS = 5
# A short line that takes longer than this has failed already: the limit stops a rule that is slow
# at 20,000 characters, as the limit of the long attempts stops one that is slow at 200,000.
SHORT_LIMIT_SECONDS = 2.0
LONG_ATTEMPTS = 3
LONG_LINE_KINDS = ("less_than_signs", "comment_openers", "nested_svg", "comment_openers_in_svg")
NO_TIMER_REASON = "a timer of processor time needs signal.setitimer, which this platform lacks"


class ProcessorTimeLimitError(Exception):
    """The cleaning of a line ran past its limit of processor time."""


def long_line(kind: str, length: int) -> tuple[str, str]:
    """Return a line of about ``length`` characters of one kind, and the text it cleans to.

    On each kind a rule that reads ahead from every ``<`` or every comment opener again, or that
    passes over the whole text once per level of nesting, does work that grows with the square of
    the length.
    """
    openers = "<!--" * (length // 4)
    depth = length // 11
    lines = {
        "less_than_signs": ("<" * length, "<" * length),
        "comment_openers": (openers, openers),
        "nested_svg": ("<svg>" * depth + "x" + "</svg>" * depth, "x"),
        "comment_openers_in_svg": ("<svg>" + openers + "</svg>", openers),
    }
    return lines[kind]


def processor_seconds(text: str, limit_seconds: float) -> float:
    """Return the processor time that one cleaning of the text takes, up to ``limit_seconds``.

    Raises:
        ProcessorTimeLimitError: If the cleaning takes more than ``limit_seconds`` of processor
            time.
    """
    started = time.process_time()
    cleaned_within(text, limit_seconds)
    return time.process_time() - started


def cleaned_within(text: str, limit_seconds: float) -> str:
    """Return the cleaned text, stopping the cleaning once it has used ``limit_seconds``.

    A timer of processor time interrupts a cleaning that runs past the limit, so that a rule
    whose work grows with the square of the length fails the test at the limit instead of running
    for minutes.

    Raises:
        ProcessorTimeLimitError: If the cleaning takes more than ``limit_seconds`` of processor
            time.
    """

    def interrupt(signal_number: int, frame: FrameType | None) -> None:
        """Stop the cleaning: its processor time has reached the limit."""
        raise ProcessorTimeLimitError(f"the cleaning took more than {limit_seconds:.3f} s")

    previous_handler = signal.signal(signal.SIGPROF, interrupt)
    signal.setitimer(signal.ITIMER_PROF, limit_seconds)
    try:
        return strip_html(text)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous_handler)


@pytest.mark.slow
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason=NO_TIMER_REASON)
@pytest.mark.parametrize("kind", LONG_LINE_KINDS)
def test_ten_times_the_length_takes_at_most_thirty_times_the_processor_time(kind: str) -> None:
    """Ten times the length of a line takes at most thirty times its processor time.

    The test bounds the ratio of two processor times of one process, so the speed of the machine
    cancels out. On these four kinds of line a rule whose time grows with the square of the
    length fails it; a smaller quadratic part, or a slow input of another kind, can pass. Processor
    time leaves out the time the process waits while others run, which reduces the effect of a
    loaded machine without removing it: a change of the processor's clock, contention for memory
    and caches, or other threads of the process can still slow a run. The least of five short
    runs and three attempts at the long line reduce that risk further. Every run has a limit of
    processor time, so a rule that is already slow on the short line fails instead of running on.
    """
    short_text, short_cleaned = long_line(kind, SHORT_LENGTH)
    long_text, long_cleaned = long_line(kind, SHORT_LENGTH * GROWTH)
    try:
        assert cleaned_within(short_text, SHORT_LIMIT_SECONDS) == short_cleaned
        short_seconds = min(
            processor_seconds(short_text, SHORT_LIMIT_SECONDS) for _ in range(SHORT_REPEATS)
        )
    except ProcessorTimeLimitError:
        pytest.fail(
            f"{kind}: {len(short_text)} characters took more than {SHORT_LIMIT_SECONDS} s of "
            "processor time"
        )
    for _ in range(LONG_ATTEMPTS):
        try:
            cleaned = cleaned_within(long_text, BOUND * short_seconds)
        except ProcessorTimeLimitError:
            continue
        assert cleaned == long_cleaned
        return
    pytest.fail(
        f"{kind}: {len(long_text)} characters took more than {BOUND} times the "
        f"{short_seconds:.4f} s of {len(short_text)} characters in each of {LONG_ATTEMPTS} attempts"
    )
