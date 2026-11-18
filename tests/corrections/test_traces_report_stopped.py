"""What the report of a run that stopped does when the report itself fails.

``report_stopped`` logs what a battery or a study counted before an error stopped it. A report
that fails with an ``Exception`` never takes the place of that error, also when the failure
cannot be described, when the error takes no note, or when the report raises the error again.
An interrupt or an exit raised during the report is not held back.
"""

from __future__ import annotations

import pytest
from corrections_support import battery_data, battery_frame

from copernicus.corrections import BatterySettings, battery, battery_folds, run_battery
from copernicus.corrections._stopped_runs import report_stopped

pytestmark = pytest.mark.filterwarnings("error")

SETTINGS = BatterySettings(
    families=[("multiple_imputation", "model_draws")],
    sizes=(3,),
    schemes=("kfold2",),
    seeds=(1,),
)


class _UndescribedError(Exception):
    """A failure whose description cannot be formed."""

    def __repr__(self) -> str:
        raise RuntimeError("no description")


class _NoNoteError(RuntimeError):
    """An error that refuses a note."""

    def add_note(self, note: str) -> None:
        raise TypeError("no notes here")


def _raises(failure: BaseException) -> object:
    def report() -> None:
        raise failure

    return report


def _notes(error: BaseException) -> str:
    return "".join(getattr(error, "__notes__", []))


def test_a_report_that_raises_the_stopping_error_again_leaves_it_as_it_is() -> None:
    earlier = ValueError("earlier")
    stopped = RuntimeError("stopped")
    stopped.__context__ = earlier
    report_stopped(stopped, _raises(stopped))  # type: ignore[arg-type]
    assert stopped.__context__ is earlier
    assert _notes(stopped) == ""


def test_a_failure_without_a_description_is_attached_by_the_name_of_its_type() -> None:
    stopped = RuntimeError("stopped")
    report_stopped(stopped, _raises(_UndescribedError()))  # type: ignore[arg-type]
    attached = _notes(stopped) or type(stopped.__context__).__name__
    assert "_UndescribedError" in attached


def test_an_error_that_takes_no_note_gets_the_failure_as_its_context() -> None:
    failure = ValueError("the summary failed")
    try:
        raise _NoNoteError("stopped")
    except _NoNoteError as stopped:
        # Raised while the stopping error is handled, the failure gets that error as its
        # context from Python; the link back is cut so that the two do not form a ring.
        report_stopped(stopped, _raises(failure))  # type: ignore[arg-type]
        assert stopped.__context__ is failure
        assert failure.__context__ is None


def test_an_error_that_takes_no_note_keeps_a_context_it_already_has() -> None:
    earlier = ValueError("earlier")
    stopped = _NoNoteError("stopped")
    stopped.__context__ = earlier
    report_stopped(stopped, _raises(ValueError("the summary failed")))  # type: ignore[arg-type]
    assert stopped.__context__ is earlier


@pytest.mark.parametrize("leaving", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_or_an_exit_during_the_report_is_not_held_back(
    leaving: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stopped after the folds")

    def leave(counts: object) -> str:
        raise leaving("during the report")

    monkeypatch.setattr(battery, "_add_diagnostics", stop)
    monkeypatch.setattr(battery_folds, "summary", leave)
    data = battery_data(battery_frame(n_groups=80, group_size=1))
    with pytest.raises(leaving, match="during the report") as left:
        run_battery(data, SETTINGS)
    context = left.value.__context__
    assert isinstance(context, RuntimeError) and "after the folds" in str(context)
