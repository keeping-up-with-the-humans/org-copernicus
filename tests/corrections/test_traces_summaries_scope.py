"""Where the counting of the cases' warnings starts and ends.

Only ``run_battery`` and ``repeated_study`` count the warnings of the cases in place of logging
them, because only they log the summaries. A fold run on its own logs the warnings of its fits;
a battery or a study that stops with an error still logs the cases of the calls it finished;
and a context kept from inside a run logs a later warning instead of counting it for a run that
is over.
"""

from __future__ import annotations

import logging
from collections import Counter
from contextvars import copy_context

import pytest
from corrections_support import battery_data, battery_frame
from traces_support import study_corpus

from copernicus.corrections import (
    BatterySettings,
    _open_cases,
    battery,
    battery_folds,
    get_correction,
    repeated_study,
    run_battery,
    study,
)

pytestmark = pytest.mark.filterwarnings("error")

IMPUTATION = "copernicus.corrections.families.imputation"
BATTERY, STUDY = "copernicus.corrections.battery", "copernicus.corrections.study"
SETTINGS = BatterySettings(
    families=[("multiple_imputation", "model_draws")],
    sizes=(3,),
    schemes=("kfold2",),
    seeds=(1,),
)
# Three training texts for three coefficients: every fit has resamples of numerical rank below 3
# and fits with no more rows than coefficients, so each of its two folds meets both cases.
OPENING = (
    "battery, entry multiple_imputation/model_draws, size 3: of the warnings that its fits log "
    "one by one when called on their own, "
)
BOTH = (
    "{n} that a fit has bootstrap resamples whose design has a numerical rank below 3 "
    "(n_rank_deficient_resamples), and {n} that a fit has bootstrap fits with no more rows than "
    "coefficients (n_resamples_without_residual_df)"
)
STOPPED = "; the battery stopped with an error after these fits and returns no table"


def _warnings(caplog: pytest.LogCaptureFixture, name: str) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == name and r.levelno == logging.WARNING
    ]


def _single_texts() -> object:
    return battery_data(battery_frame(n_groups=80, group_size=1))


def test_a_fold_run_on_its_own_logs_the_warnings_of_its_fits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = _single_texts()
    corrections = {
        entry: get_correction(entry.family, entry.variant, **entry.kwargs)
        for entry in SETTINGS.per_text_entries
    }
    tables = battery_folds.Tables(SETTINGS)
    point = battery_folds.Point(1, "kfold2", 3, (0, 0, 0))
    split = battery.partition(data, "kfold2", 1)[0]  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        battery_folds.run_fold(data, corrections, tables, point, 0, split)  # type: ignore[arg-type]
    logged = _warnings(caplog, IMPUTATION)
    assert len(logged) == 2 and tables.cases == {}
    assert "bootstrap resamples of the 3 training rows" in logged[0]
    assert "bootstrap fits have no more rows than coefficients" in logged[1]


def test_a_battery_that_stops_with_an_error_logs_the_cases_of_its_finished_fits(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _single_texts()
    with caplog.at_level(logging.WARNING):
        run_battery(data, SETTINGS)  # type: ignore[arg-type]
    finished = [text for text in _warnings(caplog, BATTERY) if text.startswith("battery, entry")]
    assert finished == [
        OPENING + BOTH.format(n="2 say") + "; the provenance of each scored fold holds the "
        "records of its fit"
    ]
    caplog.clear()

    def stop(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stopped after the folds")

    monkeypatch.setattr(battery, "_add_diagnostics", stop)
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="after the folds"):
        run_battery(data, SETTINGS)  # type: ignore[arg-type]
    assert _warnings(caplog, IMPUTATION) == []
    assert _warnings(caplog, BATTERY) == [OPENING + BOTH.format(n="2 say") + STOPPED]


def test_a_fit_that_raises_inside_the_battery_keeps_the_cases_it_met(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = battery_folds._fit_entry

    def fit_then_stop(*args: object, **kwargs: object) -> None:
        fit(*args, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("stopped after the first fit")

    monkeypatch.setattr(battery_folds, "_fit_entry", fit_then_stop)
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="the first fit"):
        run_battery(_single_texts(), SETTINGS)  # type: ignore[arg-type]
    assert _warnings(caplog, IMPUTATION) == []
    assert _warnings(caplog, BATTERY) == [OPENING + BOTH.format(n="1 says") + STOPPED]


def test_a_study_that_stops_with_an_error_logs_the_cases_of_its_finished_draws(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    draw = study.study_draw

    def one_draw_then_stop(*args: object) -> object:
        if args[-1] == 1:
            raise RuntimeError("stopped after one draw")
        return draw(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(study, "study_draw", one_draw_then_stop)
    corpus = study_corpus(covariate_on_score=True)
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="after one draw"):
        repeated_study(corpus, truth_gold="truth", sizes=(20, 40), draws=3)
    assert _warnings(caplog, STUDY) == [
        "repeated study, size 20: of the warnings that its inner calls log one by one when "
        "called on their own, 1 says that a cross-fitted prediction has folds whose design has "
        "a numerical rank below 3 (n_rank_deficient_folds); the study stopped with an error "
        "during this size and returns no table"
    ]


def test_a_context_kept_from_inside_a_run_logs_a_later_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("copernicus.corrections.families.latent")
    with _open_cases.gathered() as counts:
        _open_cases.warn_not_converged(log, "inside", 50, 1e-6)
        kept = copy_context()
    with caplog.at_level(logging.WARNING):
        kept.run(_open_cases.warn_not_converged, log, "afterwards", 50, 1e-6)
    assert counts == Counter({"not_converged": 1})
    (message,) = [record.getMessage() for record in caplog.records]
    assert message.startswith("afterwards: the principal-axis iteration reached its cap of 50")


def _report_failure(error: BaseException) -> str:
    """What the error that stopped a run holds of a report that failed after it."""
    notes = "".join(getattr(error, "__notes__", []))
    return notes or repr(error.__context__)


def _broken_summary(counts: object) -> str:
    raise ValueError("the summary failed")


@pytest.mark.parametrize("stopper", [RuntimeError, KeyboardInterrupt])
def test_a_report_that_fails_keeps_the_error_that_stopped_the_battery(
    stopper: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop(*args: object, **kwargs: object) -> None:
        raise stopper("stopped after the folds")

    monkeypatch.setattr(battery, "_add_diagnostics", stop)
    monkeypatch.setattr(battery_folds, "summary", _broken_summary)
    with pytest.raises(stopper, match="after the folds") as stopped:
        run_battery(_single_texts(), SETTINGS)  # type: ignore[arg-type]
    assert "the summary failed" in _report_failure(stopped.value)


@pytest.mark.parametrize("stopper", [RuntimeError, KeyboardInterrupt])
def test_a_report_that_fails_keeps_the_error_that_stopped_the_study(
    stopper: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    draw = study.study_draw

    def one_draw_then_stop(*args: object) -> object:
        if args[-1] == 1:
            raise stopper("stopped after one draw")
        return draw(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(study, "study_draw", one_draw_then_stop)
    monkeypatch.setattr(study, "summary", _broken_summary)
    corpus = study_corpus(covariate_on_score=True)
    with pytest.raises(stopper, match="after one draw") as stopped:
        repeated_study(corpus, truth_gold="truth", sizes=(20,), draws=3)
    assert "the summary failed" in _report_failure(stopped.value)


def test_a_report_that_fails_after_a_finished_run_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(battery_folds, "summary", _broken_summary)
    with pytest.raises(ValueError, match="the summary failed"):
        run_battery(_single_texts(), SETTINGS)  # type: ignore[arg-type]


def test_a_warning_made_while_the_study_logs_its_summary_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("copernicus.corrections.estimates")

    class WarnsOnASummary(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.getMessage().startswith("repeated study, size"):
                _open_cases.warn_zero_se(log, "made while the summary is logged")

    handler = WarnsOnASummary()
    logger = logging.getLogger(STUDY)
    logger.addHandler(handler)
    try:
        with caplog.at_level(logging.WARNING):
            repeated_study(
                study_corpus(covariate_on_score=True), truth_gold="truth", sizes=(20,), draws=2
            )
    finally:
        logger.removeHandler(handler)
    made = [text for text in _warnings(caplog, log.name) if text.startswith("made while")]
    assert len(made) == 1
    (summary_text,) = _warnings(caplog, STUDY)
    assert "2 say that a cross-fitted prediction" in summary_text
    assert "standard error is exactly 0" not in summary_text
