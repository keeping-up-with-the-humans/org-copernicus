"""Settings and summary of the coding runner."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from copernicus.coders.one_stage import OneStageConfig
from copernicus.coders.store_types import METHODS
from copernicus.coders.two_stage_types import TwoStageConfig
from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = ["CodingSettings", "CodingSummary", "resolve_temperature"]

AUTO = "auto"
AUTO_TEMPERATURE_SINGLE_RUN = 0.0
AUTO_TEMPERATURE_REPEATED_RUNS = 0.3
MAX_TEMPERATURE = 2.0


@dataclass(frozen=True)
class CodingSettings:
    """Settings of one coding run.

    ``runs``, ``text_range``, ``concurrency``, ``save_every``, ``max_failure_share``, and
    ``retry_failed`` do not enter the identity of the store, so a later invocation may
    extend the runs or code another shard into the same store. ``"auto"`` resolves to a
    temperature that depends on ``runs``, so a store is extended to more runs only under an
    explicit temperature.

    Attributes:
        method: ``"two_stage"`` or ``"one_stage"``.
        runs: Number of repeated codings of each text; at least 1.
        temperature: Sampling temperature in ``[0, 2]``, or ``"auto"`` for 0.0 with one
            run and 0.3 with several.
        seed: Seed of the wording rotation and the sampling.
        wording_enable: ``"all"`` or the wording variant ids to pin.
        wording_disable: Wording variant ids to leave out.
        shuffle_markers: Whether the marker order of each attribute is shuffled.
        concurrency: Number of attribute evaluations in flight; at least 1. The
            attribute evaluations of one batch (``save_every`` texts times the missing
            runs times the attributes) are submitted together and the batch drains
            before the next one starts, so at most that many jobs are in flight and the
            pool idles at the end of every batch. The OpenAI-compatible backend answers
            concurrent calls in parallel and gains throughput from a higher value. The
            vLLM backend runs one call at a time behind a lock and the runner does not
            batch calls for it, so with vLLM keep 1: more threads only wait.
        save_every: Number of texts coded between two writes to the store; at least 1.
        max_failure_share: Largest share of attribute results that may end in
            ``extraction_failed`` or ``scoring_failed``. After every batch, once at least 30
            attribute calls and one full batch (``save_every`` texts with all runs) have
            been made, the share among this invocation's calls is checked, so a failing
            backend stops early. At the end of the run, also when it codes nothing, the
            share among the attribute results of the requested scope (the texts of
            ``text_range`` times the runs 1 to ``runs``) is checked, where the results
            this invocation codes and the stored results of the scope it keeps count alike.
        retry_failed: Whether a stored cell of the scope with an attribute result in
            ``extraction_failed`` or ``scoring_failed`` is coded again; its stored row and
            evidence are replaced in the transaction that writes the new row. Otherwise
            such a cell is kept and counted by the failure check at the end of the run.
        text_range: ``(start, end)`` positions ``[start, end)`` of the texts in text-id
            order, or ``None`` for all texts.
        two_stage: Config of the two-stage LLM coder.
        one_stage: Config of the one-stage LLM coder.
    """

    method: Literal["two_stage", "one_stage"] = "two_stage"
    runs: int = 2
    temperature: float | Literal["auto"] = "auto"
    seed: int = 42
    wording_enable: Sequence[str] | Literal["all"] = "all"
    wording_disable: Sequence[str] = ()
    shuffle_markers: bool = True
    concurrency: int = 1
    save_every: int = 10
    max_failure_share: float = 0.10
    retry_failed: bool = False
    text_range: tuple[int, int] | None = None
    two_stage: TwoStageConfig = TwoStageConfig()
    one_stage: OneStageConfig = OneStageConfig()

    def __post_init__(self) -> None:
        """Check the settings.

        Raises:
            SettingsError: If a setting is outside its documented range; the message names
                the field and its value.
        """
        temperature_ok = self.temperature == AUTO or (
            isinstance(self.temperature, (int, float))
            and not isinstance(self.temperature, bool)
            and 0 <= self.temperature <= MAX_TEMPERATURE
        )
        range_ok = self.text_range is None or 0 <= self.text_range[0] <= self.text_range[1]
        checks = (
            ("method", self.method in METHODS, f"one of {METHODS}"),
            ("runs", self.runs >= 1, "at least 1"),
            ("temperature", temperature_ok, f"'{AUTO}' or a number in [0, {MAX_TEMPERATURE}]"),
            ("concurrency", self.concurrency >= 1, "at least 1"),
            ("save_every", self.save_every >= 1, "at least 1"),
            ("max_failure_share", 0 <= self.max_failure_share <= 1, "in [0, 1]"),
            ("text_range", range_ok, "None or (start, end) with 0 <= start <= end"),
        )
        for field, valid, expectation in checks:
            if not valid:
                raise SettingsError(f"{field} must be {expectation}, got {getattr(self, field)!r}")


@dataclass(frozen=True)
class CodingSummary:
    """What one invocation of `code_texts` did, up to its end or its stop.

    The requested scope is the texts of ``text_range`` times the runs 1 to ``runs``. Its
    cells are coded in this invocation, coded again (``retried``), or kept from the store
    (``resumed``); the failure check at the end of a run judges the results of all of them,
    the checks after each batch this invocation's alone.

    Attributes:
        n_texts: Texts coded in this invocation (at least one cell coded or coded again).
        n_runs: Number of runs requested by the settings.
        n_attribute_calls: Attribute codings made in this invocation.
        status_counts: Number of this invocation's attribute results per status, for every
            status.
        stored_status_counts: Number of attribute results per status, for every status, of
            the cells of the scope that the store held and this invocation kept.
        prompt_tokens: Prompt tokens spent in this invocation.
        completion_tokens: Generated tokens spent in this invocation.
        seconds: Wall-clock time of this invocation.
        store_path: Location of the run store.
        resumed: Cells (text and run) of the scope kept because the store already held
            them, cells with a failed attribute included unless ``retry_failed`` is set.
        retried: Stored cells with a failed attribute that this invocation coded again
            (``retry_failed``); they are not counted in ``resumed``.
    """

    n_texts: int
    n_runs: int
    n_attribute_calls: int
    status_counts: Mapping[str, int]
    stored_status_counts: Mapping[str, int]
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    store_path: Path
    resumed: int
    retried: int


def resolve_temperature(setting: float | Literal["auto"], runs: int) -> float:
    """Turn the temperature setting into a number.

    Args:
        setting: ``"auto"`` or a temperature.
        runs: Number of runs of the coding.

    Returns:
        0.0 for ``"auto"`` with one run, 0.3 for ``"auto"`` with several runs, and the
        given temperature otherwise.
    """
    if setting == AUTO:
        return AUTO_TEMPERATURE_SINGLE_RUN if runs == 1 else AUTO_TEMPERATURE_REPEATED_RUNS
    return float(setting)
