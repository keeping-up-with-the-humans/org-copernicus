"""Stub corrections and data of the battery tests: failing, altered and degenerate fits."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest
from corrections_support import BATTERY_SCORES, battery_frame

from copernicus.corrections import CorrectionData, get_correction
from copernicus.corrections import battery as battery_module
from copernicus.corrections.families import Fitted, ScoreCorrection
from copernicus.errors import EstimatorError

LINEAR = ("regression_calibration", "linear")
RIDGE = ("supervised_stacking", "ridge")
SMALL = {"sizes": (100,), "schemes": ("kfold5",), "seeds": (42,)}


@dataclass
class Delegating:
    """A correction that defers its names and ``applies`` to a real one."""

    inner: ScoreCorrection

    @property
    def family(self) -> str:
        return self.inner.family

    @property
    def variant(self) -> str:
        return self.inner.variant

    @property
    def target(self) -> Any:
        return self.inner.target

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        return self.inner.applies(data)


@dataclass
class FailingOnFold(Delegating):
    """A real correction that raises ``error`` on its ``failing_call``-th fit."""

    error: Exception = field(default_factory=lambda: EstimatorError("planted"))
    failing_call: int = 1
    calls: list[int] = field(default_factory=list)

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> Fitted:
        self.calls.append(len(self.calls))
        if len(self.calls) == self.failing_call:
            raise self.error
        return self.inner.fit(data, train_ids, rng=rng)


@dataclass(frozen=True)
class AlteredFitted:
    """A fitted correction whose predictions pass through a change."""

    inner: Fitted
    change: Callable[[pd.Series], pd.Series]

    @property
    def provenance(self) -> Mapping[str, object]:
        return self.inner.provenance

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        return self.change(self.inner.predict(data, ids))


@dataclass
class AlteredPredictions(Delegating):
    """A real correction whose fitted predictions pass through a change."""

    change: Callable[[pd.Series], pd.Series] = field(default=lambda predicted: predicted)

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> Fitted:
        return AlteredFitted(self.inner.fit(data, train_ids, rng=rng), self.change)


def patch_linear(
    monkeypatch: pytest.MonkeyPatch, wrap: Callable[[ScoreCorrection], ScoreCorrection]
) -> None:
    """Make the battery build its linear calibration through ``wrap``."""

    def build(family: str, variant: str, **params: object) -> ScoreCorrection:
        correction = get_correction(family, variant, **params)
        return wrap(correction) if (family, variant) == LINEAR else correction

    monkeypatch.setattr(battery_module, "get_correction", build)


def failing_linear(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Make the battery's linear calibration raise ``error`` on its third fit."""
    patch_linear(monkeypatch, lambda inner: FailingOnFold(inner, error=error, failing_call=3))


def altered_linear(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[pd.Series], pd.Series]
) -> None:
    """Make the predictions of the battery's linear calibration pass through ``change``."""
    patch_linear(monkeypatch, lambda inner: AlteredPredictions(inner, change=change))


@dataclass(frozen=True)
class HeldOutConstantFitted:
    """A fitted correction that predicts its training texts and a constant everywhere else."""

    inner: Fitted
    train_ids: tuple[str, ...]

    @property
    def provenance(self) -> Mapping[str, object]:
        return self.inner.provenance

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        predicted = self.inner.predict(data, ids)
        if tuple(predicted.index) == self.train_ids:
            return predicted
        return pd.Series(0.5, index=predicted.index)


@dataclass
class HeldOutConstant(FailingOnFold):
    """A real correction whose ``failing_call``-th fit predicts a constant on held-out texts."""

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> Fitted:
        self.calls.append(len(self.calls))
        fitted = self.inner.fit(data, train_ids, rng=rng)
        if len(self.calls) == self.failing_call:
            return HeldOutConstantFitted(fitted, tuple(np.asarray(train_ids).astype(str)))
        return fitted


def binary_data(*, share: float = 0.5, frame: pd.DataFrame | None = None) -> CorrectionData:
    """The battery world with gold cut to 0/1, the top ``share`` of the latent value being 1."""
    world = battery_frame() if frame is None else frame.copy()
    world["gold"] = (world["gold"] > np.quantile(world["gold"], 1.0 - share)).astype(float)
    return CorrectionData(frame=world, score_columns=BATTERY_SCORES)
