"""Fake ``torch`` and ``transformers`` modules that record how an encoder loads its model."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

GTE = "gte_qwen2_7b"
GTE_REVISION = "a8d08b36ada9cacfe34c4d6f80957772a025daf2"
GTE_REMEDY = 'pip install "transformers>=4.56,<5"'
TRANSFORMERS_4 = "4.57.6"
TRANSFORMERS_5 = "5.12.1"
FAKE_TORCH_VERSION = "2.99.0"


class FakeDtype:
    """Stands in for ``torch.dtype``."""


class FakeDevice:
    """Stands in for ``torch.device``; remembers its name."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeModel:
    """Records the calls a loaded model receives; its config names the loaded commit."""

    def __init__(self, commit: object = "f00dfeed") -> None:
        self.moved_to: list[FakeDevice] = []
        self.evaluating = False
        self.config = SimpleNamespace(_commit_hash=commit)

    def to(self, device: FakeDevice) -> FakeModel:
        self.moved_to.append(device)
        return self

    def eval(self) -> FakeModel:
        self.evaluating = True
        return self


class FakeLibraries:
    """Fake ``torch`` and ``transformers`` that record how the encoder uses them.

    A loader given an error records its call and then raises that error, as a load that fails
    inside transformers or inside a model repository's own code does.
    """

    def __init__(
        self,
        *,
        cuda: bool = False,
        mps: bool = False,
        commit: object = "f00dfeed",
        transformers_version: str = "9.99.0",
        tokenizer_error: BaseException | None = None,
        model_error: BaseException | None = None,
    ) -> None:
        self.tokenizer_error = tokenizer_error
        self.model_error = model_error
        self.bfloat16 = FakeDtype()
        self.torch = SimpleNamespace(
            __version__=FAKE_TORCH_VERSION,
            dtype=FakeDtype,
            bfloat16=self.bfloat16,
            device=FakeDevice,
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        )
        self.tokenizer = SimpleNamespace(padding_side="unset")
        self.model = FakeModel(commit)
        self.tokenizer_calls: list[tuple[str, dict[str, Any]]] = []
        self.model_calls: list[tuple[str, dict[str, Any]]] = []
        self.transformers = SimpleNamespace(
            __version__=transformers_version,
            AutoTokenizer=SimpleNamespace(from_pretrained=self._tokenizer),
            AutoModel=SimpleNamespace(from_pretrained=self._model),
        )

    def _tokenizer(self, name: str, **options: Any) -> Any:
        self.tokenizer_calls.append((name, options))
        if self.tokenizer_error is not None:
            raise self.tokenizer_error
        return self.tokenizer

    def _model(self, name: str, **options: Any) -> FakeModel:
        self.model_calls.append((name, options))
        if self.model_error is not None:
            raise self.model_error
        return self.model


def install(
    monkeypatch: pytest.MonkeyPatch, libraries: FakeLibraries, *, missing: tuple[str, ...] = ()
) -> None:
    """Make the encoder's lazy imports of torch and transformers return the fakes."""
    real = importlib.import_module
    fakes = {"torch": libraries.torch, "transformers": libraries.transformers}

    def fake_import(name: str, *args: Any) -> Any:
        if name in missing:
            raise ImportError(f"No module named {name!r}", name=name)
        return fakes[name] if name in fakes else real(name, *args)

    monkeypatch.setattr("copernicus.embeddings.encoder.importlib.import_module", fake_import)
