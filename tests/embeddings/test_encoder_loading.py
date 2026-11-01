"""Tests for loading an encoder, against fake ``torch`` and ``transformers`` modules."""

from __future__ import annotations

import importlib
import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from copernicus.embeddings import ENCODERS, Encoder, EncoderSpec
from copernicus.errors import ConfigurationError

pytest.importorskip("torch")
pytestmark = pytest.mark.embed

EMBED_HINT = (
    'install the embed extra: pip install "org-copernicus[embed]", '
    'or pip install ".[embed]" from the package\'s root folder'
)
GTE = "gte_qwen2_7b"
GTE_REVISION = "a8d08b36ada9cacfe34c4d6f80957772a025daf2"


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
    """Fake ``torch`` and ``transformers`` that record how the encoder uses them."""

    def __init__(
        self, *, cuda: bool = False, mps: bool = False, commit: object = "f00dfeed"
    ) -> None:
        self.bfloat16 = FakeDtype()
        self.torch = SimpleNamespace(
            __version__="2.99.0",
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
            __version__="9.99.0",
            AutoTokenizer=SimpleNamespace(from_pretrained=self._tokenizer),
            AutoModel=SimpleNamespace(from_pretrained=self._model),
        )

    def _tokenizer(self, name: str, **options: Any) -> Any:
        self.tokenizer_calls.append((name, options))
        return self.tokenizer

    def _model(self, name: str, **options: Any) -> FakeModel:
        self.model_calls.append((name, options))
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


def test_a_large_encoder_loads_with_its_dtype_remote_code_and_padding_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libraries = FakeLibraries(cuda=True)
    install(monkeypatch, libraries)
    spec = ENCODERS[GTE]
    left = EncoderSpec(
        spec.name,
        spec.hf_id,
        spec.pooling,
        spec.max_tokens,
        spec.dtype,
        spec.trust_remote_code,
        padding_side="left",
        revision=spec.revision,
    )
    runtime = Encoder(left)._load()
    assert libraries.model_calls == [
        (
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            {"trust_remote_code": True, "revision": GTE_REVISION, "dtype": libraries.bfloat16},
        )
    ]
    assert libraries.tokenizer_calls == [
        (
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            {"trust_remote_code": True, "revision": GTE_REVISION},
        )
    ]
    assert libraries.tokenizer.padding_side == "left"
    assert [device.name for device in libraries.model.moved_to] == ["cuda"]
    assert libraries.model.evaluating
    assert runtime.model is libraries.model and runtime.tokenizer is libraries.tokenizer
    from copernicus.embeddings.pooling import POOLING_FUNCTIONS

    assert runtime.pool is POOLING_FUNCTIONS["last_token"]


def test_the_registry_default_padding_side_is_applied_to_the_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libraries = FakeLibraries()
    install(monkeypatch, libraries)
    Encoder(GTE)._load()
    assert libraries.tokenizer.padding_side == ENCODERS[GTE].padding_side == "right"


def test_a_small_encoder_loads_without_dtype_or_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libraries = FakeLibraries()
    install(monkeypatch, libraries)
    Encoder("minilm")._load()
    assert libraries.model_calls == [
        ("sentence-transformers/all-MiniLM-L6-v2", {"trust_remote_code": False})
    ]
    assert libraries.tokenizer_calls == [
        ("sentence-transformers/all-MiniLM-L6-v2", {"trust_remote_code": False})
    ]


def test_a_pinned_revision_is_passed_to_both_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    libraries = FakeLibraries()
    install(monkeypatch, libraries)
    pinned = replace(ENCODERS["minilm"], revision="0123abcd")
    runtime = Encoder(pinned)._load()
    assert libraries.model_calls == [
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            {"trust_remote_code": False, "revision": "0123abcd"},
        )
    ]
    assert libraries.tokenizer_calls == [
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            {"trust_remote_code": False, "revision": "0123abcd"},
        )
    ]
    assert runtime.versions == {
        "resolved_revision": "f00dfeed",
        "torch_version": "2.99.0",
        "transformers_version": "9.99.0",
    }


@pytest.mark.parametrize("commit", [None, 7, b"bytes"])
def test_a_model_that_names_no_commit_records_an_unknown_revision(
    monkeypatch: pytest.MonkeyPatch, commit: object
) -> None:
    install(monkeypatch, FakeLibraries(commit=commit))
    assert Encoder("minilm")._load().versions["resolved_revision"] == "unknown"


def test_the_model_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    libraries = FakeLibraries()
    install(monkeypatch, libraries)
    encoder = Encoder("minilm")
    assert encoder._load() is encoder._load()
    assert len(libraries.model_calls) == 1 and len(libraries.tokenizer_calls) == 1


@pytest.mark.parametrize(
    ("cuda", "mps", "requested", "expected"),
    [
        (True, True, None, "cuda"),
        (True, False, None, "cuda"),
        (False, True, None, "mps"),
        (False, False, None, "cpu"),
        (True, True, "cpu", "cpu"),
        (False, False, "mps", "mps"),
    ],
)
def test_the_device_is_the_explicit_one_else_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, requested: str | None, expected: str
) -> None:
    libraries = FakeLibraries(cuda=cuda, mps=mps)
    install(monkeypatch, libraries)
    runtime = Encoder("minilm", device=requested)._load()
    assert runtime.device.name == expected
    assert [device.name for device in libraries.model.moved_to] == [expected]


def test_a_missing_transformers_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeLibraries(), missing=("transformers",))
    with pytest.raises(
        ConfigurationError,
        match=f"transformers is not installed; {re.escape(EMBED_HINT)}",
    ):
        Encoder("minilm")._load()


def test_an_error_inside_the_library_is_not_reported_as_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, FakeLibraries(), missing=("copernicus.embeddings.pooling",))
    with pytest.raises(ImportError, match="copernicus.embeddings.pooling"):
        Encoder("minilm")._load()
