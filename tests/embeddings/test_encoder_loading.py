"""Tests for loading an encoder, against fake ``torch`` and ``transformers`` modules."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from fake_libraries import (
    GTE,
    GTE_REMEDY,
    GTE_REVISION,
    TRANSFORMERS_4,
    TRANSFORMERS_5,
    FakeLibraries,
    install,
)

from copernicus.embeddings import ENCODERS, Encoder, EncoderSpec
from copernicus.errors import ConfigurationError

pytest.importorskip("torch")
pytestmark = pytest.mark.embed

EMBED_HINT = (
    'install the embed extra: pip install ".[embed]" from the package\'s root folder, '
    'or pip install "org-copernicus[embed]" once a release has been uploaded to the package index'
)
OTHER_REGISTERED = ("minilm", "mpnet", "bge_m3")


def test_a_large_encoder_loads_with_its_dtype_remote_code_and_padding_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libraries = FakeLibraries(cuda=True, transformers_version=TRANSFORMERS_4)
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
    libraries = FakeLibraries(transformers_version=TRANSFORMERS_4)
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


def test_gte_under_transformers_5_is_refused_before_any_load_naming_the_reason_and_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libraries = FakeLibraries(transformers_version=TRANSFORMERS_5)
    install(monkeypatch, libraries)
    with pytest.raises(ConfigurationError) as refused:
        Encoder(GTE)._load()
    message = str(refused.value)
    for part in (repr(GTE), TRANSFORMERS_5, "rope_theta", "rope_parameters", GTE_REMEDY):
        assert part in message, part
    assert libraries.tokenizer_calls == [] and libraries.model_calls == []


def test_gte_under_transformers_4_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    libraries = FakeLibraries(transformers_version=TRANSFORMERS_4)
    install(monkeypatch, libraries)
    Encoder(GTE)._load()
    assert [name for name, _ in libraries.tokenizer_calls] == [ENCODERS[GTE].hf_id]
    assert [name for name, _ in libraries.model_calls] == [ENCODERS[GTE].hf_id]


@pytest.mark.parametrize("version", [TRANSFORMERS_4, TRANSFORMERS_5])
@pytest.mark.parametrize("name", OTHER_REGISTERED)
def test_the_other_registered_encoders_load_under_both_lines(
    monkeypatch: pytest.MonkeyPatch, name: str, version: str
) -> None:
    libraries = FakeLibraries(transformers_version=version)
    install(monkeypatch, libraries)
    Encoder(name)._load()
    assert [hf_id for hf_id, _ in libraries.model_calls] == [ENCODERS[name].hf_id]


@pytest.mark.parametrize(
    ("version", "loads"),
    [
        ("4.56.0", True),
        ("4.57.0.dev0", True),
        ("4.99.99", True),
        ("v4.57.3", True),
        (" 4.57.3 ", True),
        ("4.57.3+review.local", True),
        ("4.57.3.post1", True),
        ("0!4.57.6", True),
        ("5", False),
        ("5.0.0rc1", False),
        ("5.0.0.dev0", False),
        ("5.23.0", False),
        ("5.12.1+review.local", False),
        ("10.0.0", False),
        ("1!5.12.1", False),
        ("1!4.57.6", False),
    ],
)
def test_gte_reads_the_release_numbers_of_the_installed_version(
    monkeypatch: pytest.MonkeyPatch, version: str, loads: bool
) -> None:
    libraries = FakeLibraries(transformers_version=version)
    install(monkeypatch, libraries)
    if loads:
        Encoder(GTE)._load()
    else:
        with pytest.raises(ConfigurationError, match=re.escape(GTE_REMEDY)):
            Encoder(GTE)._load()
    assert len(libraries.model_calls) == int(loads)


@pytest.mark.parametrize("version", ["4.garbage", "4.57.3oops", "4..5", "4.57.3 rc1", "", "x"])
def test_an_installed_version_that_the_parser_cannot_read_is_refused_before_any_load(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    libraries = FakeLibraries(transformers_version=version)
    install(monkeypatch, libraries)
    with pytest.raises(ConfigurationError) as refused:
        Encoder(GTE)._load()
    message = str(refused.value)
    assert f"the installed version {version!r} cannot be read as a version" in message
    assert GTE_REMEDY in message
    assert libraries.tokenizer_calls == [] and libraries.model_calls == []


def test_an_installed_version_that_cannot_be_read_is_refused_only_for_a_bounded_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libraries = FakeLibraries(transformers_version="unknown")
    install(monkeypatch, libraries)
    with pytest.raises(ConfigurationError) as refused:
        Encoder(GTE)._load()
    assert "'unknown'" in str(refused.value) and GTE_REMEDY in str(refused.value)
    assert libraries.model_calls == []
    Encoder("minilm")._load()
    assert len(libraries.model_calls) == 1


def test_the_bound_of_a_specification_of_ones_own_is_checked_with_a_general_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mine = EncoderSpec("mine", "org/mine", "mean", 64, transformers_below="4.60")
    refused_libraries = FakeLibraries(transformers_version="4.60.0")
    install(monkeypatch, refused_libraries)
    with pytest.raises(ConfigurationError) as refused:
        Encoder(mine)._load()
    message = str(refused.value)
    assert "'mine'" in message and "4.60.0" in message and "transformers_below" in message
    assert 'pip install "transformers>=4.56,<4.60"' in message and "rope_theta" not in message
    assert refused_libraries.model_calls == []
    loaded_libraries = FakeLibraries(transformers_version="4.59.2")
    install(monkeypatch, loaded_libraries)
    Encoder(mine)._load()
    assert len(loaded_libraries.model_calls) == 1


def test_another_revision_of_gte_keeps_the_bound_but_not_the_reason_of_the_pinned_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = replace(ENCODERS[GTE], revision="0" * 40)
    install(monkeypatch, FakeLibraries(transformers_version=TRANSFORMERS_5))
    with pytest.raises(ConfigurationError) as refused:
        Encoder(other)._load()
    assert GTE_REMEDY in str(refused.value) and "rope_theta" not in str(refused.value)


@pytest.mark.parametrize("version", [TRANSFORMERS_4, TRANSFORMERS_5, "99.0.0", "unknown"])
def test_a_specification_without_a_bound_loads_under_any_version(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    libraries = FakeLibraries(transformers_version=version)
    install(monkeypatch, libraries)
    Encoder(EncoderSpec("mine", "org/mine", "mean", 64))._load()
    assert [hf_id for hf_id, _ in libraries.model_calls] == ["org/mine"]
