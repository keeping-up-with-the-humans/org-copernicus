"""Tests for the encoder registry."""

from __future__ import annotations

import logging
import re
import subprocess
import sys

import pytest
from ci_support import pyproject_text

from copernicus.embeddings import DEFAULT_ENCODER, ENCODERS, EncoderSpec, resolve_encoder
from copernicus.errors import SettingsError

REGISTRY_LOGGER = "copernicus.embeddings.registry"
GTE_QWEN2_REVISION = "a8d08b36ada9cacfe34c4d6f80957772a025daf2"
NO_COMMIT = (
    "pins no commit: set revision to the 40-character commit hash whose code you have reviewed"
)


def test_registry_has_four_encoders_and_default_mpnet() -> None:
    assert set(ENCODERS) == {"minilm", "mpnet", "bge_m3", "gte_qwen2_7b"}
    assert DEFAULT_ENCODER == "mpnet"
    assert ENCODERS["bge_m3"].pooling == "cls"
    assert ENCODERS["gte_qwen2_7b"].dtype == "bfloat16"
    spec = resolve_encoder("org/some-model", pooling="cls", max_tokens=128)
    assert spec.hf_id == "org/some-model" and spec.max_tokens == 128 and spec.pooling == "cls"
    with pytest.raises(SettingsError):
        resolve_encoder("x", pooling="median")


def test_an_ad_hoc_id_without_pooling_or_window_is_loaded_with_defaults_and_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=REGISTRY_LOGGER):
        spec = resolve_encoder("org/some-model")
    assert (spec.pooling, spec.max_tokens) == ("mean", 512)
    (record,) = caplog.records
    message = record.getMessage()
    assert "org/some-model" in message and "mean" in message and "512" in message
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=REGISTRY_LOGGER):
        resolve_encoder("org/some-model", pooling="cls", max_tokens=128)
    assert caplog.records == []


def test_options_that_contradict_a_registered_or_given_spec_are_refused() -> None:
    assert resolve_encoder("minilm", pooling="mean", max_tokens=256) is ENCODERS["minilm"]
    with pytest.raises(SettingsError, match="minilm.*mean.*256"):
        resolve_encoder("minilm", pooling="cls")
    with pytest.raises(SettingsError, match="minilm"):
        resolve_encoder("minilm", max_tokens=8)
    custom = EncoderSpec("mine", "org/mine", "mean", 64)
    with pytest.raises(SettingsError, match="mine"):
        resolve_encoder(custom, max_tokens=32)


def test_registered_encoders_carry_their_ids_windows_and_poolings() -> None:
    expected = {
        "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean", 256, None, False, None),
        "mpnet": ("sentence-transformers/all-mpnet-base-v2", "mean", 384, None, False, None),
        "bge_m3": ("BAAI/bge-m3", "cls", 8192, None, False, None),
        "gte_qwen2_7b": (
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "last_token",
            8192,
            "bfloat16",
            True,
            GTE_QWEN2_REVISION,
        ),
    }
    for name, (hf_id, pooling, window, dtype, remote, revision) in expected.items():
        spec = ENCODERS[name]
        assert (spec.name, spec.hf_id, spec.pooling, spec.max_tokens) == (
            name,
            hf_id,
            pooling,
            window,
        )
        assert (
            spec.dtype,
            spec.trust_remote_code,
            spec.special_tokens_reserved,
            spec.revision,
        ) == (dtype, remote, 2, revision)


def test_only_the_encoder_that_runs_remote_code_pins_a_revision() -> None:
    pinned = {name for name, spec in ENCODERS.items() if spec.revision is not None}
    remote = {name for name, spec in ENCODERS.items() if spec.trust_remote_code}
    assert pinned == remote == {"gte_qwen2_7b"}


def test_remote_code_requires_a_pinned_revision() -> None:
    with pytest.raises(SettingsError, match=NO_COMMIT):
        EncoderSpec("x", "org/model", "mean", 128, trust_remote_code=True)
    spec = EncoderSpec(
        "x", "org/model", "mean", 128, trust_remote_code=True, revision=GTE_QWEN2_REVISION
    )
    assert spec.revision == GTE_QWEN2_REVISION


def test_an_empty_revision_counts_as_unpinned() -> None:
    for revision in ("", " ", "\t\n"):
        with pytest.raises(SettingsError, match=NO_COMMIT):
            EncoderSpec("x", "org/model", "mean", 128, trust_remote_code=True, revision=revision)


@pytest.mark.parametrize(
    "revision",
    ["main", GTE_QWEN2_REVISION[:7], "v1.0", GTE_QWEN2_REVISION.upper(), GTE_QWEN2_REVISION + "0"],
    ids=["branch", "short-hash", "tag", "upper-case", "41-characters"],
)
def test_remote_code_refuses_a_branch_a_tag_or_a_short_hash(revision: str) -> None:
    with pytest.raises(SettingsError, match=NO_COMMIT):
        EncoderSpec("x", "org/model", "mean", 128, trust_remote_code=True, revision=revision)


def test_without_remote_code_a_branch_or_a_tag_may_be_pinned() -> None:
    for revision in ("main", "v1.0", GTE_QWEN2_REVISION[:7]):
        assert EncoderSpec("x", "org/model", "mean", 128, revision=revision).revision == revision


def test_the_identity_of_a_spec_lists_the_sidecar_identity_keys() -> None:
    from dataclasses import replace

    from copernicus._io import IDENTITY_KEYS

    identity = ENCODERS["gte_qwen2_7b"].identity()
    assert list(identity) == list(IDENTITY_KEYS)
    assert identity == {
        "hf_id": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "revision": GTE_QWEN2_REVISION,
        "pooling": "last_token",
        "max_tokens": 8192,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "padding_side": "right",
        "special_tokens_reserved": 2,
    }
    pinned = replace(ENCODERS["minilm"], revision="0123abcd")
    assert pinned.identity()["revision"] == "0123abcd"


def test_a_registry_name_resolves_to_its_spec_and_a_spec_to_itself() -> None:
    assert resolve_encoder("minilm") is ENCODERS["minilm"]
    custom = EncoderSpec("mine", "org/mine", "mean", 64)
    assert resolve_encoder(custom) is custom


def test_the_registry_cannot_be_changed() -> None:
    with pytest.raises(TypeError):
        ENCODERS["extra"] = ENCODERS["minilm"]  # type: ignore[index]


def test_a_spec_with_unknown_options_or_no_room_for_text_is_refused() -> None:
    with pytest.raises(SettingsError, match="pooling"):
        EncoderSpec("x", "org/x", "median", 64)  # type: ignore[arg-type]
    with pytest.raises(SettingsError, match="padding side"):
        EncoderSpec("x", "org/x", "mean", 64, padding_side="middle")  # type: ignore[arg-type]
    with pytest.raises(SettingsError, match="no room"):
        EncoderSpec("x", "org/x", "mean", 2)


def test_only_gte_qwen2_7b_needs_an_earlier_line_of_transformers() -> None:
    bounds = {name: spec.transformers_below for name, spec in ENCODERS.items()}
    assert bounds == {"minilm": None, "mpnet": None, "bge_m3": None, "gte_qwen2_7b": "5"}


@pytest.mark.parametrize(
    "bound", ["five", "", " 5", "5 ", "5.x", "v5", "5.", ".5", "5..1", "5.0rc1", 5, 5.0]
)
def test_a_transformers_bound_that_is_not_a_version_string_is_refused(bound: object) -> None:
    with pytest.raises(SettingsError, match=r"'x'.*transformers_below.*release number"):
        EncoderSpec("x", "org/x", "mean", 64, transformers_below=bound)  # type: ignore[arg-type]


@pytest.mark.parametrize("bound", ["3", "4", "4.56", "4.56.0", "4.55.9"])
def test_a_transformers_bound_that_no_version_of_the_embed_extra_lies_below_is_refused(
    bound: str,
) -> None:
    with pytest.raises(SettingsError, match=r"transformers_below.*4\.56 or later"):
        EncoderSpec("x", "org/x", "mean", 64, transformers_below=bound)


@pytest.mark.parametrize("bound", ["5", "4.57", "4.56.1", "6.0.0", "12"])
def test_a_release_number_above_the_minimum_is_a_transformers_bound(bound: str) -> None:
    assert EncoderSpec("x", "org/x", "mean", 64, transformers_below=bound).transformers_below == (
        bound
    )


def test_a_specification_without_a_transformers_bound_keeps_its_identity() -> None:
    spec = EncoderSpec("x", "org/x", "mean", 64)
    assert spec.transformers_below is None
    assert "transformers_below" not in spec.identity()
    assert "transformers_below" not in ENCODERS["gte_qwen2_7b"].identity()


def test_the_minimum_of_a_transformers_bound_is_the_lower_bound_of_the_embed_extra() -> None:
    from copernicus.embeddings.registry import TRANSFORMERS_MINIMUM

    extra = re.search(r"^embed = \[(.*)\]$", pyproject_text(), re.MULTILINE)
    assert extra is not None
    bounds = re.findall(r'"transformers([<>=!~,.0-9]*)"', extra.group(1))
    assert bounds == [">=4.56"]
    assert ".".join(str(number) for number in TRANSFORMERS_MINIMUM) == "4.56"


@pytest.mark.embed
def test_transformers_5_no_longer_provides_rope_theta_on_the_qwen2_configuration() -> None:
    from copernicus._versions import read_version

    transformers = pytest.importorskip("transformers")
    installed = read_version(transformers.__version__)
    assert installed is not None, transformers.__version__
    if installed.epoch == 0 and installed.release[0] < 5:
        pytest.skip(
            f"transformers {transformers.__version__} is of the 4 line, whose configuration "
            "objects still provide rope_theta; the fact concerns transformers 5 and later"
        )
    configuration = transformers.Qwen2Config()
    assert not hasattr(configuration, "rope_theta")
    assert "rope_theta" in configuration.rope_parameters


def test_importing_the_embeddings_package_does_not_import_torch_or_transformers() -> None:
    code = (
        "import sys, copernicus.embeddings as e; "
        "print('torch' in sys.modules, 'transformers' in sys.modules); "
        "e.Encoder('minilm')"
        "; print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.split("\n")[:2] == ["False False", "False False"]
