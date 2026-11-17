"""The check before an encoder loads its model, and the error of a load that fails.

`check_transformers_line` and `loading` are functions of `copernicus.embeddings._loading`, which
imports neither torch nor transformers, so these tests run where neither is installed, as in the
continuous integration. The tests of an encoder's own load, which use stand-in libraries, need
torch and carry the embed marker.
"""

from __future__ import annotations

import pytest

from copernicus.embeddings import ENCODERS, EncoderSpec
from copernicus.embeddings._loading import check_transformers_line, loading
from copernicus.errors import ConfigurationError

GTE = "gte_qwen2_7b"
GTE_REMEDY = 'pip install "transformers>=4.56,<5"'
MINE = EncoderSpec("mine", "org/mine", "mean", 64, transformers_below="4.60")
WITHOUT_BOUND = EncoderSpec("open", "org/open", "mean", 64)
TORCH = "2.5.1"


@pytest.mark.parametrize("installed", ["4.56.0", "4.57.6", "4.57.0.dev0", "v4.57.3", "0!4.57.6"])
def test_the_large_encoder_may_load_below_transformers_5(installed: str) -> None:
    check_transformers_line(ENCODERS[GTE], installed)


@pytest.mark.parametrize(
    "installed", ["5", "5.0.0rc1", "5.12.1", "5.12.1+review.local", "10.0.0", "1!4.57.6"]
)
def test_the_large_encoder_is_refused_from_transformers_5_on_with_the_reason_and_the_remedy(
    installed: str,
) -> None:
    with pytest.raises(ConfigurationError) as refused:
        check_transformers_line(ENCODERS[GTE], installed)
    message = str(refused.value)
    for part in (repr(GTE), f"transformers {installed}:", "rope_theta", GTE_REMEDY):
        assert part in message, part


@pytest.mark.parametrize("installed", ["4.garbage", "4.57.3oops", "4..5", "", "unknown"])
def test_an_installed_version_that_cannot_be_read_is_refused_for_an_encoder_with_a_bound(
    installed: str,
) -> None:
    with pytest.raises(ConfigurationError) as refused:
        check_transformers_line(ENCODERS[GTE], installed)
    message = str(refused.value)
    assert f"the installed version {installed!r} cannot be read as a version" in message
    assert GTE_REMEDY in message


def test_the_bound_of_a_specification_of_ones_own_is_checked_with_a_general_reason() -> None:
    check_transformers_line(MINE, "4.59.2")
    with pytest.raises(ConfigurationError) as refused:
        check_transformers_line(MINE, "4.60.0")
    message = str(refused.value)
    assert "'mine'" in message and "transformers_below" in message
    assert 'pip install "transformers>=4.56,<4.60"' in message and "rope_theta" not in message


@pytest.mark.parametrize("installed", ["4.57.6", "5.12.1", "99.0.0", "unknown", ""])
def test_a_specification_without_a_bound_is_not_checked(installed: str) -> None:
    check_transformers_line(WITHOUT_BOUND, installed)


def test_a_model_that_fails_to_load_raises_the_library_error_from_the_original_one() -> None:
    spec = ENCODERS[GTE]
    original = OSError("the files are not in the cache and the hub cannot be reached")
    with pytest.raises(ConfigurationError) as failed, loading("model", spec, "4.57.6", TORCH):
        raise original
    message = str(failed.value)
    for part in (
        f"encoder {GTE!r} could not load its model",
        spec.hf_id,
        f"revision {spec.revision}",
        "the repository's own code allowed",
        "transformers 4.57.6",
        f"torch {TORCH}",
        "OSError: the files are not in the cache",
    ):
        assert part in message, part
    assert failed.value.__cause__ is original


def test_a_tokenizer_that_fails_to_load_names_the_default_branch_and_the_code_not_allowed() -> None:
    spec = ENCODERS["minilm"]
    original = AttributeError("'Qwen2Config' object has no attribute 'rope_theta'")
    with pytest.raises(ConfigurationError) as failed, loading("tokenizer", spec, "5.12.1", TORCH):
        raise original
    message = str(failed.value)
    for part in (
        "encoder 'minilm' could not load its tokenizer",
        "the default branch, no revision pinned",
        "the repository's own code not allowed",
        "AttributeError: 'Qwen2Config' object has no attribute 'rope_theta'",
    ):
        assert part in message, part
    assert failed.value.__cause__ is original


def test_an_interruption_during_a_load_passes_as_it_is() -> None:
    with pytest.raises(KeyboardInterrupt), loading("model", ENCODERS["minilm"], "4.57.6", TORCH):
        raise KeyboardInterrupt


def test_a_load_that_succeeds_passes_through_unchanged() -> None:
    with loading("model", ENCODERS["minilm"], "4.57.6", TORCH):
        loaded = "the model"
    assert loaded == "the model"
