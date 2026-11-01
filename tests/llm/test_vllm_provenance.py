"""The vLLM backend's provenance: the engine options are recorded as JSON, and only JSON ones."""

from __future__ import annotations

import re

import pytest
from vllm_support import COMMIT, FakeLLM, install_fake_vllm

from copernicus.errors import SettingsError
from copernicus.llm import BackendInfo, VLLMBackend


@pytest.fixture()
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> type[FakeLLM]:
    """Install a fake ``vllm`` module and return its ``LLM`` class."""
    return install_fake_vllm(monkeypatch)


def test_provenance_describes_the_engine_after_the_engine_kwargs(
    fake_vllm: type[FakeLLM],
) -> None:
    backend = VLLMBackend(
        "org/model-AWQ",
        chat_template_kwargs={"enable_thinking": False},
        engine_kwargs={"quantization": "gptq", "dtype": "bfloat16", "max_model_len": 4096},
    )
    assert backend.info() == BackendInfo("org/model-AWQ", "vllm", "gptq", 4096)
    assert backend.provenance() == {
        "kind": "vllm",
        "model": "org/model-AWQ",
        "quantization": "gptq",
        "dtype": "bfloat16",
        "kv_cache_dtype": "none",
        "max_model_len": "4096",
        "trust_remote_code": "False",
        "revision": "none",
        "tokenizer_revision": "none",
        "engine_options": '{"dtype":"bfloat16","max_model_len":4096,"quantization":"gptq"}',
        "chat_template_kwargs": '{"enable_thinking":false}',
        "top_p": "0.9",
    }
    plain = VLLMBackend("plain-model", kv_cache_dtype="fp8").provenance()
    assert [plain[key] for key in ("quantization", "dtype", "kv_cache_dtype")] == [
        "none",
        "none",
        "fp8",
    ]
    assert plain["chat_template_kwargs"] == "{}"


def test_provenance_records_the_pinned_commits_and_the_engine_options(
    fake_vllm: type[FakeLLM],
) -> None:
    backend = VLLMBackend(
        "m",
        trust_remote_code=True,
        engine_kwargs={"revision": COMMIT, "enforce_eager": True, "max_num_seqs": 8},
    )
    provenance = backend.provenance()
    assert (provenance["revision"], provenance["tokenizer_revision"]) == (COMMIT, COMMIT)
    assert provenance["engine_options"] == (
        '{"enforce_eager":true,"max_num_seqs":8,"revision":"' + COMMIT + '"}'
    )


def test_no_engine_options_are_recorded_as_an_empty_mapping(fake_vllm: type[FakeLLM]) -> None:
    assert VLLMBackend("m").provenance()["engine_options"] == "{}"
    assert VLLMBackend("m", engine_kwargs={}).provenance()["engine_options"] == "{}"


def test_a_nested_option_is_recorded_as_nested_canonical_json_with_typed_values(
    fake_vllm: type[FakeLLM],
) -> None:
    options = {"hf_overrides": {"b": 1, "a": [1, 2.5, None, True]}, "enforce_eager": False}
    backend = VLLMBackend("m", engine_kwargs=options)
    assert backend.provenance()["engine_options"] == (
        '{"enforce_eager":false,"hf_overrides":{"a":[1,2.5,null,true],"b":1}}'
    )
    assert fake_vllm.instances[0].kwargs["hf_overrides"] is options["hf_overrides"]


def test_a_set_option_is_refused_before_the_engine_is_built(fake_vllm: type[FakeLLM]) -> None:
    with pytest.raises(SettingsError, match=r"engine option 'hf_overrides' is not a JSON value"):
        VLLMBackend("m", engine_kwargs={"enforce_eager": True, "hf_overrides": {1, 2}})
    assert fake_vllm.instances == []
    with pytest.raises(SettingsError, match=r"\(set\); engine options are recorded") as refused:
        VLLMBackend("m", engine_kwargs={"hf_overrides": {1, 2}})
    assert isinstance(refused.value.__cause__, TypeError)
    assert "strings, numbers, booleans, None, and lists or dicts of them" in str(refused.value)
    assert fake_vllm.instances == []


def test_an_option_of_another_type_is_refused_naming_its_type(fake_vllm: type[FakeLLM]) -> None:
    with pytest.raises(SettingsError, match=r"engine option 'logits_processor' .*\(object\)"):
        VLLMBackend("m", engine_kwargs={"logits_processor": object()})
    assert fake_vllm.instances == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_is_refused_naming_its_key(
    fake_vllm: type[FakeLLM], value: float
) -> None:
    with pytest.raises(SettingsError, match=rf"engine option 'swap_space' .*\(float {value}\)"):
        VLLMBackend("m", engine_kwargs={"swap_space": value})
    assert fake_vllm.instances == []


def test_a_non_finite_number_inside_a_nested_option_is_refused_naming_the_top_key(
    fake_vllm: type[FakeLLM],
) -> None:
    with pytest.raises(
        SettingsError, match=r"engine option 'hf_overrides' holds a value that is not JSON \(dict\)"
    ):
        VLLMBackend("m", engine_kwargs={"hf_overrides": {"scale": [1.0, float("nan")]}})
    assert fake_vllm.instances == []


@pytest.mark.parametrize(
    ("value", "kind"),
    [(float("nan"), "float nan"), (float("inf"), "float inf"), ({1, 2}, "set")],
    ids=["nan", "inf", "set"],
)
def test_a_window_that_is_not_a_json_value_is_refused_before_it_is_read(
    fake_vllm: type[FakeLLM], value: object, kind: str
) -> None:
    with pytest.raises(SettingsError, match=rf"engine option 'max_model_len' .*\({kind}\)"):
        VLLMBackend("m", engine_kwargs={"max_model_len": value})
    assert fake_vllm.instances == []


def test_option_names_that_are_not_strings_are_refused(fake_vllm: type[FakeLLM]) -> None:
    for options in ({1: "x"}, {"b": "c", 1: "a"}):
        with pytest.raises(
            SettingsError, match=r"engine option names must be strings; got 1 \(int\)"
        ):
            VLLMBackend("m", engine_kwargs=options)  # type: ignore[arg-type]
    assert fake_vllm.instances == []


@pytest.mark.parametrize(
    "value",
    ["abc", [1], None, 3.5, True, 0, -4],
    ids=["str", "list", "none", "float", "bool", "zero", "negative"],
)
def test_a_window_that_is_not_a_positive_integer_is_refused_before_the_engine_is_built(
    fake_vllm: type[FakeLLM], value: object
) -> None:
    expected = re.escape(
        "engine option 'max_model_len' must be an integer of at least 1, "
        f"got {value!r} ({type(value).__name__})"
    )
    with pytest.raises(SettingsError, match=expected):
        VLLMBackend("m", engine_kwargs={"max_model_len": value})
    assert fake_vllm.instances == []


def test_an_integer_window_from_the_engine_options_is_applied(fake_vllm: type[FakeLLM]) -> None:
    backend = VLLMBackend("m", engine_kwargs={"max_model_len": 2048})
    assert backend.info().context_window == 2048
    assert fake_vllm.instances[0].kwargs["max_model_len"] == 2048
    assert backend.provenance()["max_model_len"] == "2048"


@pytest.mark.parametrize(
    ("options", "key"),
    [
        ({"hf_overrides": {1: "a"}}, "1 (int)"),
        ({"hf_overrides": {"ok": {2.5: "x"}}}, "2.5 (float)"),
        ({"hf_overrides": [{"ok": 1}, {2: "b"}]}, "2 (int)"),
        ({"hf_overrides": ({True: 1},)}, "True (bool)"),
        ({"hf_overrides": {None: 1}}, "None (NoneType)"),
        ({"hf_overrides": {1: "a", "b": 2}}, "1 (int)"),
    ],
    ids=["int", "float-deep", "in-list", "bool-in-tuple", "none", "mixed"],
)
def test_a_nested_mapping_key_that_is_not_a_string_is_refused_at_every_level(
    fake_vllm: type[FakeLLM], options: dict[str, object], key: str
) -> None:
    expected = re.escape(
        f"engine option 'hf_overrides' holds a mapping whose key {key} is not a string; "
        "names must be strings at every level"
    )
    with pytest.raises(SettingsError, match=expected):
        VLLMBackend("m", engine_kwargs=options)
    assert fake_vllm.instances == []


def test_a_nested_mapping_with_string_keys_is_recorded_as_nested_canonical_json(
    fake_vllm: type[FakeLLM],
) -> None:
    options = {"hf_overrides": [{"b": {"y": 1, "x": [{"d": 2, "c": 3}]}, "a": None}]}
    backend = VLLMBackend("m", engine_kwargs=options)
    assert backend.provenance()["engine_options"] == (
        '{"hf_overrides":[{"a":null,"b":{"x":[{"c":3,"d":2}],"y":1}}]}'
    )


def test_an_option_that_holds_itself_is_refused_as_not_json(fake_vllm: type[FakeLLM]) -> None:
    loop: list[object] = []
    loop.append(loop)
    with pytest.raises(SettingsError, match=r"engine option 'hf_overrides' holds a value"):
        VLLMBackend("m", engine_kwargs={"hf_overrides": loop})
    assert fake_vllm.instances == []
