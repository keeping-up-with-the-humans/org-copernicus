"""What an embedding pass refuses: a window without an attended token, a vector without length.

The tests run the encoder on the fake tokenizer and table model of `fake_runtime`, so no weights
are loaded; they need torch and carry the embed marker.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from fake_runtime import (
    DROPPED,
    FAKE_VERSIONS,
    SPECIAL,
    DroppingTokenizer,
    TableModel,
    fake_runtime,
)

from copernicus.embeddings import Encoder, EncoderSpec
from copernicus.embeddings.checkpoint import fragments_fingerprint, write_checkpoint
from copernicus.errors import ConfigurationError

pytestmark = pytest.mark.embed
WINDOW = 4
FOUR_SPECIAL = " ".join([SPECIAL] * WINDOW)
TEXTS = ["a b", DROPPED, "c", f"w x y z {FOUR_SPECIAL}"]
IDS = ["kept", "dropped", "plain", "special"]
REMEDY = "leave these texts out, or change them so that the tokenizer gives every window a token"
WITHOUT_TOKEN = {
    "mean": "so mean pooling has no token vector to average there",
    "last_token": "so last-token pooling has no last token to take there",
    "cls": "so cls pooling has no first token to take there",
}
POOLINGS = ["mean", "last_token", "cls"]
SHORT = "which the encoder does not scale to unit length"
OTHER_ENCODER = "leave these texts out or embed them with another encoder"


def spec_of(pooling: str, padding_side: str = "right", window: int = WINDOW) -> EncoderSpec:
    """A window of four tokens by default, none of them reserved: the tokenizer adds none."""
    return EncoderSpec(
        "fake",
        "org/fake",
        pooling,  # type: ignore[arg-type]
        window,
        padding_side=padding_side,  # type: ignore[arg-type]
        special_tokens_reserved=0,
    )


def encoder_on(
    monkeypatch: pytest.MonkeyPatch,
    pooling: str,
    model: TableModel | None = None,
    *,
    tokenizer: DroppingTokenizer | None = None,
    padding_side: str = "right",
    batch_size: int = 32,
    window: int = WINDOW,
) -> tuple[Encoder, TableModel]:
    """Return an encoder whose runtime is the fake one, with the dropping tokenizer."""
    model = model or TableModel()
    runtime = fake_runtime(model, pooling, tokenizer or DroppingTokenizer(padding_side))
    monkeypatch.setattr(Encoder, "_load", lambda self: runtime)
    spec = spec_of(pooling, padding_side, window)
    return Encoder(spec, batch_size=batch_size), model


def without_token(pooling: str, count: int, named: str) -> str:
    """The whole refusal of a window without an attended token, as a pattern."""
    message = (
        f"{count} texts have a window that the tokenizer turned into no token, "
        f"{WITHOUT_TOKEN[pooling]}: {named}; {REMEDY}"
    )
    return f"^{re.escape(message)}$"


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("pooling", POOLINGS)
def test_a_window_without_a_token_is_refused_by_text_before_the_model_runs(
    monkeypatch: pytest.MonkeyPatch, pooling: str, padding_side: str
) -> None:
    encoder, model = encoder_on(monkeypatch, pooling, padding_side=padding_side)
    with pytest.raises(ConfigurationError, match=without_token(pooling, 2, "'dropped', 'special'")):
        encoder.embed(TEXTS, IDS)
    assert model.batches == []


def test_the_refusal_names_the_texts_of_every_batch_of_the_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Batches follow the length of the windows: "c" runs first, "@" stops the pass, and the
    # longer "@ @ @ @ @" in a later batch is named as well.
    encoder, model = encoder_on(monkeypatch, "mean", batch_size=1)
    texts = ["a b", " ".join([DROPPED] * 5), "c", DROPPED]
    with pytest.raises(ConfigurationError, match=without_token("mean", 2, "'t1', 't3'")):
        encoder.embed(texts, ["t0", "t1", "t2", "t3"])
    assert model.batches == [1]


def test_the_refusal_names_a_text_once_and_the_first_ten_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, _ = encoder_on(monkeypatch, "mean")
    twice = " ".join([SPECIAL] * (2 * WINDOW))
    with pytest.raises(ConfigurationError, match=without_token("mean", 1, "'both'")):
        encoder.embed(["a b", twice], ["kept", "both"])
    many = [DROPPED] * 12
    shown = ", ".join(f"'t{index}'" for index in range(10)) + " and 2 more"
    with pytest.raises(ConfigurationError, match=without_token("mean", 12, shown)):
        encoder.embed(["a b", *many], ["kept", *(f"t{index}" for index in range(12))])


@pytest.mark.parametrize("pooling", POOLINGS)
def test_a_resumed_window_without_a_token_is_refused_before_the_first_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pooling: str
) -> None:
    encoder, model = encoder_on(monkeypatch, pooling)
    texts, ids = ["a b", TEXTS[3]], ["kept", "special"]
    fragments = ["a b", "w x y z", ""]
    fingerprint = fragments_fingerprint(
        encoder.spec, fragments, resolved_revision=FAKE_VERSIONS["resolved_revision"]
    )
    checkpoint = tmp_path / "state"
    finished = np.asarray([2], dtype=np.int64)
    write_checkpoint(checkpoint, fingerprint, finished, np.zeros((1, 4)), np.zeros(1))
    with pytest.raises(ConfigurationError, match=without_token(pooling, 1, "'special'")):
        encoder.embed(texts, ids, checkpoint=checkpoint)
    assert model.batches == [] and (tmp_path / "state.npz").is_file()


@pytest.mark.parametrize("pooling", POOLINGS)
def test_a_batch_of_windows_without_any_token_is_refused_by_text(
    monkeypatch: pytest.MonkeyPatch, pooling: str
) -> None:
    # One window per batch: the batch of the dropped text has no token position at all.
    encoder, model = encoder_on(monkeypatch, pooling, batch_size=1)
    with pytest.raises(ConfigurationError, match=without_token(pooling, 1, "'dropped'")):
        encoder.embed([DROPPED, "a b"], ["dropped", "kept"])
    assert model.batches == []


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_cls_pooling_reads_the_first_token_of_a_text_on_either_padding_side(
    monkeypatch: pytest.MonkeyPatch, padding_side: str
) -> None:
    # A text of one token beside a text of four: the shorter row is padded, and the table
    # model's vector at a padding position is far from every token's.
    tokenizer = DroppingTokenizer(padding_side)
    first_long, first_short = (
        tokenizer.encode(word, add_special_tokens=False)[0] for word in ("p", "short")
    )
    encoder, model = encoder_on(monkeypatch, "cls", tokenizer=tokenizer, padding_side=padding_side)
    result = encoder.embed(["p q r s", "short"], ["long", "short"])
    expected = model.table[[first_long, first_short]].numpy()
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(result.vectors, expected, atol=1e-6)


def test_a_checkpoint_of_cls_pooling_at_position_zero_is_not_resumed_with_left_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The fingerprint as it was while cls pooling read position 0, with a finished short window
    # whose saved vector would be a padding position's.
    encoder, model = encoder_on(monkeypatch, "cls", padding_side="left")
    spec, fragments = encoder.spec, ["p q r s", "short"]
    commit = FAKE_VERSIONS["resolved_revision"]
    settings = (
        *(spec.hf_id, spec.revision, commit, spec.pooling, spec.max_tokens, spec.dtype),
        *(spec.padding_side, spec.trust_remote_code, len(fragments)),
    )
    digest = hashlib.sha256("|".join(str(value) for value in settings).encode("utf-8"))
    for fragment in fragments:
        digest.update(b"\x00" + fragment.encode("utf-8"))
    checkpoint = tmp_path / "state"
    write_checkpoint(checkpoint, digest.hexdigest(), np.asarray([1]), np.ones((1, 4)), np.ones(1))
    with pytest.raises(ConfigurationError, match="belongs to other texts or another encoder"):
        encoder.embed(fragments, ["long", "short"], checkpoint=checkpoint)
    assert model.batches == [] and (tmp_path / "state.npz").is_file()


def test_a_resumed_cls_pass_with_left_padding_equals_a_pass_in_one_go(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    texts, ids = ["p q r s", "short", "a b"], ["long", "short", "two"]
    whole, _ = encoder_on(monkeypatch, "cls", padding_side="left", batch_size=2)
    expected = whole.embed(texts, ids)
    dying, _ = encoder_on(
        monkeypatch, "cls", TableModel(fail_after_calls=1), padding_side="left", batch_size=2
    )
    with pytest.raises(RuntimeError, match="model killed"):
        dying.embed(texts, ids, checkpoint=tmp_path / "state", checkpoint_every=1)
    resumed, model = encoder_on(monkeypatch, "cls", padding_side="left", batch_size=2)
    result = resumed.embed(texts, ids, checkpoint=tmp_path / "state", checkpoint_every=1)
    assert model.batches == [1]
    assert np.array_equal(result.vectors, expected.vectors)
    assert np.array_equal(result.chunk_vectors, expected.chunk_vectors)


def zeroed(word: str) -> tuple[TableModel, DroppingTokenizer]:
    """A table model whose vector of the word is zero, and the tokenizer that gives its id."""
    tokenizer = DroppingTokenizer()
    model = TableModel()
    model.table[tokenizer.encode(word, add_special_tokens=False)[0]] = 0.0
    return model, tokenizer


def test_a_text_vector_without_length_is_refused_by_text(monkeypatch: pytest.MonkeyPatch) -> None:
    model, tokenizer = zeroed("z")
    encoder, _ = encoder_on(monkeypatch, "mean", model, tokenizer=tokenizer)
    message = f"1 texts have a vector of length below 1e-12, {SHORT}: 'zero'; {OTHER_ENCODER}"
    with pytest.raises(ConfigurationError, match=f"^{re.escape(message)}$"):
        encoder.embed(["a b", "z"], ["kept", "zero"])


def test_a_window_vector_without_length_is_refused_by_text(monkeypatch: pytest.MonkeyPatch) -> None:
    model, tokenizer = zeroed("z")
    encoder, _ = encoder_on(monkeypatch, "mean", model, tokenizer=tokenizer)
    message = (
        "1 texts have a window whose pooled vector has a length below 1e-12, "
        f"{SHORT}: 'half'; {OTHER_ENCODER}"
    )
    with pytest.raises(ConfigurationError, match=f"^{re.escape(message)}$"):
        encoder.embed(["a b", "w x y w z z z z"], ["kept", "half"])


def test_windows_that_cancel_leave_a_text_vector_without_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two windows of two tokens whose vectors are opposite: their mean is exactly zero.
    tokenizer = DroppingTokenizer()
    p, q, r, s = tokenizer.encode("p q r s", add_special_tokens=False)
    model = TableModel()
    model.table[r], model.table[s] = -model.table[p], -model.table[q]
    encoder, _ = encoder_on(monkeypatch, "mean", model, tokenizer=tokenizer, window=2)
    message = f"1 texts have a vector of length below 1e-12, {SHORT}: 'cancel'; {OTHER_ENCODER}"
    with pytest.raises(ConfigurationError, match=f"^{re.escape(message)}$"):
        encoder.embed(["p q r s"], ["cancel"])


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_a_vector_that_is_not_finite_is_refused_by_text(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    tokenizer = DroppingTokenizer()
    model = TableModel()
    model.table[tokenizer.encode("z", add_special_tokens=False)[0], 0] = value
    encoder, _ = encoder_on(monkeypatch, "mean", model, tokenizer=tokenizer)
    with pytest.raises(ConfigurationError) as refused:
        encoder.embed(["a b", "z"], ["kept", "broken"])
    message = str(refused.value)
    assert "that is not finite" in message and "'broken'" in message and "'kept'" not in message


@pytest.mark.filterwarnings("error")
def test_a_weighted_sum_that_overflows_is_refused_by_text_not_by_a_numpy_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A window vector of finite length 1e308 and two tokens: its weighted sum overflows before
    # the text's vector is scaled. With warnings as errors the pass still names the text.
    import torch

    tokenizer = DroppingTokenizer()
    first = tokenizer.encode("z", add_special_tokens=False)[0]
    model = TableModel()
    model.table = model.table.to(torch.float64)
    model.table[first] = 0.0
    model.table[first, 0] = 1e308
    encoder, _ = encoder_on(monkeypatch, "cls", model, tokenizer=tokenizer)
    with pytest.raises(ConfigurationError) as refused:
        encoder.embed(["a b", "z z"], ["kept", "huge"])
    message = str(refused.value)
    assert "that is not finite" in message and "'huge'" in message and "'kept'" not in message
