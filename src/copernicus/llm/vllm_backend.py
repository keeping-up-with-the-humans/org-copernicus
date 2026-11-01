"""Backend that runs a model in-process with the vLLM engine."""

from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from copernicus._commits import COMMIT_HASH_LENGTH, is_commit_hash
from copernicus.errors import BackendError
from copernicus.llm._engine_options import context_window, engine_options_json
from copernicus.llm.base import (
    TOP_P,
    BackendInfo,
    GenerationRequest,
    GenerationResult,
    provenance_text,
)

logger = logging.getLogger(__name__)

__all__ = ["VLLMBackend", "clamp_max_tokens"]

BACKEND_KIND = "vllm"
CONTEXT_MARGIN = 8
NAME_QUANTIZATIONS = ("awq", "gptq", "fp8")
FLOAT16_QUANTIZATIONS = ("awq", "gptq")


def clamp_max_tokens(
    max_new_tokens: int, prompt_tokens: int, max_model_len: int, margin: int = CONTEXT_MARGIN
) -> int:
    """Fit a generation budget into the context window left after the prompt.

    The engine stops a sequence at the window, so a request whose prompt plus budget
    exceeds it must not be submitted as such. The margin absorbs the off-by-one between
    the tokenizer's count and the engine's count on special tokens.

    Args:
        max_new_tokens: Requested number of generated tokens.
        prompt_tokens: Number of prompt tokens.
        max_model_len: Context window in tokens.
        margin: Tokens kept free at the end of the window.

    Returns:
        The budget, reduced to fit the window and never below 1.
    """
    return max(1, min(max_new_tokens, max_model_len - prompt_tokens - margin))


def infer_quantization(model: str) -> str | None:
    """Infer the quantization scheme from a model name.

    Args:
        model: Model name or path.

    Returns:
        ``"awq"``, ``"gptq"``, or ``"fp8"`` when the lowercase name contains it,
        otherwise ``None``.
    """
    lowered = model.lower()
    for label in NAME_QUANTIZATIONS:
        if label in lowered:
            return label
    return None


def _pin_commits(model: str, llm_kwargs: dict[str, Any]) -> None:
    """Refuse remote code without a pinned commit and pin the tokenizer to the model's commit.

    The engine loads the tokenizer from its default branch unless told otherwise, and a
    tokenizer may ship code of its own, so an unset or blank ``tokenizer_revision`` becomes the
    model's ``revision``.

    Args:
        model: The model, for the message.
        llm_kwargs: The engine options after the merge; changed in place.

    Raises:
        BackendError: If ``revision`` is not a full commit hash, or ``tokenizer_revision`` is
            set to something other than one.
    """
    revision = llm_kwargs.get("revision")
    if not is_commit_hash(revision):
        raise BackendError(
            f"backend {model!r} allows the model repository's own code to run "
            "(trust_remote_code) but pins no commit: set revision to the "
            f"{COMMIT_HASH_LENGTH}-character commit hash whose code you have reviewed, in "
            "the engine options"
        )
    tokenizer_revision = llm_kwargs.get("tokenizer_revision")
    if tokenizer_revision is None or not str(tokenizer_revision).strip():
        llm_kwargs["tokenizer_revision"] = revision
    elif not is_commit_hash(tokenizer_revision):
        raise BackendError(
            f"backend {model!r} allows the model repository's own code to run "
            f"(trust_remote_code) but its tokenizer_revision {tokenizer_revision!r} is no "
            f"{COMMIT_HASH_LENGTH}-character commit hash; set it to one, or leave it out to "
            "load the tokenizer from the model's commit"
        )


class VLLMBackend:
    """A backend that serves a model through the in-process vLLM engine.

    Prompts are rendered with the engine's own tokenizer chat template. A prompt that
    does not fit the context window raises ``BackendError`` instead of being truncated
    or silently answered with an empty string. The seed travels with each request.

    The offline engine expects one caller at a time, so ``generate`` and
    ``generate_batch`` hold a lock for the whole call: threads may share the backend, but
    their calls run one after the other and never see each other's outputs.
    """

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int = 32768,
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        dtype: str | None = None,
        kv_cache_dtype: str | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        trust_remote_code: bool = False,
        engine_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Load the model into the vLLM engine.

        Args:
            model: Hugging Face model ID or local path.
            max_model_len: Context window in tokens.
            quantization: Quantization scheme passed to vLLM; inferred from the model
                name (``awq``, ``gptq``, ``fp8``) when ``None``, and unquantized when
                the name carries none of them.
            gpu_memory_utilization: Fraction of GPU memory the engine may use.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            dtype: Weight dtype; ``float16`` for AWQ and GPTQ when ``None``, because
                their kernels reject bfloat16.
            kv_cache_dtype: KV-cache dtype override, for example ``"fp8"``.
            chat_template_kwargs: Extra arguments for the chat template, for example
                ``{"enable_thinking": False}`` for models that think by default.
            trust_remote_code: Whether the engine may run code that the model repository
                ships; off unless the repository is trusted. It requires a pinned
                commit: pass the full 40-character commit hash as ``revision`` in
                ``engine_kwargs`` (a branch, a tag or a shortened hash is refused). The
                tokenizer is pinned to the same commit through ``tokenizer_revision``
                unless ``engine_kwargs`` sets that option itself, to a commit hash as well
                (a ``None`` or blank value counts as not set), because the engine loads the
                tokenizer from the default branch otherwise.
            engine_kwargs: Further arguments for the engine constructor, merged last; the
                quantization, dtypes, window, remote-code switch, and revisions that
                `info` and `provenance` report are read after this merge. The engine
                receives the values as given, while the run provenance records them as
                JSON, so each value must be a string, a number that is finite, a boolean,
                ``None``, or a list or dict of them.

        Raises:
            BackendError: If vLLM is not installed, or the repository's own code may run
                (``trust_remote_code``) but the engine options carry no ``revision`` that is
                a full commit hash, or set a ``tokenizer_revision`` that is not one.
            SettingsError: If an option name of ``engine_kwargs``, or a key of a mapping nested
                in an option, is not a string, an option is not a JSON value, or the context
                window (``max_model_len``, also from ``engine_kwargs``) is not an integer of
                at least 1; the engine is not built then.
        """
        try:
            vllm = importlib.import_module("vllm")
        except ImportError as error:
            raise BackendError(
                "vLLM is not installed; install the vllm extra: "
                'pip install "org-copernicus[vllm]", '
                'or pip install ".[vllm]" from the package\'s root folder'
            ) from error
        engine_options = engine_options_json(engine_kwargs or {})
        self._sampling_params = vllm.SamplingParams
        self._model = model
        resolved = quantization if quantization is not None else infer_quantization(model)
        llm_kwargs: dict[str, Any] = {
            "model": model,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
        }
        if resolved is not None:
            llm_kwargs["quantization"] = resolved
        if dtype is not None:
            llm_kwargs["dtype"] = dtype
        elif resolved in FLOAT16_QUANTIZATIONS:
            llm_kwargs["dtype"] = "float16"
        if kv_cache_dtype is not None:
            llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
        llm_kwargs.update(engine_kwargs or {})
        max_model_len_option = context_window(llm_kwargs["max_model_len"])
        if llm_kwargs["trust_remote_code"]:
            _pin_commits(model, llm_kwargs)
        # Optional engine options are absent unless set; the engine then uses its defaults.
        self._quantization: str | None = llm_kwargs.get("quantization")
        self._dtype: str | None = llm_kwargs.get("dtype")
        self._kv_cache_dtype: str | None = llm_kwargs.get("kv_cache_dtype")
        self._revision: str | None = llm_kwargs.get("revision")
        self._tokenizer_revision: str | None = llm_kwargs.get("tokenizer_revision")
        self._trust_remote_code = bool(llm_kwargs["trust_remote_code"])
        self._max_model_len = max_model_len_option
        self._engine_options = engine_options
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        logger.info("loading vLLM model %s with the engine kwargs %s", model, sorted(llm_kwargs))
        self._llm = vllm.LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        self._lock = threading.Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Answer one request.

        Args:
            request: The generation request.

        Returns:
            The generated text, token counts, and the sampling parameters applied.

        Raises:
            BackendError: If the prompt does not fit the context window.
        """
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Answer the requests in one engine call, returning results in request order.

        Every prompt is rendered and measured before anything is generated, so a prompt
        that does not fit the window fails the whole call up front. Concurrent calls from
        several threads are serialised.

        Args:
            requests: The generation requests.

        Returns:
            One result per request, in the order of ``requests``.

        Raises:
            BackendError: If any prompt does not fit the context window.
        """
        with self._lock:
            return self._generate_locked(requests)

    def _generate_locked(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Render, measure, and generate; the caller holds the lock."""
        prompts = [self._render(request) for request in requests]
        token_counts = [len(self._tokenizer.encode(prompt)) for prompt in prompts]
        for position, count in enumerate(token_counts):
            if count > self._max_model_len - CONTEXT_MARGIN:
                prefix = f"request {position}: " if len(requests) > 1 else ""
                raise BackendError(
                    f"{prefix}prompt of {count} tokens exceeds the context window "
                    f"of {self._max_model_len}"
                )
        params = [
            self._sampling_params(**self._sampling_kwargs(request, count))
            for request, count in zip(requests, token_counts, strict=True)
        ]
        outputs = self._llm.generate(prompts, params)
        return [
            self._to_result(request, count, output)
            for request, count, output in zip(requests, token_counts, outputs, strict=True)
        ]

    def info(self) -> BackendInfo:
        """Describe the backend.

        Returns:
            Backend info with the model name, kind ``"vllm"``, the quantization the
            engine was given (``None`` when unquantized), and the context window.
        """
        return BackendInfo(self._model, BACKEND_KIND, self._quantization, self._max_model_len)

    def provenance(self) -> dict[str, str]:
        """Describe the settings that decide what the engine generates.

        Returns:
            Kind ``"vllm"``, the model, the quantization, weight dtype, and KV-cache dtype
            the engine was given (``"none"`` where the engine default applies), the
            context window, the remote-code switch, the revisions of the model and of the
            tokenizer (``"none"`` when the default branch is loaded), the options of
            ``engine_kwargs`` as canonical JSON of their values (sorted keys at every level, so
            a number stays a number and a nested option stays nested; ``{}`` for none), the
            chat-template arguments as canonical JSON, and the nucleus-sampling threshold.
        """
        return {
            "kind": BACKEND_KIND,
            "model": self._model,
            "quantization": provenance_text(self._quantization),
            "dtype": provenance_text(self._dtype),
            "kv_cache_dtype": provenance_text(self._kv_cache_dtype),
            "max_model_len": provenance_text(self._max_model_len),
            "trust_remote_code": provenance_text(self._trust_remote_code),
            "revision": provenance_text(self._revision),
            "tokenizer_revision": provenance_text(self._tokenizer_revision),
            "engine_options": self._engine_options,
            "chat_template_kwargs": provenance_text(self._chat_template_kwargs),
            "top_p": provenance_text(TOP_P),
        }

    def _render(self, request: GenerationRequest) -> str:
        """Render the request through the tokenizer's chat template."""
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self._chat_template_kwargs
        )
        return str(rendered)

    def _sampling_kwargs(self, request: GenerationRequest, prompt_tokens: int) -> dict[str, Any]:
        """Build the per-request sampling arguments."""
        kwargs: dict[str, Any] = {
            "max_tokens": clamp_max_tokens(
                request.max_new_tokens, prompt_tokens, self._max_model_len
            )
        }
        if request.temperature is not None and request.temperature > 0:
            kwargs["temperature"] = request.temperature
            kwargs["top_p"] = TOP_P
        else:
            kwargs["temperature"] = 0.0
        if request.seed is not None:
            kwargs["seed"] = request.seed
        return kwargs

    @staticmethod
    def _to_result(request: GenerationRequest, prompt_tokens: int, output: Any) -> GenerationResult:
        """Build the result for one request from the engine output."""
        completion = output.outputs[0]
        token_ids = getattr(completion, "token_ids", None)
        sampled = request.temperature is not None and request.temperature > 0
        return GenerationResult(
            text=completion.text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(token_ids) if token_ids is not None else None,
            temperature_applied=request.temperature if sampled else 0.0,
            top_p_applied=TOP_P if sampled else None,
            seed_applied=request.seed,
        )
