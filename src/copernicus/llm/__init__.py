"""LLM backend protocol, JSON extraction, and the fake, OpenAI-compatible, and vLLM backends."""

from copernicus.llm.base import (
    TOP_P,
    Backend,
    BackendInfo,
    GenerationRequest,
    GenerationResult,
    ModelCapabilities,
    capabilities_for_model,
    capability_provenance,
    provenance_text,
)
from copernicus.llm.fake import FakeBackend, Responder, ScriptedResponder, cue_responder
from copernicus.llm.json_extract import extract_json
from copernicus.llm.openai_compatible import OpenAICompatibleBackend
from copernicus.llm.vllm_backend import VLLMBackend, clamp_max_tokens

__all__ = [
    "TOP_P",
    "Backend",
    "BackendInfo",
    "FakeBackend",
    "GenerationRequest",
    "GenerationResult",
    "ModelCapabilities",
    "OpenAICompatibleBackend",
    "Responder",
    "ScriptedResponder",
    "VLLMBackend",
    "capabilities_for_model",
    "capability_provenance",
    "clamp_max_tokens",
    "cue_responder",
    "extract_json",
    "provenance_text",
]
