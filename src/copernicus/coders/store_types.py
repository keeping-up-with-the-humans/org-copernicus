"""Identity, row, and settings types of the run store."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from copernicus._version import __version__
from copernicus.coders.evidence import verifier_version
from copernicus.coders.one_stage import OneStageConfig
from copernicus.coders.render import template_hashes
from copernicus.coders.rotation import DIMENSIONS, Assignment, WordingPool
from copernicus.coders.two_stage_types import AttributeResult, TwoStageConfig

logger = logging.getLogger(__name__)

__all__ = [
    "METHODS",
    "RESERVED_KEYS",
    "ResultRow",
    "RunIdentity",
    "identity_settings",
    "wording_hash",
]

METHODS = ("two_stage", "one_stage")
WORDING_HASH_LENGTH = 16
RESERVED_KEYS = (
    "method",
    "model_name",
    "temperature",
    "seed",
    "rubric_hash",
    "rubric_id",
    "attribute_ids",
    "marker_ids",
)


@dataclass(frozen=True)
class RunIdentity:
    """Everything that must be equal for two results to belong to the same run.

    Attributes:
        method: ``"two_stage"`` or ``"one_stage"``.
        model_name: Name of the model that codes the texts.
        temperature: Sampling temperature of the run.
        seed: Seed of the wording rotation.
        rubric_hash: ``Rubric.content_hash()`` of the rubric in use.
        settings: Every config field as a string, plus ``template_hashes``,
            ``wording_hash``, ``verifier_version``, and ``library_version`` (see
            `identity_settings`); the runner adds ``shuffle_markers`` and every
            entry of the backend's provenance under ``backend_<key>``.
    """

    method: Literal["two_stage", "one_stage"]
    model_name: str
    temperature: float
    seed: int
    rubric_hash: str
    settings: Mapping[str, str]


@dataclass(frozen=True)
class ResultRow:
    """The coding result of one text in one run.

    Attributes:
        text_id: Identifier of the text, as a string.
        text_key: Integer key of the text used for seeding.
        text_hash: Text hash of the text the row was coded on
            (`copernicus.coders.text_hash.hash_text` of the text the coder received).
        run_number: Number of the run.
        assignment: The wording and shuffle seed drawn for this text and run.
        attributes: Result per attribute id; must cover every rubric attribute.
        seconds: Wall-clock time spent on the text.
        prompt_tokens: Prompt tokens spent on the text.
        completion_tokens: Generated tokens spent on the text.
    """

    text_id: str
    text_key: int
    text_hash: str
    run_number: int
    assignment: Assignment
    attributes: Mapping[str, AttributeResult]
    seconds: float
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        """Check the identifier type.

        Raises:
            TypeError: If ``text_id`` is not a string; an integer id would never
                match the keys that ``completed_keys`` returns.
        """
        if not isinstance(self.text_id, str):
            raise TypeError(f"text_id must be a str, got {self.text_id!r}")


def wording_hash(pool: WordingPool) -> str:
    """Fingerprint the wording of a pool, including which variants are enabled.

    The rotation draws among the enabled variants in row order, so the hash
    covers, per dimension, the ordered list of ``(variant_id, enabled, texts)``.

    Args:
        pool: The wording pool.

    Returns:
        The first 16 hex characters of the SHA-256 of the canonical JSON.
    """
    content = {
        dimension: [[w.variant_id, w.enabled, dict(w.texts)] for w in getattr(pool, dimension)]
        for dimension in DIMENSIONS
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:WORDING_HASH_LENGTH]


def identity_settings(config: TwoStageConfig | OneStageConfig, pool: WordingPool) -> dict[str, str]:
    """Build the coder part of the ``settings`` of a `RunIdentity`.

    The runner adds the run-level entries: ``shuffle_markers`` and the backend's
    provenance, each key prefixed ``backend_``.

    Args:
        config: The config of the coder.
        pool: The wording pool of the run.

    Returns:
        Every field of the config as ``str(value)``, plus ``template_hashes``,
        ``wording_hash``, ``verifier_version``, and ``library_version``.
    """
    settings = {f.name: str(getattr(config, f.name)) for f in dataclasses.fields(config)}
    settings["template_hashes"] = json.dumps(template_hashes(), sort_keys=True)
    settings["wording_hash"] = wording_hash(pool)
    settings["verifier_version"] = verifier_version()
    settings["library_version"] = __version__
    return settings
