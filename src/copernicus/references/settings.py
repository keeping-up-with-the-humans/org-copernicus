"""Settings of reference-text generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = ["STYLES", "GeneratorSettings", "Style"]

Style = Literal["neutral", "register"]
STYLES = ("neutral", "register")
TEMPERATURE_MAX = 2.0
INTEGER_FIELDS = (
    "texts_per_call",
    "calls_per_cell",
    "max_new_tokens",
    "seed",
    "max_retries",
    "concurrency",
)
POSITIVE_FIELDS = (
    "texts_per_call",
    "calls_per_cell",
    "max_new_tokens",
    "max_retries",
    "concurrency",
)


@dataclass(frozen=True)
class GeneratorSettings:
    """How reference texts are requested from a writer model.

    Attributes:
        style: ``"neutral"`` leaves the medium of the texts open; ``"register"`` has the
            writer produce the format named by ``register_description``.
        register_description: One or two imperative instruction sentences that say what
            format the texts take and how they sound, for example "Write each text as a
            short piece of written feedback on a colleague's draft report, in plain,
            slightly informal wording."; inserted unchanged into the system prompt after
            "...might actually produce.". Required for ``"register"`` and ``None`` for
            ``"neutral"``.
        texts_per_call: Number of texts asked for, and required, in one call.
        calls_per_cell: Number of calls per writer and cell.
        temperature: Sampling temperature, 0 to 2.
        max_new_tokens: Upper bound on the generated tokens of a call.
        seed: Base seed; a call adds its index and its cell's position.
        max_retries: Attempts per call, counting the first.
        concurrency: First attempts sent to a writer together, through its
            ``generate_batch``; retries are sent one at a time. It changes how fast a run
            goes, not what it writes, so a run may resume with another value. A failing
            request of a batch, or a failing retry of one of its calls, discards the
            answers of the batch that were not yet recorded: up to ``concurrency - 1``
            first-attempt answers, paid for but missing from the resume file and from
            ``usage``. A concurrency of 1 loses none.
    """

    style: Style = "neutral"
    register_description: str | None = None
    texts_per_call: int = 10
    calls_per_cell: int = 1
    temperature: float = 0.9
    max_new_tokens: int = 16384
    seed: int = 42
    max_retries: int = 3
    concurrency: int = 1

    def __post_init__(self) -> None:
        """Check the settings.

        Raises:
            SettingsError: If a setting has the wrong type or is outside its range, or the
                style and the description do not fit together.
        """
        for name in INTEGER_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise SettingsError(f"{name} must be an integer, got {value!r}")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise SettingsError(f"temperature must be a number, got {self.temperature!r}")
        if self.style not in STYLES:
            raise SettingsError(f"style must be one of {STYLES}, got {self.style!r}")
        described = self.register_description is not None and self.register_description.strip()
        if self.style == "register" and not described:
            raise SettingsError("style 'register' needs a non-empty register_description")
        if self.style == "neutral" and self.register_description is not None:
            raise SettingsError("register_description applies to style 'register' only")
        for name in POSITIVE_FIELDS:
            if getattr(self, name) < 1:
                raise SettingsError(f"{name} must be at least 1, got {getattr(self, name)}")
        if not 0 <= self.temperature <= TEMPERATURE_MAX:
            raise SettingsError(
                f"temperature must lie between 0 and {TEMPERATURE_MAX}, got {self.temperature}"
            )
