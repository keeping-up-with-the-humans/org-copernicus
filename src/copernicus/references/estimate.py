"""The token estimate of a generation run, printed before any paid call is made."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

from copernicus.errors import ConfigurationError
from copernicus.references.cells import build_cells
from copernicus.references.prompts import system_prompt, user_prompt
from copernicus.references.settings import GeneratorSettings
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = ["COMPLETION_TOKENS_PER_TEXT", "TokenEstimate", "estimate_tokens", "token_counter"]

COMPLETION_TOKENS_PER_TEXT = 120
TIKTOKEN_ENCODING = "o200k_base"
COUNTER_TIKTOKEN = f"tiktoken {TIKTOKEN_ENCODING}"
COUNTER_CHARACTERS = "characters / 4"
CHARACTERS_PER_TOKEN = 4
TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class TokenEstimate:
    """The tokens a generation run is expected to use, retries and surplus not counted.

    Attributes:
        prompt_tokens: Tokens of the system and user prompts of every call.
        completion_tokens: Tokens of the answers, assumed to be 120 per requested text.
        calls: Calls of the run: cells times writers times calls per cell.
        counter: How the prompt tokens were counted: ``"tiktoken o200k_base"`` or
            ``"characters / 4"``.
    """

    prompt_tokens: int
    completion_tokens: int
    calls: int
    counter: str

    def cost(self, price_input: float, price_output: float) -> float:
        """Return the estimated cost in US dollars.

        Args:
            price_input: Price of a million prompt tokens, in US dollars.
            price_output: Price of a million completion tokens, in US dollars.

        Returns:
            The cost of the estimated prompt and completion tokens.
        """
        return (
            self.prompt_tokens * price_input + self.completion_tokens * price_output
        ) / TOKENS_PER_MILLION


def _quarter_of_characters(text: str) -> int:
    """Count a text as a quarter of its characters, the usual rule of thumb for English."""
    return len(text) // CHARACTERS_PER_TOKEN


def token_counter(*, use_tiktoken: bool = True) -> tuple[Callable[[str], int], str]:
    """Return the function that counts the tokens of a prompt, and its name.

    tiktoken's ``o200k_base`` encoding counts when tiktoken is installed and the encoding
    can be loaded. Loading it the first time downloads it, so on a machine without network
    access (a cluster node running a vLLM writer, which installs tiktoken) the load fails;
    the estimate is then counted as characters / 4 with a warning, because it only guides
    the budget before a run and its counter is reported with it.

    Args:
        use_tiktoken: ``False`` counts characters / 4 without importing tiktoken, so the
            count never reaches the network, as for a run with a fake backend.

    Returns:
        The counting function and its name: ``"tiktoken o200k_base"`` or ``"characters / 4"``.
    """
    if not use_tiktoken:
        return _quarter_of_characters, COUNTER_CHARACTERS
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ImportError:
        return _quarter_of_characters, COUNTER_CHARACTERS
    try:
        encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    except (OSError, ValueError) as error:
        logger.warning(
            "the token estimate counts characters / 4: tiktoken's %s encoding could not be "
            "loaded (%s)",
            TIKTOKEN_ENCODING,
            error,
        )
        return _quarter_of_characters, COUNTER_CHARACTERS
    return (lambda text: len(encoding.encode(text))), COUNTER_TIKTOKEN


def estimate_tokens(
    rubric: Rubric, settings: GeneratorSettings, *, n_writers: int = 1
) -> TokenEstimate:
    """Estimate the tokens of a generation run before it starts.

    Every prompt the run sends on a first attempt is rendered and counted; the answers are
    assumed to take 120 tokens per requested text. Retries and the texts of surplus answers
    come on top.

    Args:
        rubric: The rubric whose cells get texts.
        settings: The generation settings.
        n_writers: Number of writer models.

    Returns:
        The estimate; its ``calls`` equal `copernicus.references.plan_calls`.

    Raises:
        ConfigurationError: If there is no writer.
    """
    if n_writers < 1:
        raise ConfigurationError("reference texts need at least one writer model")
    count, counter = token_counter()
    cells = build_cells(rubric)
    system = system_prompt(settings.style, settings.texts_per_call, settings.register_description)
    per_round = sum(
        count(system) + count(user_prompt(cell, rubric, settings.texts_per_call)) for cell in cells
    )
    repeats = n_writers * settings.calls_per_cell
    calls = len(cells) * repeats
    return TokenEstimate(
        prompt_tokens=per_round * repeats,
        completion_tokens=calls * settings.texts_per_call * COMPLETION_TOKENS_PER_TEXT,
        calls=calls,
        counter=counter,
    )
