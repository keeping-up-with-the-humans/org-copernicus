"""Reference texts for every pole of every target of a rubric.

A writer model is asked, through any backend, for short texts that show the positive or
the negative pole of the construct or of one attribute at three intensities. A leakage
screen flags the texts that quote the vocabulary of the rubric. Later steps embed the
texts and score real texts against them.
"""

from copernicus.references.cells import Cell, build_cells
from copernicus.references.estimate import TokenEstimate, estimate_tokens, token_counter
from copernicus.references.generator import DeferredWriter, generate_references, plan_calls
from copernicus.references.prompts import PROMPT_VERSION, system_prompt, user_prompt
from copernicus.references.reference_set import ReferenceSet, Shortfall
from copernicus.references.screen import leakage_terms, screen_report, screen_texts
from copernicus.references.settings import GeneratorSettings

__all__ = [
    "PROMPT_VERSION",
    "Cell",
    "DeferredWriter",
    "GeneratorSettings",
    "ReferenceSet",
    "Shortfall",
    "TokenEstimate",
    "build_cells",
    "estimate_tokens",
    "generate_references",
    "leakage_terms",
    "plan_calls",
    "screen_report",
    "screen_texts",
    "system_prompt",
    "token_counter",
    "user_prompt",
]
