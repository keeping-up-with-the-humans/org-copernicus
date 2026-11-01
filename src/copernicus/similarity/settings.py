"""The configuration axes of the similarity coder."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

from copernicus.errors import SettingsError
from copernicus.similarity.reduce import REDUCTIONS

logger = logging.getLogger(__name__)

__all__ = ["LONG_TEXT", "POLARITIES", "TARGET_LEVELS", "WRITER_POOLINGS", "SimilaritySettings"]

POLARITIES = ("net", "positive", "negative")
"""The polarities: the positive pole minus the negative pole (``net``), or one pole alone."""
WRITER_POOLINGS = ("mean", "max", "pooled")
"""How the references of several writer models are pooled.

With ``mean`` or ``max`` each writer is reduced on its own and the writers are pooled by that
function; with ``pooled`` all references of a pole are reduced at once.
"""
LONG_TEXT = ("max_chunk", "mean_chunk", "pooled_embedding", "first_window")
"""How the chunks of a text longer than the window of the encoder enter the similarity."""
TARGET_LEVELS = ("construct", "attribute")
"""Whether the construct is scored directly or the attributes are scored and rolled up."""

ALLOWED = {
    "polarity": POLARITIES,
    "reduction": REDUCTIONS,
    "writer_pooling": WRITER_POOLINGS,
    "long_text": LONG_TEXT,
    "target_level": TARGET_LEVELS,
}


@dataclass(frozen=True)
class SimilaritySettings:
    """One configuration of the similarity coder.

    The defaults are the paper's configuration of the similarity coder. The hybrid coder
    has its own defaults, `copernicus.hybrid.HYBRID_DEFAULTS`.

    Attributes:
        polarity: ``"net"`` (positive pole minus negative pole), ``"positive"``, or
            ``"negative"``.
        reduction: How the cosines to the references of a pole are reduced to one score
            per text: ``"mean"``, ``"max"``, or ``"pca"``.
        writer_pooling: ``"mean"`` or ``"max"`` reduce the cosines to the references of
            each writer model on their own and pool the per-writer scores with that
            function; ``"pooled"`` reduces the cosines to all references of the pole at
            once, whoever wrote them.
        long_text: How the chunks of a long text enter the cosine: ``"max_chunk"``,
            ``"mean_chunk"`` (per reference), ``"pooled_embedding"`` (the vector of the
            whole text), or ``"first_window"`` (the first chunk only).
        target_level: ``"construct"`` scores the construct directly; ``"attribute"``
            scores the attributes and rolls them up into one construct score.
    """

    polarity: str = "net"
    reduction: str = "pca"
    writer_pooling: str = "mean"
    long_text: str = "max_chunk"
    target_level: str = "construct"

    def __post_init__(self) -> None:
        """Check every field against its allowed values.

        Raises:
            SettingsError: Naming the field and its value, if a value is not allowed.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if value not in ALLOWED[field.name]:
                raise SettingsError(
                    f"similarity setting {field.name} is {value!r}; use one of "
                    f"{ALLOWED[field.name]}"
                )

    def label(self) -> str:
        """Return the five values joined by hyphens, in the order of the fields."""
        return "-".join(getattr(self, field.name) for field in fields(self))
