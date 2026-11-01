"""The population hash: the fingerprint of a table's texts and groups that binds a draw to it.

``CorrectionData`` and the human-sample draw both compute it, so it lives below either module;
``copernicus.corrections.sampling`` exports it.
"""

from __future__ import annotations

import hashlib
import logging
import re

import numpy as np
import numpy.typing as npt

from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["POPULATION_HASH_PATTERN", "population_hash_of"]

POPULATION_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
"""The form of a population hash: 64 lowercase hexadecimal characters."""


def population_hash_of(text_id: npt.ArrayLike, group_id: npt.ArrayLike) -> str:
    """The fingerprint of a table's texts and their groups, which binds a draw to its table.

    It is the SHA-256 digest of the UTF-8 encoding of the lines ``"<text_id>\\t<group_id>\\n"``
    of every text, sorted, with ids and groups compared as strings. The order of the rows does
    not change it; an added, removed or regrouped text, or a renamed group, does.
    ``draw_groups`` records it for the pool it draws from, ``CorrectionData.population_hash``
    holds it for the data, and ``attach_draw`` compares the two.

    Args:
        text_id: Text id of every text.
        group_id: Group id of every text, aligned with ``text_id``.

    Returns:
        The digest as 64 lowercase hexadecimal characters.

    Raises:
        ConfigurationError: If the two are not one-dimensional or differ in length.
    """
    texts = np.asarray(text_id).astype(str)
    groups = np.asarray(group_id).astype(str)
    if texts.ndim != 1 or texts.shape != groups.shape:
        raise ConfigurationError("text_id and group_id must be one-dimensional and of equal length")
    lines = sorted(
        f"{text}\t{group}\n" for text, group in zip(texts.tolist(), groups.tolist(), strict=True)
    )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
