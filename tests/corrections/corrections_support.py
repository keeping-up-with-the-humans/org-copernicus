"""Synthetic correction data shared by the tests of the correction families."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from copernicus.corrections.data import CorrectionData
from copernicus.embeddings.sets import EmbeddingSet

N_GROUPS = 20
GROUP_SIZE = 6
LABELLED_EVERY = 3
N_ROWS = 12
NOISE_STREAM = 1


def make_correction_data(
    *,
    n_groups: int = N_GROUPS,
    group_size: int = GROUP_SIZE,
    seed: int = 7,
    gold_kind: str = "continuous",
    label_all: bool = False,
) -> CorrectionData:
    """Texts in groups with two score columns, a covariate and gold on a subset.

    ``score_good`` follows the latent value closely, ``score_poor`` is noise. Gold is
    ``"continuous"`` (the latent value), ``"binary"`` (its median split) or ``"ordinal"``
    (five levels). Two texts out of three carry a gold value unless ``label_all``.

    Args:
        n_groups: Number of groups.
        group_size: Texts per group.
        seed: Seed of the generator.
        gold_kind: One of ``"continuous"``, ``"binary"``, ``"ordinal"``.
        label_all: Whether every text has a gold value.

    Returns:
        The correction data.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    latent = rng.normal(size=n)
    if gold_kind == "binary":
        gold = (latent > np.median(latent)).astype(float)
    elif gold_kind == "ordinal":
        gold = np.digitize(latent, np.quantile(latent, [0.2, 0.4, 0.6, 0.8])).astype(float) + 1.0
    else:
        gold = latent.copy()
    if not label_all:
        gold[np.arange(n) % LABELLED_EVERY == 0] = np.nan
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(n)],
            "group_id": [f"g{index // group_size:03d}" for index in range(n)],
            "score_good": 0.5 * latent + rng.normal(scale=0.3, size=n) + 3.0,
            "score_poor": rng.normal(size=n),
            "gold": gold,
            "log_chars": rng.normal(loc=5.0, scale=0.4, size=n),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))


def planted_data(
    gold_of: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    n_groups: int = N_GROUPS,
    group_size: int = GROUP_SIZE,
    seed: int = 11,
) -> CorrectionData:
    """Data whose gold is a function ``gold_of(score, covariate)`` of one noise-free score.

    Every text is labelled, and a second score column is pure noise.

    Args:
        gold_of: A callable taking the score and covariate arrays and returning gold values.
        n_groups: Number of groups.
        group_size: Texts per group.
        seed: Seed of the generator.

    Returns:
        The correction data.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    score = rng.uniform(0.0, 4.0, size=n)
    covariate = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(n)],
            "group_id": [f"g{index // group_size:03d}" for index in range(n)],
            "score_good": score,
            "score_poor": rng.normal(size=n),
            "gold": gold_of(score, covariate),
            "log_chars": covariate,
        }
    )
    return CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))


def make_frame(n: int = N_ROWS) -> pd.DataFrame:
    """Twelve texts in four groups with two score columns, partial gold and a covariate."""
    rng = np.random.default_rng(3)
    truth = np.linspace(1.0, 5.0, n)
    gold = truth.copy()
    gold[::3] = np.nan
    return pd.DataFrame(
        {
            "text_id": [f"t{i:02d}" for i in range(n)],
            "group_id": [f"g{i // 3}" for i in range(n)],
            "good": truth + rng.normal(scale=0.1, size=n),
            "poor": rng.normal(size=n),
            "gold": gold,
            "log_chars": rng.normal(size=n),
        }
    )


def make_data(frame: pd.DataFrame | None = None) -> CorrectionData:
    return CorrectionData(
        frame=make_frame() if frame is None else frame, score_columns=("good", "poor")
    )


def make_embedding_set(ids: Sequence[object], vectors: np.ndarray, encoder: str) -> EmbeddingSet:
    """An embedding set with one chunk per text and float32 vectors, as an encoder writes them."""
    n = len(ids)
    return EmbeddingSet(
        ids=np.asarray(ids),
        vectors=vectors.astype(np.float32),
        chunk_vectors=vectors.astype(np.float32),
        chunk_owner=np.arange(n, dtype=np.int64),
        chunk_weight=np.ones(n, dtype=np.int64),
        encoder=encoder,
        metadata=None,
        sidecar={},
    )


def embedding_data(
    gold: np.ndarray,
    score: np.ndarray,
    vectors: Mapping[str, np.ndarray],
    *,
    group_size: int = 5,
    seed: int = 0,
    second_score: np.ndarray | None = None,
) -> CorrectionData:
    """Texts in groups with gold, two score columns and embedding sets in the given order.

    ``score_good`` is ``score``; ``score_poor`` is ``second_score`` or, by default, noise.
    Every text is labelled where ``gold`` is finite.

    Args:
        gold: Gold value of each text (NaN where unlabelled).
        score: The informative score column.
        vectors: Embedding matrix per encoder name, rows in text order.
        group_size: Texts per group.
        seed: Seed of the noise column and the covariate. They come from a stream of their
            own, so a caller that drew its signal from ``default_rng(seed)`` does not get
            its signal back as the noise column.
        second_score: The second score column, if not noise.

    Returns:
        The correction data with the embedding sets attached.
    """
    rng = np.random.default_rng([seed, NOISE_STREAM])
    n = gold.size
    ids = [f"t{index:04d}" for index in range(n)]
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": [f"g{index // group_size:03d}" for index in range(n)],
            "score_good": score,
            "score_poor": rng.normal(size=n) if second_score is None else second_score,
            "gold": gold,
            "log_chars": rng.normal(loc=5.0, scale=0.4, size=n),
        }
    )
    sets = {name: make_embedding_set(ids, matrix, name) for name, matrix in vectors.items()}
    return CorrectionData(frame=frame, score_columns=("score_good", "score_poor"), embeddings=sets)


BATTERY_SCORES = ("score_a", "score_b", "score_c")


def battery_frame(
    *, n_groups: int = 150, group_size: int = 4, seed: int = 5, label_all: bool = True
) -> pd.DataFrame:
    """Texts in groups with three score columns of differing quality, gold and a covariate.

    The gold value is a latent value with a group effect. ``score_a``, ``score_b`` and
    ``score_c`` measure it with independent noise of rising size (``score_b`` on another
    scale), so a learner that combines them ranks the texts better than the best one alone.
    Every text carries gold unless ``label_all`` is False; then every third text has none.

    Args:
        n_groups: Number of groups.
        group_size: Texts per group.
        seed: Seed of the generator.
        label_all: Whether every text has a gold value.

    Returns:
        A frame with ``text_id``, ``group_id``, the three scores, ``gold`` and ``log_chars``.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    groups = np.arange(n) // group_size
    latent = rng.normal(scale=0.5, size=n_groups)[groups] + rng.normal(size=n)
    gold = latent.copy()
    if not label_all:
        gold[np.arange(n) % LABELLED_EVERY == 0] = np.nan
    return pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(n)],
            "group_id": [f"g{group:03d}" for group in groups],
            "score_a": latent + rng.normal(scale=0.7, size=n),
            "score_b": 2.0 * latent + rng.normal(scale=1.7, size=n) + 10.0,
            "score_c": latent + rng.normal(scale=1.0, size=n),
            "gold": gold,
            "log_chars": rng.normal(loc=5.0, scale=0.4, size=n),
        }
    )


def battery_data(frame: pd.DataFrame | None = None) -> CorrectionData:
    """Correction data over ``battery_frame`` (or the frame given) with its three scores."""
    return CorrectionData(
        frame=battery_frame() if frame is None else frame, score_columns=BATTERY_SCORES
    )
