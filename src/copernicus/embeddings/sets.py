"""The container of the embeddings of a set of texts, with its files on disk."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copernicus._digests import ARRAYS_HASH_KEY, arrays_hash, check_arrays_hash, without_arrays_hash
from copernicus._io import (
    as_unicode_1d,
    encoder_identity,
    encoder_versions,
    read_sidecar,
    stem_paths,
    write_atomically,
    write_json,
)
from copernicus._version import __version__
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["EmbeddingSet"]

ARRAY_NAMES = ("ids", "vectors", "chunk_vectors", "chunk_owner", "chunk_weight")
MAX_IDS_IN_MESSAGE = 10
OWNER = "embedding set"
EMBED_AGAIN = "embed the texts again (Encoder.embed, or copernicus embed texts)"


@dataclass(frozen=True, eq=False)
class EmbeddingSet:
    """Vectors of a set of texts: one per text and one per chunk of a text.

    Texts longer than the window of the encoder are split into chunks; the vector of a
    text combines the raw vectors of its chunks, weighted by their token counts, and is
    normalised to unit length once. The vector of a chunk is normalised on its own.

    Attributes:
        ids: Text ids as strings, shape ``[N]``; unique. Integer ids, and lists or object
            arrays of strings or integers, are turned into a string array on construction.
        vectors: Unit-length vector per text, float32, shape ``[N, D]``, in id order.
        chunk_vectors: Unit-length vector per chunk, float32, shape ``[M, D]``; the
            chunks of a text keep their order in the text.
        chunk_owner: Index into ``ids`` of the text each chunk belongs to, shape ``[M]``.
        chunk_weight: Token count of each chunk, shape ``[M]``.
        encoder: Name of the encoder that made the vectors.
        metadata: One row per text, aligned with ``ids``; ``None`` when there is none. It is
            saved as JSON records, so a round trip through the files keeps strings, numbers
            and booleans but not every pandas dtype: datetimes come back as epoch
            milliseconds and nullable integers with a missing value as floats. Read dated
            or typed covariates from the text table, not from an embedding set.
        sidecar: The description of the encoder and of the file: the identity keys
            ``hf_id``, ``revision``, ``pooling``, ``max_tokens``, ``dtype``,
            ``trust_remote_code``, ``padding_side``, and ``special_tokens_reserved``; the
            version keys ``resolved_revision``, ``torch_version``, and
            ``transformers_version`` when an encoder made the set; and, once loaded from
            disk, the counts, the library version, the creation time, and ``arrays_hash``,
            the digest of the arrays of the file.
    """

    ids: np.ndarray
    vectors: np.ndarray
    chunk_vectors: np.ndarray
    chunk_owner: np.ndarray
    chunk_weight: np.ndarray
    encoder: str
    metadata: pd.DataFrame | None
    sidecar: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Turn the ids into strings and check that the arrays describe the same texts.

        Raises:
            ConfigurationError: If the ids are not one-dimensional strings or integers,
                shapes disagree, ids repeat, a text has no chunk, or a chunk names a text
                that does not exist.
        """
        object.__setattr__(self, "ids", as_unicode_1d(self.ids, "ids"))
        n_texts, n_chunks = len(self.ids), len(self.chunk_owner)
        problems = []
        if self.vectors.shape[0] != n_texts:
            problems.append(f"{self.vectors.shape[0]} vectors for {n_texts} ids")
        if self.chunk_vectors.shape[0] != n_chunks or len(self.chunk_weight) != n_chunks:
            problems.append("chunk vectors, owners and weights differ in length")
        if self.chunk_vectors.shape[1:] != self.vectors.shape[1:]:
            problems.append("chunk vectors and text vectors differ in dimension")
        if len(set(self.ids.tolist())) != n_texts:
            problems.append("ids are not unique")
        if self.metadata is not None and len(self.metadata) != n_texts:
            problems.append(f"{len(self.metadata)} metadata rows for {n_texts} ids")
        if not problems and not np.array_equal(np.unique(self.chunk_owner), np.arange(n_texts)):
            problems.append("every text needs at least one chunk, and every chunk a text")
        if problems:
            raise ConfigurationError("inconsistent embedding set: " + "; ".join(problems))

    def __len__(self) -> int:
        """Return the number of texts."""
        return len(self.ids)

    def save(self, stem: str | Path) -> tuple[Path, Path]:
        """Write the arrays to ``<stem>.npz`` and the description to ``<stem>.json``.

        The description holds the encoder name, the identity keys and the known version
        keys of the sidecar, the numbers of texts and chunks, the library version, the
        creation time in UTC, ``arrays_hash``, the digest of the arrays in the order they
        are written, and the metadata as a list of records when there is any. It is written
        last, and `load` refuses arrays that it does not name.

        Args:
            stem: Path without suffix; the parent folder must exist.

        Returns:
            The path of the array file and the path of the description.

        Raises:
            ConfigurationError: If the sidecar lacks one of the identity keys.
        """
        description: dict[str, Any] = {"encoder": self.encoder}
        description.update(encoder_identity(self.sidecar, OWNER))
        description.update(encoder_versions(self.sidecar))
        array_path, sidecar_path = stem_paths(stem)
        description.update(
            n_texts=len(self),
            n_chunks=len(self.chunk_owner),
            library_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        arrays = {name: getattr(self, name) for name in ARRAY_NAMES}
        description[ARRAYS_HASH_KEY] = arrays_hash(arrays)
        if self.metadata is not None:
            description["metadata"] = json.loads(self.metadata.to_json(orient="records"))
        write_atomically(array_path, lambda handle: np.savez(handle, **arrays))
        write_json(sidecar_path, description)
        logger.info(
            "wrote %d texts and %d chunks to %s", len(self), len(self.chunk_owner), array_path
        )
        return array_path, sidecar_path

    @classmethod
    def load(cls, stem: str | Path, *, expect_encoder: str | None = None) -> EmbeddingSet:
        """Read an embedding set written by `save`.

        Args:
            stem: Path without suffix.
            expect_encoder: Name of the encoder the caller needs; ``None`` accepts any.

        Returns:
            The embedding set, with the whole description in ``sidecar``.

        Raises:
            ConfigurationError: If a file is missing, the description is not a JSON object,
                the arrays or the description are inconsistent, the set was made with another
                encoder than expected, the description records no ``arrays_hash`` (it was
                written by an earlier version: embed the texts again) or a value that is not
                a digest, or the arrays are not the ones the description names, as a save
                stopped between its two writes leaves.
        """
        array_path, sidecar_path = stem_paths(stem)
        for path in (array_path, sidecar_path):
            if not path.is_file():
                raise ConfigurationError(f"embedding set {stem}: {path.name} does not exist")
        description = read_sidecar(sidecar_path)
        absent = [key for key in ("encoder", "n_texts", "n_chunks") if key not in description]
        if absent:
            raise ConfigurationError(f"embedding set {stem}: {sidecar_path.name} lacks {absent}")
        made_with = description["encoder"]
        if expect_encoder is not None and made_with != expect_encoder:
            raise ConfigurationError(
                f"embedding set {stem} was made with encoder {made_with}, expected {expect_encoder}"
            )
        with np.load(array_path, allow_pickle=False) as arrays:
            missing = [name for name in ARRAY_NAMES if name not in arrays.files]
            if missing:
                raise ConfigurationError(f"embedding set {stem}: {array_path.name} lacks {missing}")
            loaded = {name: arrays[name] for name in ARRAY_NAMES}
        if description["n_texts"] != len(loaded["ids"]):
            raise ConfigurationError(
                f"embedding set {stem}: the description counts {description['n_texts']} texts, "
                f"the arrays hold {len(loaded['ids'])}"
            )
        if description["n_chunks"] != len(loaded["chunk_owner"]):
            raise ConfigurationError(
                f"embedding set {stem}: the description counts {description['n_chunks']} chunks, "
                f"the arrays hold {len(loaded['chunk_owner'])}"
            )
        check_arrays_hash(description, loaded, stem, OWNER, EMBED_AGAIN)
        records = description.get("metadata")
        metadata = pd.DataFrame.from_records(records) if records is not None else None
        return cls(encoder=made_with, metadata=metadata, sidecar=description, **loaded)

    def subset(self, ids: Sequence[Any]) -> EmbeddingSet:
        """Keep the given texts, in the given order, together with their chunks.

        Args:
            ids: Ids of the texts to keep; compared as strings.

        Returns:
            A new set whose ``chunk_owner`` indexes the kept texts; its sidecar leaves out
            ``arrays_hash``, since `save` records the digest of the arrays it writes.

        Raises:
            ConfigurationError: If an id is unknown or given twice.
        """
        wanted = [str(text_id) for text_id in ids]
        position = {text_id: row for row, text_id in enumerate(self.ids.tolist())}
        unknown = [text_id for text_id in wanted if text_id not in position]
        if unknown:
            shown = ", ".join(repr(text_id) for text_id in unknown[:MAX_IDS_IN_MESSAGE])
            raise ConfigurationError(f"{len(unknown)} ids are not in the embedding set: {shown}")
        if len(set(wanted)) != len(wanted):
            raise ConfigurationError("subset ids must be unique")
        rows = np.asarray([position[text_id] for text_id in wanted], dtype=np.int64)
        new_row = np.full(len(self), -1, dtype=np.int64)
        new_row[rows] = np.arange(len(rows))
        kept = np.flatnonzero(new_row[self.chunk_owner] >= 0)
        kept = kept[np.argsort(new_row[self.chunk_owner[kept]], kind="stable")]
        sidecar = {**without_arrays_hash(self.sidecar), "n_texts": len(rows), "n_chunks": len(kept)}
        return replace(
            self,
            ids=self.ids[rows],
            vectors=self.vectors[rows],
            chunk_vectors=self.chunk_vectors[kept],
            chunk_owner=new_row[self.chunk_owner[kept]],
            chunk_weight=self.chunk_weight[kept],
            metadata=None
            if self.metadata is None
            else self.metadata.iloc[rows].reset_index(drop=True),
            sidecar=sidecar,
        )

    def first_chunk_vectors(self) -> np.ndarray:
        """Return the vector of the first chunk of each text.

        Returns:
            Array of shape ``[N, D]`` in text order: the encoder's reading of the
            beginning of each text, that is of the first window.
        """
        _, first = np.unique(self.chunk_owner, return_index=True)
        return self.chunk_vectors[first]
