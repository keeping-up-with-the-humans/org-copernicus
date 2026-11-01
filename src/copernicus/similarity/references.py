"""Reference embeddings: the vectors of the kept reference texts, grouped into poles."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
from copernicus.embeddings import Encoder
from copernicus.errors import ConfigurationError
from copernicus.references import ReferenceSet
from copernicus.references.binding import BOUND_WITH_KEY, RUBRIC_HASH_KEY, check_binding
from copernicus.references.cells import LEVELS, POLARITIES, TARGET_CONSTRUCT
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = ["ReferenceEmbeddings", "embed_references"]

LABEL_NAMES = ("target", "polarity", "level", "style", "writer_model")
UNIT_TOLERANCE = 1e-4
OWNER = "reference embeddings"
EMBED_AGAIN = (
    "embed the reference set again (embed_references, or copernicus embed references), which "
    "makes no writer call"
)


@dataclass(frozen=True, eq=False)
class ReferenceEmbeddings:
    """Vectors of reference texts, each labelled with the pole and cell it stands for.

    A pole is the set of references of one target and one polarity, over all levels,
    styles, and writer models.

    The references are bound to the content of their rubric: the sidecar holds the
    ``rubric_hash`` of the reference set they were embedded from and its ``bound_with``,
    and `assert_complete`, which every scoring path calls, refuses a rubric whose content
    hash differs.

    The five label arrays are turned into one-dimensional string arrays on construction;
    lists and object arrays of strings are accepted.

    Attributes:
        vectors: Unit-length vector per reference, float32, shape ``[K, D]``.
        target: ``"construct"`` or an attribute id per reference, shape ``[K]``.
        polarity: ``"positive"`` or ``"negative"`` per reference.
        level: ``"top"``, ``"mid"``, or ``"low"`` per reference.
        style: Style of the generation settings per reference.
        writer_model: Name of the writer model that wrote each reference.
        rubric_id: Identifier of the rubric the references were written for.
        encoder: Name of the encoder that made the vectors.
        sidecar: The description of the encoder and of the file: the identity keys and,
            when known, the version keys of the embedding set (see
            `copernicus.embeddings.EmbeddingSet`), ``rubric_id``, ``rubric_hash`` (the
            content hash of the rubric the references were written for; required),
            ``bound_with`` (how the set came by that hash, ``"generated"`` or
            ``"for_rubric"``; required),
            ``n_references``, ``prompt_versions`` when known, and, once loaded from disk,
            the library version, the creation time and ``arrays_hash``, the digest of the
            arrays of the file.
    """

    vectors: np.ndarray
    target: np.ndarray
    polarity: np.ndarray
    level: np.ndarray
    style: np.ndarray
    writer_model: np.ndarray
    rubric_id: str
    encoder: str
    sidecar: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Turn the labels into strings and check that the arrays describe the same references.

        Raises:
            ConfigurationError: If a label array is not one-dimensional strings or
                integers, the vectors are not a two-dimensional float32 array of unit-length
                rows, a label array differs in length from the vectors, a polarity or level
                is unknown, there is no reference, or the sidecar has no ``rubric_hash`` that
                is the content hash of a rubric or no ``bound_with`` that is
                ``"generated"`` or ``"for_rubric"``.
        """
        check_binding(self.sidecar, "the sidecar of the reference embeddings")
        for name in LABEL_NAMES:
            object.__setattr__(self, name, as_unicode_1d(getattr(self, name), name))
        if self.vectors.ndim != 2:
            raise ConfigurationError("reference vectors must be a two-dimensional array")
        if self.vectors.dtype != np.float32:
            raise ConfigurationError(f"reference vectors must be float32, got {self.vectors.dtype}")
        n_references = self.vectors.shape[0]
        if n_references == 0:
            raise ConfigurationError("reference embeddings hold no references")
        problems = [
            f"{len(getattr(self, name))} {name} entries for {n_references} vectors"
            for name in LABEL_NAMES
            if len(getattr(self, name)) != n_references
        ]
        if problems:
            raise ConfigurationError("inconsistent reference embeddings: " + "; ".join(problems))
        norms = np.linalg.norm(self.vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=UNIT_TOLERANCE):
            raise ConfigurationError("reference vectors must have unit length")
        for name, allowed in (("polarity", POLARITIES), ("level", LEVELS)):
            unknown = sorted(set(getattr(self, name).tolist()) - set(allowed))
            if unknown:
                raise ConfigurationError(f"unknown {name} {unknown} in reference embeddings")

    def __len__(self) -> int:
        """Return the number of references."""
        return self.vectors.shape[0]

    @property
    def rubric_hash(self) -> str:
        """Return the content hash of the rubric the references were written for."""
        return str(self.sidecar[RUBRIC_HASH_KEY])

    @property
    def bound_with(self) -> str:
        """Return how the references came by their rubric hash: generated or for_rubric."""
        return str(self.sidecar[BOUND_WITH_KEY])

    def targets(self) -> list[str]:
        """Return the sorted distinct targets: ``"construct"`` and attribute ids."""
        return sorted(set(self.target.tolist()))

    def writers(self) -> list[str]:
        """Return the sorted distinct writer models."""
        return sorted(set(self.writer_model.tolist()))

    def pole_indices(self, target: str, polarity: str) -> np.ndarray:
        """Return the rows of the references of one pole.

        Args:
            target: ``"construct"`` or an attribute id present in these references.
            polarity: ``"positive"`` or ``"negative"``.

        Returns:
            Row indices into ``vectors``, over all levels, styles, and writer models.

        Raises:
            ConfigurationError: If the target is not among the references or the
                polarity is unknown.
        """
        if target not in self.targets():
            raise ConfigurationError(f"unknown target {target!r}; known: {self.targets()}")
        if polarity not in POLARITIES:
            raise ConfigurationError(f"unknown polarity {polarity!r}; use one of {POLARITIES}")
        return np.flatnonzero((self.target == target) & (self.polarity == polarity))

    def subset_writers(self, writers: Sequence[str]) -> ReferenceEmbeddings:
        """Keep the references of the given writer models, in their original order.

        Args:
            writers: Names of the writer models to keep.

        Returns:
            New reference embeddings, with the sidecar of these (its ``rubric_hash``
            included, its ``arrays_hash`` left out, since `save` records the digest of the
            arrays it writes) and the new number of references.

        Raises:
            ConfigurationError: If a writer model has no references here.
        """
        known = self.writers()
        for name in writers:
            if name not in known:
                raise ConfigurationError(f"unknown writer model {name!r}; known: {known}")
        keep = np.isin(self.writer_model, list(writers))
        return replace(
            self,
            vectors=self.vectors[keep],
            target=self.target[keep],
            polarity=self.polarity[keep],
            level=self.level[keep],
            style=self.style[keep],
            writer_model=self.writer_model[keep],
            sidecar={**without_arrays_hash(self.sidecar), "n_references": int(keep.sum())},
        )

    def assert_complete(
        self, rubric: Rubric, targets: Sequence[str] | None = None, *, label: str | None = None
    ) -> None:
        """Check that the references belong to the rubric as it reads now and hold the poles.

        The references belong to the rubric when it has their rubric id and their content
        hash: a rubric edited under the same ids has another hash, and its references
        describe the earlier definition. Every scoring path calls this check before it
        scores.

        Args:
            rubric: The rubric the references will be used with.
            targets: The targets whose two poles must be present (``"construct"`` or
                attribute ids); ``None`` for the construct and every attribute of the rubric.
            label: Names these references at the start of the message, for example the key
                of a grid input; ``None`` names none.

        Raises:
            ConfigurationError: If the references were written for another rubric, or for
                another content of this rubric (naming both hashes), or naming each target
                and polarity that has no reference.
        """
        problem = self._mismatch(rubric, targets)
        if problem is not None:
            raise ConfigurationError(problem if label is None else f"{label}: {problem}")

    def _mismatch(self, rubric: Rubric, targets: Sequence[str] | None) -> str | None:
        """Say why the references do not fit the rubric or the targets, or return ``None``."""
        if self.rubric_id != rubric.rubric_id:
            return (
                f"reference embeddings are for rubric {self.rubric_id}, "
                f"not for rubric {rubric.rubric_id}"
            )
        current = rubric.content_hash()
        if self.rubric_hash != current:
            return (
                f"the reference embeddings were generated for rubric {rubric.rubric_id} with "
                f"hash {self.rubric_hash}, but the rubric now has hash {current}: the "
                "references describe the earlier definition; regenerate them (or rebuild a "
                "hand-written set with ReferenceSet.for_rubric)"
            )
        attribute_ids = [attribute.attribute_id for attribute in rubric.attributes]
        wanted = [TARGET_CONSTRUCT, *attribute_ids] if targets is None else list(targets)
        lacking = [
            f"the {polarity} pole of target {target}"
            for target in wanted
            for polarity in POLARITIES
            if not np.any((self.target == target) & (self.polarity == polarity))
        ]
        if lacking:
            return f"reference embeddings for rubric {rubric.rubric_id} lack " + "; ".join(lacking)
        return None

    def save(self, stem: str | Path) -> tuple[Path, Path]:
        """Write the arrays to ``<stem>.npz`` and the description to ``<stem>.json``.

        The description is written last and records ``arrays_hash``, the digest of the arrays.

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
            rubric_id=self.rubric_id,
            rubric_hash=self.rubric_hash,
            bound_with=self.bound_with,
            n_references=len(self),
        )
        if "prompt_versions" in self.sidecar:
            description["prompt_versions"] = list(self.sidecar["prompt_versions"])
        description.update(
            library_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        arrays = {"vectors": self.vectors} | {name: getattr(self, name) for name in LABEL_NAMES}
        description[ARRAYS_HASH_KEY] = arrays_hash(arrays)
        write_atomically(array_path, lambda handle: np.savez(handle, **arrays))
        write_json(sidecar_path, description)
        logger.info("wrote %d reference embeddings to %s", len(self), array_path)
        return array_path, sidecar_path

    @classmethod
    def load(cls, stem: str | Path, *, expect_encoder: str | None = None) -> ReferenceEmbeddings:
        """Read reference embeddings written by `save`.

        Args:
            stem: Path without suffix.
            expect_encoder: Name of the encoder the caller needs; ``None`` accepts any.

        Returns:
            The reference embeddings, with the whole description in ``sidecar``.

        Raises:
            ConfigurationError: If a file is missing, the description is not a JSON object,
                an array or a field of the description is absent or inconsistent, the
                description has no ``rubric_hash``, no ``bound_with`` or no ``arrays_hash``
                (it was written by an earlier version: embed the reference set again) or a
                value that is not a content hash, a binding or a digest, the references were
                made with another encoder than expected, or the arrays are not the ones the
                description names, as a save stopped between its two writes leaves.
        """
        array_path, sidecar_path = stem_paths(stem)
        for path in (array_path, sidecar_path):
            if not path.is_file():
                raise ConfigurationError(f"reference embeddings {stem}: {path.name} does not exist")
        description = read_sidecar(sidecar_path)
        absent = [key for key in ("encoder", "rubric_id", "n_references") if key not in description]
        if absent:
            raise ConfigurationError(
                f"reference embeddings {stem}: {sidecar_path.name} lacks {absent}"
            )
        for key in (RUBRIC_HASH_KEY, BOUND_WITH_KEY):
            if key not in description:
                raise ConfigurationError(
                    f"reference embeddings {stem}: {sidecar_path.name} has no {key}: it was "
                    f"written by an earlier version of copernicus than this one ({__version__}); "
                    f"{EMBED_AGAIN}"
                )
        check_binding(description, f"reference embeddings {stem}: {sidecar_path.name}")
        made_with = description["encoder"]
        if expect_encoder is not None and made_with != expect_encoder:
            raise ConfigurationError(
                f"reference embeddings {stem} were made with encoder {made_with}, "
                f"expected {expect_encoder}"
            )
        with np.load(array_path, allow_pickle=False) as arrays:
            names = ("vectors", *LABEL_NAMES)
            lacking = [name for name in names if name not in arrays.files]
            if lacking:
                raise ConfigurationError(
                    f"reference embeddings {stem}: {array_path.name} lacks {lacking}"
                )
            loaded = {name: arrays[name] for name in names}
        if description["n_references"] != len(loaded["vectors"]):
            raise ConfigurationError(
                f"reference embeddings {stem}: the description counts "
                f"{description['n_references']} references, the arrays hold "
                f"{len(loaded['vectors'])}"
            )
        check_arrays_hash(description, loaded, stem, OWNER, EMBED_AGAIN)
        return cls(
            rubric_id=description["rubric_id"], encoder=made_with, sidecar=description, **loaded
        )


def embed_references(refset: ReferenceSet, encoder: Encoder) -> ReferenceEmbeddings:
    """Embed the kept reference texts of a reference set.

    Each reference is expected to fit into one window of the encoder. A reference that
    does not is represented by its text vector, the token-weighted mean of its chunks.

    Args:
        refset: The reference texts; the ones the leakage screen excluded are skipped.
        encoder: The encoder that makes the vectors.

    Returns:
        The reference embeddings, in the order of the kept texts of the set, with the
        ``rubric_hash`` and the ``bound_with`` of the set in their sidecar.

    Raises:
        ConfigurationError: If the set has no kept text.
    """
    frame = refset.kept()
    if len(frame) == 0:
        raise ConfigurationError(
            f"the reference set for rubric {refset.rubric_id} has no reference texts"
        )
    texts = frame["text"].astype(str).tolist()
    embedded = encoder.embed(texts, [str(position) for position in range(len(texts))])
    multi_chunk = int(np.sum(np.bincount(embedded.chunk_owner, minlength=len(texts)) > 1))
    if multi_chunk:
        logger.warning(
            "%d reference texts are longer than one window of encoder %s; using their text vectors",
            multi_chunk,
            encoder.spec.name,
        )
    sidecar: dict[str, Any] = encoder_identity(embedded.sidecar, "embedding set")
    sidecar.update(encoder_versions(embedded.sidecar))
    sidecar.update(
        rubric_id=refset.rubric_id,
        rubric_hash=refset.rubric_hash,
        bound_with=refset.bound_with,
        n_references=len(texts),
        prompt_versions=sorted(set(frame["prompt_version"].astype(str))),
    )
    labels = {name: np.asarray(frame[name].astype(str).tolist(), dtype=str) for name in LABEL_NAMES}
    return ReferenceEmbeddings(
        vectors=embedded.vectors,
        rubric_id=refset.rubric_id,
        encoder=encoder.spec.name,
        sidecar=sidecar,
        **labels,
    )
