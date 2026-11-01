"""The encoder: a Hugging Face model that turns texts into chunked, pooled vectors."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from copernicus._io import LOADED_COMMIT_KEY, UNKNOWN_REVISION
from copernicus.data.table import TextTable
from copernicus.embeddings.checkpoint import (
    checkpoint_file,
    fragments_fingerprint,
    read_checkpoint,
    write_checkpoint,
)
from copernicus.embeddings.chunking import split_into_token_chunks
from copernicus.embeddings.registry import DEFAULT_ENCODER, EncoderSpec, resolve_encoder
from copernicus.embeddings.sets import EmbeddingSet
from copernicus.errors import ConfigurationError, SettingsError

logger = logging.getLogger(__name__)

__all__ = ["CHECKPOINT_EVERY", "Encoder", "EncoderRuntime"]

EXTRA_HINT = (
    'install the embed extra: pip install "org-copernicus[embed]", '
    'or pip install ".[embed]" from the package\'s root folder'
)
NORM_FLOOR = 1e-12
CHECKPOINT_EVERY = 2048
MAX_IDS_IN_MESSAGE = 10


@dataclass(frozen=True)
class EncoderRuntime:
    """The loaded pieces an embedding pass works with.

    Attributes:
        torch: The ``torch`` module.
        tokenizer: The tokenizer, with the padding side of the specification applied.
        model: The model, on its device and in evaluation mode.
        pool: The pooling function of the specification.
        device: The ``torch.device`` the model and the batches live on.
        versions: ``resolved_revision`` (the commit the weights were loaded from, or
            ``"unknown"`` when the model does not name one), ``torch_version``, and
            ``transformers_version``; recorded in the sidecar of every embedding set.
            Texts and references whose sidecars record different commits are not scored
            together, and a checkpoint is resumed only under the commit it was written
            with.
    """

    torch: Any
    tokenizer: Any
    model: Any
    pool: Callable[[Any, Any], Any]
    device: Any
    versions: Mapping[str, str]


def _import_required(name: str) -> ModuleType:
    """Import a module of the ``embed`` extra.

    Raises:
        ConfigurationError: If the module is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ConfigurationError(f"{name} is not installed; {EXTRA_HINT}") from error


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length; an all-zero row stays zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms < NORM_FLOOR, 1.0, norms)


class Encoder:
    """Embeds texts with one encoder of the registry or any Hugging Face model.

    The model is loaded on first use, so building an encoder is cheap and does not need
    torch. Texts longer than the window are split into consecutive windows (chunks);
    every chunk is embedded and the chunk vectors are combined into one vector per text.
    """

    def __init__(
        self,
        spec: EncoderSpec | str = DEFAULT_ENCODER,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialise the encoder without loading the model.

        Args:
            spec: A specification, a registry name, or a Hugging Face id, which gets the
                default pooling and window of `resolve_encoder` with a warning; pass
                ``resolve_encoder(id, pooling=..., max_tokens=...)`` to choose them.
            device: Torch device name; chosen automatically (cuda, then mps, then cpu)
                when ``None``.
            batch_size: Chunks per forward pass; at least 1.

        Raises:
            SettingsError: If the batch size is below 1.
        """
        if batch_size < 1:
            raise SettingsError(f"batch_size must be at least 1, got {batch_size}")
        self.spec = resolve_encoder(spec)
        self.device = device
        self.batch_size = batch_size
        self._runtime: EncoderRuntime | None = None

    def _load(self) -> EncoderRuntime:
        """Import torch and transformers and load the tokenizer and the model once.

        A pinned revision of the specification is passed to both loaders. The commit the
        weights came from is read from the model's configuration.

        Returns:
            The loaded runtime.

        Raises:
            ConfigurationError: If torch or transformers is not installed, or the dtype
                of the specification is not a torch dtype.
        """
        if self._runtime is not None:
            return self._runtime
        torch = _import_required("torch")
        transformers = _import_required("transformers")
        pooling = importlib.import_module("copernicus.embeddings.pooling")
        spec = self.spec
        device = torch.device(self.device if self.device is not None else self._best_device(torch))
        pinned = {} if spec.revision is None else {"revision": spec.revision}
        options: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code, **pinned}
        if spec.dtype is not None:
            dtype = getattr(torch, spec.dtype, None)
            if not isinstance(dtype, torch.dtype):
                raise ConfigurationError(
                    f"encoder {spec.name!r}: {spec.dtype!r} is not a torch dtype"
                )
            options["dtype"] = dtype
        logger.info("loading encoder %s (%s) on %s", spec.name, spec.hf_id, device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code, **pinned
        )
        tokenizer.padding_side = spec.padding_side
        model = transformers.AutoModel.from_pretrained(spec.hf_id, **options).to(device).eval()
        commit = getattr(model.config, "_commit_hash", None)
        versions = {
            LOADED_COMMIT_KEY: commit if isinstance(commit, str) else UNKNOWN_REVISION,
            "torch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
        }
        self._runtime = EncoderRuntime(
            torch, tokenizer, model, pooling.POOLING_FUNCTIONS[spec.pooling], device, versions
        )
        return self._runtime

    @staticmethod
    def _best_device(torch: Any) -> str:
        """Return ``cuda`` if available, else ``mps`` if available, else ``cpu``."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def embed(
        self,
        texts: Sequence[str],
        ids: Sequence[Any],
        *,
        checkpoint: Path | None = None,
        checkpoint_every: int = CHECKPOINT_EVERY,
    ) -> EmbeddingSet:
        """Embed texts into per-text and per-chunk unit vectors.

        Every text is split into chunks of the encoder's window; the chunks are embedded
        in batches of similar length. The vector of a chunk is its pooled vector scaled
        to unit length; the vector of a text is the mean of the raw pooled chunk vectors,
        weighted by the token counts of the chunks, scaled to unit length once. For mean
        pooling this equals the mean over all tokens of the text.

        Args:
            texts: The texts to embed; at least one, none empty or whitespace only (such a
                text would be embedded as the special tokens alone).
            ids: One id per text; unique, stored as strings.
            checkpoint: Path without suffix. After every ``checkpoint_every`` finished
                chunks the finished chunks are written to ``<checkpoint>.npz``; a rerun
                of an interrupted pass with the same path and the same texts skips them,
                provided the model was loaded from the same commit. The file is deleted
                when the pass completes, so the path can be reused.
            checkpoint_every: Chunks between two checkpoint writes; at least 1.

        Returns:
            The embedding set, aligned with ``ids``.

        Raises:
            ConfigurationError: If there are no texts, ids and texts differ in number,
                ids repeat, a text is empty or whitespace only (the first ten such ids are
                named), or a checkpoint belongs to other texts, another encoder, or another
                commit of the model (the default branch of its repository moved).
            SettingsError: If ``checkpoint_every`` is below 1.
        """
        if len(texts) == 0:
            raise ConfigurationError("there are no texts to embed")
        if len(ids) != len(texts):
            raise ConfigurationError(f"{len(ids)} ids for {len(texts)} texts")
        if len(set(map(str, ids))) != len(ids):
            raise ConfigurationError("ids must be unique")
        blank = [str(text_id) for text_id, text in zip(ids, texts, strict=True) if not text.strip()]
        if blank:
            shown = ", ".join(repr(text_id) for text_id in blank[:MAX_IDS_IN_MESSAGE])
            rest = len(blank) - MAX_IDS_IN_MESSAGE
            raise ConfigurationError(
                f"{len(blank)} texts are empty or whitespace only and cannot be embedded: "
                f"{shown}" + (f" and {rest} more" if rest > 0 else "")
            )
        if checkpoint_every < 1:
            raise SettingsError(f"checkpoint_every must be at least 1, got {checkpoint_every}")
        runtime = self._load()
        fragments: list[str] = []
        owner: list[int] = []
        for index, text in enumerate(texts):
            pieces = split_into_token_chunks(
                text, runtime.tokenizer, self.spec.max_tokens, self.spec.special_tokens_reserved
            )
            fragments.extend(pieces)
            owner.extend([index] * len(pieces))
        logger.info("embedding %d texts as %d chunks", len(texts), len(fragments))
        raw, weight = self._pool_fragments(runtime, fragments, checkpoint, checkpoint_every)
        chunk_owner = np.asarray(owner, dtype=np.int64)
        total = np.zeros((len(texts), raw.shape[1]), dtype=np.float64)
        np.add.at(total, chunk_owner, raw * weight[:, None])
        text_weight = np.zeros(len(texts), dtype=np.float64)
        np.add.at(text_weight, chunk_owner, weight)
        if np.any(text_weight <= 0):
            raise ConfigurationError(
                f"{int((text_weight <= 0).sum())} texts have no attended token"
            )
        result = EmbeddingSet(
            ids=np.asarray([str(text_id) for text_id in ids]),
            vectors=_unit_rows(total / text_weight[:, None]).astype(np.float32),
            chunk_vectors=_unit_rows(raw).astype(np.float32),
            chunk_owner=chunk_owner,
            chunk_weight=weight,
            encoder=self.spec.name,
            metadata=None,
            sidecar={**self.spec.identity(), **runtime.versions},
        )
        if checkpoint is not None:
            checkpoint_file(checkpoint).unlink(missing_ok=True)
        return result

    def embed_table(self, table: TextTable, **options: Any) -> EmbeddingSet:
        """Embed the texts of a text table; its other columns become the metadata.

        Args:
            table: The text table.
            **options: Keyword arguments of `embed`.

        Returns:
            The embedding set, with the table's frame without the text column as metadata
            and the text ids as strings.
        """
        frame = table.frame
        result = self.embed(list(frame["text"]), list(table.ids()), **options)
        metadata = frame.drop(columns="text").reset_index(drop=True)
        metadata["text_id"] = metadata["text_id"].astype(str)
        return replace(result, metadata=metadata)

    def _pool_fragments(
        self,
        runtime: EncoderRuntime,
        fragments: list[str],
        checkpoint: Path | None,
        checkpoint_every: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw pooled vector and the token count of every fragment.

        Fragments are batched in order of length, so a batch is padded to its neighbours
        and not to the longest fragment overall. Finished fragments of an earlier run are
        read from the checkpoint and skipped.
        """
        n_fragments = len(fragments)
        fingerprint = fragments_fingerprint(
            self.spec, fragments, resolved_revision=runtime.versions[LOADED_COMMIT_KEY]
        )
        raw: np.ndarray | None = None
        weight = np.zeros(n_fragments, dtype=np.float64)
        finished = np.zeros(n_fragments, dtype=bool)
        saved = read_checkpoint(checkpoint, fingerprint) if checkpoint is not None else None
        if saved is not None:
            done, saved_raw, saved_weight = saved
            raw = np.zeros((n_fragments, saved_raw.shape[1]), dtype=np.float64)
            raw[done], weight[done], finished[done] = saved_raw, saved_weight, True
            logger.info("resuming: %d of %d chunks are finished", len(done), n_fragments)
        order = np.argsort([len(fragment) for fragment in fragments], kind="stable")
        pending = order[~finished[order]]
        since_write = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            pooled, counts = self._pool_batch(runtime, [fragments[i] for i in batch])
            if raw is None:
                raw = np.zeros((n_fragments, pooled.shape[1]), dtype=np.float64)
            raw[batch], weight[batch], finished[batch] = pooled, counts, True
            since_write += len(batch)
            unfinished = start + self.batch_size < len(pending)
            if checkpoint is not None and since_write >= checkpoint_every and unfinished:
                done = np.flatnonzero(finished)
                write_checkpoint(checkpoint, fingerprint, done, raw[done], weight[done])
                since_write = 0
        if raw is None:
            raise ConfigurationError("no chunk was embedded")
        return raw, weight

    def _pool_batch(
        self, runtime: EncoderRuntime, batch: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one forward pass; return the raw pooled vectors and the attended token counts."""
        torch = runtime.torch
        inputs = runtime.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.spec.max_tokens,
            return_tensors="pt",
        ).to(runtime.device)
        with torch.no_grad():
            hidden = runtime.model(**inputs).last_hidden_state
            pooled = runtime.pool(hidden, inputs["attention_mask"])
        counts = inputs["attention_mask"].sum(dim=1)
        return (
            pooled.cpu().to(torch.float64).numpy(),
            counts.cpu().to(torch.float64).numpy(),
        )
