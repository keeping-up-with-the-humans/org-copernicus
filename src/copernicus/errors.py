"""Exception hierarchy for the copernicus library.

Every failure the library raises on purpose derives from ``CopernicusError``
so that callers can catch library failures separately from programming errors.

Which of the two input errors to raise:

* ``SettingsError``: a setting is outside its documented range. A settings object refuses
  itself in its validator (``GeneratorSettings``, ``SimilaritySettings``, ``EncoderSpec``,
  ``CodingSettings``), and an entry point refuses an option argument the same way (a batch
  size below 1, an unknown granularity, a pooling that contradicts a registered encoder).
* ``ConfigurationError``: the data an operation receives are inconsistent, incomplete, or
  unknown to the library: embeddings of different encoders, a missing pole, a file that
  does not describe what it should, ids that differ between inputs. Building blocks that
  receive values a settings object has already checked (``reduce_matrix``,
  ``pool_writers``, ``per_reference_similarity``, ``ReferenceEmbeddings.pole_indices``)
  treat an unknown value as inconsistent data and raise ``ConfigurationError`` as well.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from copernicus.coders.runner_settings import CodingSummary

__all__ = [
    "BackendError",
    "ConfigurationError",
    "CopernicusError",
    "EstimatorError",
    "EvidenceContractError",
    "NotApplicableError",
    "PromptContractError",
    "RubricValidationError",
    "RunFailureCeilingError",
    "SettingsError",
    "StoreError",
    "TextTableError",
]


class CopernicusError(Exception):
    """Base class for all errors raised deliberately by the library."""


class RubricValidationError(CopernicusError):
    """A rubric file or object violates the rubric schema."""


class TextTableError(CopernicusError):
    """A text table is missing required columns or holds invalid values."""


class BackendError(CopernicusError):
    """An LLM backend could not produce a usable answer.

    Attributes:
        attempts: Number of attempts made before giving up.
        last_output: Raw text of the last model output, if any was received.
    """

    def __init__(self, message: str, attempts: int = 0, last_output: str | None = None) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the failure.
            attempts: Number of attempts made before giving up.
            last_output: Raw text of the last model output, if any.
        """
        super().__init__(message)
        self.attempts = attempts
        self.last_output = last_output


class ConfigurationError(CopernicusError):
    """Inputs of an operation are inconsistent, incomplete, or unknown to the library.

    Examples are embeddings made by different encoders, an unknown pooling, a missing
    cell, or a rubric that does not match the data it is applied to.
    """


class PromptContractError(CopernicusError):
    """A prompt template or its inputs violate the prompt contract."""


class EvidenceContractError(CopernicusError):
    """Evidence returned by a coder violates the evidence contract."""


class StoreError(CopernicusError):
    """The result store is missing, inconsistent, or cannot be written."""


class EstimatorError(CopernicusError):
    """An estimator could not be fitted or evaluated."""


class NotApplicableError(CopernicusError):
    """A method does not apply to the requested construct or data."""


class SettingsError(CopernicusError, ValueError):
    """A setting is outside its documented range or cannot be applied.

    It is also a ``ValueError``, so code that checks settings values the usual way
    keeps working.
    """


class RunFailureCeilingError(CopernicusError):
    """The share of failed items in a run exceeded the permitted ceiling.

    Attributes:
        summary: What the run did until it stopped; the rows it coded are stored.
    """

    def __init__(self, message: str, summary: CodingSummary) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the failure.
            summary: What the run did until it stopped.
        """
        super().__init__(message)
        self.summary = summary

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild the error from its message and summary, for pickling and copying."""
        return (type(self), (str(self), self.summary))
