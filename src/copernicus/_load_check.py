"""The check that texts and reference texts were embedded by one load of one encoder.

Texts are scored against reference embeddings only when their records do not show two loads
of the encoder: the same encoder settings and, where both sides record one, the same commit
of the model (`copernicus._io.check_same_encoder`), and transformers versions of one major
line where both sides record one; the line decides how the model's code is loaded. Two
releases of one major version gave the same vectors up to rounding in a comparison of two such
loads, while a rule that refused every difference would refuse every pair of sets embedded on
two machines. Such a difference, any difference of torch, and a side that records no version
are therefore scored, with a warning that the provenance of the scoring records under
``load_comparison``, so that a stored result says which versions were compared, or that a
comparison was not possible.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from packaging.version import Version

from copernicus._io import TORCH_VERSION_KEY, TRANSFORMERS_VERSION_KEY, check_same_encoder
from copernicus._versions import major_line, read_version
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["LOAD_COMPARISON_KEY", "UNKNOWN_VERSION", "compare_loads"]

LOAD_COMPARISON_KEY = "load_comparison"
UNKNOWN_VERSION = "unknown"
TEXTS = "texts"
REFERENCES = "references"
# The side of a comparison, and the name a message gives it.
OWNERS = {TEXTS: "texts", REFERENCES: "reference embeddings"}
# The version keys compared, and the package each one names.
PACKAGES = {TRANSFORMERS_VERSION_KEY: "transformers", TORCH_VERSION_KEY: "torch"}
ONE_ENVIRONMENT = "embed both sides in one environment"
RECORDED = "the provenance of the scoring records"


@dataclass(frozen=True)
class _Side:
    """The versions that one side records: as written, and as read for those it records.

    Attributes:
        recorded: Each version as the sidecar records it, ``"unknown"`` for one it does not.
        read: Each recorded version as the package parser reads it.
    """

    recorded: dict[str, str]
    read: dict[str, Version]


def _read_side(sidecar: Mapping[str, Any], owner: str, model: str) -> _Side:
    """Read every version that one side records, before anything of the two sides is compared.

    A key that is absent, null, or ``"unknown"`` is not recorded.

    Raises:
        ConfigurationError: If a recorded value is not text the package parser can read,
            naming the side, the key and the value, and saying what to do.
    """
    recorded: dict[str, str] = {}
    read: dict[str, Version] = {}
    for key in PACKAGES:
        value = sidecar.get(key)
        if value is None or value == UNKNOWN_VERSION:
            recorded[key] = UNKNOWN_VERSION
            continue
        version = read_version(value)
        if version is None:
            raise ConfigurationError(
                f"the {owner} of {model} record {key} {value!r}, which is not a version that "
                f"can be read, so the two loads cannot be compared; embed the {owner} again "
                "with copernicus, which records the versions it runs under, or remove the key "
                "from a description written by hand, so that it counts as not recorded"
            )
        recorded[key] = value
        read[key] = version
    return _Side(recorded, read)


def _refuse_two_lines(texts: _Side, references: _Side, model: str) -> None:
    """Refuse two recorded transformers versions that are not on one major line.

    Raises:
        ConfigurationError: Naming both versions, whether they belong to two major versions
            or to two epochs, and saying to embed both sides in one environment.
    """
    first = texts.read.get(TRANSFORMERS_VERSION_KEY)
    second = references.read.get(TRANSFORMERS_VERSION_KEY)
    if first is None or second is None or major_line(first) == major_line(second):
        return
    lines = (
        "versions of two epochs, whose release numbers belong to two numberings"
        if first.epoch != second.epoch
        else "two major versions"
    )
    raise ConfigurationError(
        f"the texts and the reference embeddings of {model} were embedded under transformers "
        f"{texts.recorded[TRANSFORMERS_VERSION_KEY]} and "
        f"{references.recorded[TRANSFORMERS_VERSION_KEY]}, {lines}, which can load one model "
        f"into other vectors; {ONE_ENVIRONMENT}"
    )


def _warnings(sides: Mapping[str, _Side], model: str) -> list[str]:
    """Say what the comparison could not rule out, one message per side or package.

    A side that records no version gets one message; so does each package whose two
    recorded versions are two versions, not two spellings of one, such as ``v4.57.6`` and
    ``4.57.6``.
    """
    found: list[str] = []
    for side, owner in OWNERS.items():
        missing = [name for key, name in PACKAGES.items() if key not in sides[side].read]
        if missing:
            versions = "version" if len(missing) == 1 else "versions"
            found.append(
                f"the {owner} of {model} record no version of {' or '.join(missing)} (a set "
                "built by hand records none), so the two loads could not be compared; the texts "
                f"are scored against the references, and {RECORDED} the {versions} as unknown; "
                "embed both sides with copernicus in one environment to compare them"
            )
    for key, package in PACKAGES.items():
        first, second = sides[TEXTS].read.get(key), sides[REFERENCES].read.get(key)
        if first is None or second is None or first == second:
            continue
        same_line = ", two releases of one major version" if key == TRANSFORMERS_VERSION_KEY else ""
        found.append(
            f"the texts and the reference embeddings of {model} were embedded under {package} "
            f"{sides[TEXTS].recorded[key]} and {sides[REFERENCES].recorded[key]}{same_line}; "
            f"they are scored together, and {RECORDED} both versions; {ONE_ENVIRONMENT} to rule "
            "out that their vectors differ"
        )
    return found


def compare_loads(texts: Mapping[str, Any], references: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse texts and references of two loads of an encoder; return what was compared.

    Every function of the library that receives a text embedding set and reference embeddings
    calls this before it scores or describes them: `copernicus.similarity.score_similarity`,
    `copernicus.similarity.score_similarity_many`, `copernicus.similarity.run_grid` (for every
    key before any key is scored), the pole scorer `copernicus.similarity.poles.PoleScorer`,
    which logs the warnings, `copernicus.hybrid.score_hybrid` (for the library's own encoder
    also before its embedding pass), `copernicus.similarity.scoring_provenance` and the
    provenance of the hybrid coder, which record the result. The arithmetic kernel
    `copernicus.similarity.per_reference_similarity` receives arrays and checks nothing.

    Every recorded version is read whole by the package parser (`copernicus._versions`); two
    recorded transformers versions are on one line when their epochs and their first release
    numbers agree.

    Args:
        texts: The sidecar of the embedded texts.
        references: The sidecar of the reference embeddings.

    Returns:
        ``texts`` and ``references``, the transformers and torch versions that each side
        records, as recorded (``"unknown"`` for one that it does not record), and ``warnings``,
        one message for each side that records no version and for each package whose two
        recorded versions differ without being refused; the list is empty when the two loads
        agree.

    Raises:
        ConfigurationError: If an identity key differs (naming each such key with both values,
            and saying to embed both sides with one encoder specification), the loaded commits
            differ where both sides record one, a side records a version that the package
            parser cannot read (also when the other side records the same text or none), or
            the two sides record transformers versions of two major versions or two epochs
            (naming both, and saying to embed both sides in one environment).
    """
    check_same_encoder(texts, references, OWNERS[TEXTS], OWNERS[REFERENCES])
    model = str(texts["hf_id"])
    sides = {
        TEXTS: _read_side(texts, OWNERS[TEXTS], model),
        REFERENCES: _read_side(references, OWNERS[REFERENCES], model),
    }
    _refuse_two_lines(sides[TEXTS], sides[REFERENCES], model)
    return {
        TEXTS: sides[TEXTS].recorded,
        REFERENCES: sides[REFERENCES].recorded,
        "warnings": _warnings(sides, model),
    }
