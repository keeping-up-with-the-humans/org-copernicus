"""Shared fixtures for the copernicus test suite."""

import os

# The hub library reads this switch when it is first imported, so it is set before any other
# import and a test cannot set it later; a download is allowed only on request.
if not os.environ.get("COPERNICUS_TEST_DOWNLOAD"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402

import pytest  # noqa: E402

from copernicus.assets import load_toy_cues, toy_corpus_path  # noqa: E402
from copernicus.data import TextTable  # noqa: E402
from copernicus.llm.fake import FakeBackend, cue_responder  # noqa: E402
from copernicus.rubric.loader import load_rubric, rubric_from_dict, toy_rubric_path  # noqa: E402
from copernicus.rubric.model import Rubric  # noqa: E402

CUES = load_toy_cues()


@pytest.fixture(autouse=True)
def tiktoken_unavailable(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make tiktoken unimportable in every test, unless a test installs a stand-in.

    Loading a tiktoken encoding for the first time downloads it, and the tests never use
    the network; the token estimate then counts characters instead.
    """
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    yield


@pytest.fixture
def toy_rubric() -> Rubric:
    """Return the toy rubric that ships with the package."""
    return load_rubric(toy_rubric_path())


@pytest.fixture
def rewritten_toy_rubric(toy_rubric: Rubric) -> Rubric:
    """Return the toy rubric with its definition and descriptions rewritten and every id kept.

    It has the rubric id of the toy rubric and another content hash, as an edited
    operationalisation that kept its ids has.
    """
    data = toy_rubric.to_dict()
    data["concept_definition"] = "This version measures the opposite of the first definition."
    for attribute in data["attributes"]:
        attribute["description"] = "This facet is reversed relative to the first version."
        for marker in attribute["markers"]:
            for level in marker["sub_rubric"].values():
                level["description"] = "Reversed: " + level["description"]
    return rubric_from_dict(data)


@pytest.fixture
def cues() -> dict[str, list[str]]:
    """Return the cue words of the toy rubric, keyed by marker id."""
    return {marker_id: list(words) for marker_id, words in CUES.items()}


@pytest.fixture
def toy_corpus() -> TextTable:
    """Return the example corpus that ships with the package."""
    return TextTable.read(toy_corpus_path(), group_id="group_id")


@pytest.fixture
def cue_backend(toy_rubric: Rubric) -> Callable[[], FakeBackend]:
    """Return a factory that builds a fresh cue-based fake backend on each call."""

    def build() -> FakeBackend:
        return FakeBackend(cue_responder(toy_rubric, CUES))

    return build
