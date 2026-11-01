"""The endpoint of an OpenAI-compatible backend enters the run identity, its credentials never."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from openai_support import StubClient

from copernicus.coders import CodingSettings, code_texts
from copernicus.data import TextTable
from copernicus.errors import BackendError, StoreError
from copernicus.llm import OpenAICompatibleBackend
from copernicus.llm.openai_compatible import endpoint_path_digest
from copernicus.rubric.model import Rubric

NO_TEXTS = CodingSettings(runs=1, text_range=(0, 0))


def path_digest(path: str) -> str:
    """Return the first 16 hexadecimal characters of the SHA-256 of a URL path."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def provenance_of(url: str) -> dict[str, str]:
    """Return the provenance of a backend with an injected client for the URL."""
    return OpenAICompatibleBackend("m", base_url=url, client=StubClient()).provenance()


def test_two_deployments_on_one_host_differ_in_the_path_digest_alone() -> None:
    plain = provenance_of("https://host/v1")
    deployment = provenance_of("https://host/openai/deployments/a")
    assert plain["endpoint"] == deployment["endpoint"] == "https://host"
    assert plain["endpoint_path_digest"] == path_digest("/v1")
    assert deployment["endpoint_path_digest"] == path_digest("/openai/deployments/a")
    assert {key for key in plain if plain[key] != deployment[key]} == {"endpoint_path_digest"}


@pytest.mark.parametrize(
    ("url", "path"),
    [
        ("https://host/v1", "/v1"),
        ("https://host/v1/", "/v1"),
        ("https://host/v1//", "/v1"),
        ("https://host/a%20b/c/", "/a%20b/c"),
        ("https://host/OpenAI/Deployments/A", "/OpenAI/Deployments/A"),
    ],
)
def test_the_path_digest_hashes_the_path_without_its_trailing_slashes(url: str, path: str) -> None:
    assert endpoint_path_digest(url) == path_digest(path)


def test_a_trailing_slash_makes_no_other_deployment_but_another_path_does() -> None:
    assert endpoint_path_digest("https://host/v1") == endpoint_path_digest("https://host/v1/")
    assert provenance_of("https://host/v1") == provenance_of("https://host/v1/")
    assert endpoint_path_digest("https://host/v1") != endpoint_path_digest("https://host/v2")
    assert endpoint_path_digest("https://host/v1") != endpoint_path_digest("https://host/V1")


@pytest.mark.parametrize(
    "url", ["https://host", "https://host/", "https://host//", "https://host:8443/?q=1#f"]
)
def test_an_endpoint_without_a_path_has_the_path_digest_none(url: str) -> None:
    assert endpoint_path_digest(url) == "none"
    assert provenance_of(url)["endpoint_path_digest"] == "none"


def test_user_password_query_and_fragment_never_enter_the_provenance() -> None:
    given = provenance_of("https://user:pw@host/v1?key=x#f")
    assert given == provenance_of("https://host/v1")
    for part in ("user", "pw", "key=x", "#f", "?"):
        assert not any(part in value for value in given.values()), part
    marked = provenance_of("https://someuser:secretpw@host/v1?apikey=querytoken#fragmenttoken")
    for token in ("someuser", "secretpw", "apikey", "querytoken", "fragmenttoken"):
        assert not any(token in value for value in marked.values()), token


def test_an_unparsable_url_is_refused_without_repeating_it() -> None:
    with pytest.raises(BackendError, match="invalid base_url") as refused:
        endpoint_path_digest("http://[invalid/v1")
    assert "[invalid" not in str(refused.value)


def test_two_deployments_on_one_host_are_two_run_identities(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    first = OpenAICompatibleBackend("m", base_url="https://host/v1", client=StubClient())
    code_texts(toy_corpus, toy_rubric, first, store, NO_TEXTS)
    other = OpenAICompatibleBackend(
        "m", base_url="https://host/openai/deployments/a", client=StubClient()
    )
    with pytest.raises(StoreError, match="mismatch on backend_endpoint_path_digest"):
        code_texts(toy_corpus, toy_rubric, other, store, NO_TEXTS)
    same = OpenAICompatibleBackend("m", base_url="https://host/v1", client=StubClient())
    assert code_texts(toy_corpus, toy_rubric, same, store, NO_TEXTS).n_attribute_calls == 0
