import pytest

from copernicus.coders import (
    MIN_FUZZY_WORDS,
    VERIFIED_EXACT,
    VERIFIED_FUZZY,
    VERIFIED_NONE,
    VERIFY_THRESHOLD,
    collapse_whitespace,
    match_quote,
    normalize,
    process_extraction,
    validate_extraction_block,
    verifier_version,
)
from copernicus.errors import EvidenceContractError
from copernicus.rubric.model import Rubric

DOC = (
    "The team met on Monday. We decided to paint the garden shed with two colours first. "
    "Everyone agreed."
)
PASSAGE = "We decided to paint the garden shed"


def _empty(names: list[str]) -> dict[str, dict]:
    return {name: {"evaluable": False, "quotes": []} for name in names}


def test_constants_have_the_documented_values() -> None:
    assert VERIFY_THRESHOLD == 88.0
    assert MIN_FUZZY_WORDS == 3
    assert (VERIFIED_NONE, VERIFIED_EXACT, VERIFIED_FUZZY) == (0, 1, 2)


def test_verifier_version_names_rapidfuzz() -> None:
    assert verifier_version().startswith("rapidfuzz ")


def test_exact_fuzzy_and_rejected() -> None:
    doc = normalize(DOC)
    exact = match_quote(PASSAGE, doc)
    assert exact.verified == VERIFIED_EXACT and exact.score == 100.0
    assert exact.start is not None and exact.end is not None
    assert doc[exact.start : exact.end] == PASSAGE.lower()
    fuzzy = match_quote("We decided to paint the gardn shed with two colours", doc)
    assert fuzzy.verified == VERIFIED_FUZZY and fuzzy.score >= 88.0 and fuzzy.start is not None
    invented = match_quote("They refused to run any experiment at all", doc)
    assert invented.verified == VERIFIED_NONE
    assert 0.0 < invented.score < VERIFY_THRESHOLD
    assert invented.start is not None and invented.end is not None


def test_short_quotes_must_match_exactly() -> None:
    doc = normalize(DOC)
    assert match_quote("two colours", doc).verified == VERIFIED_EXACT
    assert match_quote("twoo colours", doc).verified == VERIFIED_NONE
    assert match_quote("paint the gardn shed", doc).verified == VERIFIED_FUZZY


def test_quote_longer_than_document_uses_full_ratio() -> None:
    doc = normalize("short doc here")
    padded = match_quote(
        "short doc here plus a long invented tail that is not in the text at all", doc
    )
    assert padded.verified == VERIFIED_NONE and padded.score < VERIFY_THRESHOLD
    assert (padded.start, padded.end) == (0, len(doc))
    assert match_quote("short doc here", doc).verified == VERIFIED_EXACT


def test_empty_quote_or_document_is_unverified_without_a_window() -> None:
    doc = normalize(DOC)
    for quote, document in (("", doc), ("   ", doc), ("anything at all", "")):
        match = match_quote(quote, document)
        assert match.verified == VERIFIED_NONE and match.score == 0.0
        assert match.start is None and match.end is None


def test_normalize_unifies_quotes_dashes_and_whitespace() -> None:
    assert normalize("  “Hello”   —  ’x’ ") == "\"hello\" - 'x'"
    assert collapse_whitespace(" a \n\n b\tc ") == "a b c"


def test_process_extraction_records_all_and_builds_document_ordered_unlabelled_list(
    toy_rubric: Rubric,
) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    parsed = {
        names[0]: {
            "evaluable": True,
            "quotes": ["Everyone agreed.", PASSAGE, "Everyone agreed."],
        },
        names[1]: {"evaluable": True, "quotes": [PASSAGE, "not in the text"]},
        names[2]: {"evaluable": False, "quotes": []},
    }
    p = process_extraction(parsed, attr, DOC, names, threshold=88.0, min_fuzzy_words=3)
    assert len(p.evidence) == 5
    first = attr.markers[0].marker_id
    assert [e.duplicate_of for e in p.evidence if e.marker_id == first] == [None, None, 0]
    assert p.markers_to_rate == (names[0], names[1])
    assert p.evaluable[attr.markers[2].marker_id] is False
    assert p.evaluable[first] is True
    assert p.passages == (PASSAGE, "Everyone agreed.")
    assert sum(e.used_in_stage2 for e in p.evidence) == 2
    assert p.justification[first] == f"{PASSAGE}\nEveryone agreed."
    assert p.justification[attr.markers[2].marker_id] == ""
    assert p.by_marker[names[0]] == (PASSAGE, "Everyone agreed.")
    assert p.by_marker[names[1]] == (PASSAGE,)
    assert p.by_marker[names[2]] == ()
    unverified = [e for e in p.evidence if e.quote_raw == "not in the text"]
    assert len(unverified) == 1
    assert unverified[0].verified == VERIFIED_NONE and unverified[0].match_score < 88.0
    assert unverified[0].used_in_stage2 == 0


def test_process_extraction_accepts_marker_ids_and_normalises_whitespace(
    toy_rubric: Rubric,
) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    parsed = _empty(names)
    del parsed[names[0]]
    parsed[attr.markers[0].marker_id] = {
        "evaluable": True,
        "quotes": ["We decided   to paint\nthe garden shed"],
    }
    p = process_extraction(parsed, attr, DOC, names, threshold=88.0, min_fuzzy_words=3)
    row = p.evidence[0]
    assert row.marker_id == attr.markers[0].marker_id and row.attribute_id == attr.attribute_id
    assert row.quote_raw == "We decided   to paint\nthe garden shed"
    assert row.quote_norm == PASSAGE and row.verified == VERIFIED_EXACT
    assert p.passages == (PASSAGE,)


def test_cross_marker_tie_breaks_by_marker_order_and_lists_both(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    parsed = _empty(names)
    parsed[names[0]] = {"evaluable": True, "quotes": [PASSAGE]}
    parsed[names[1]] = {"evaluable": True, "quotes": [PASSAGE]}
    order = [names[1], names[0], names[2]]
    p = process_extraction(parsed, attr, DOC, order, threshold=88.0, min_fuzzy_words=3)
    assert p.passages == (PASSAGE,)
    used = [e for e in p.evidence if e.used_in_stage2 == 1]
    assert len(used) == 1 and used[0].marker_id == attr.markers[1].marker_id
    assert p.markers_to_rate == (names[1], names[0])
    assert p.by_marker[names[0]] == (PASSAGE,) and p.by_marker[names[1]] == (PASSAGE,)


def test_process_extraction_honours_threshold_and_min_fuzzy_words(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    parsed = _empty(names)
    parsed[names[0]] = {"evaluable": True, "quotes": ["We decided to paint the gardn shed"]}
    lenient = process_extraction(parsed, attr, DOC, names, threshold=88.0, min_fuzzy_words=3)
    assert lenient.evidence[0].verified == VERIFIED_FUZZY
    strict_words = process_extraction(parsed, attr, DOC, names, threshold=88.0, min_fuzzy_words=99)
    assert strict_words.evidence[0].verified == VERIFIED_NONE
    assert strict_words.passages == () and strict_words.markers_to_rate == ()
    strict_score = process_extraction(parsed, attr, DOC, names, threshold=99.9, min_fuzzy_words=3)
    assert strict_score.evidence[0].verified == VERIFIED_NONE


def test_process_extraction_rejects_unknown_key_and_missing_marker(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    unknown = _empty(names)
    unknown["Invented marker"] = unknown.pop(names[0])
    with pytest.raises(EvidenceContractError, match="Invented marker"):
        process_extraction(unknown, attr, DOC, names, threshold=88.0, min_fuzzy_words=3)
    missing = _empty(names)
    del missing[names[1]]
    with pytest.raises(EvidenceContractError, match=names[1]):
        process_extraction(missing, attr, DOC, names, threshold=88.0, min_fuzzy_words=3)


def test_validate_block_accepts_names_and_ids(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    block = _empty(names[1:])
    block[attr.markers[0].marker_id] = {"evaluable": True, "quotes": [PASSAGE]}
    assert validate_extraction_block(block, attr) == (True, "")


def test_validate_block_rejects_marker_given_by_name_and_id(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    twice = _empty(names[2:])
    twice[names[0]] = {"evaluable": False, "quotes": []}
    twice[attr.markers[0].marker_id] = {"evaluable": False, "quotes": []}
    ok, reason = validate_extraction_block(twice, attr)
    assert not ok and "more than once" in reason


def test_validate_block_rejects_contradiction_and_unknown_key(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    claimed = _empty(names)
    claimed[names[0]] = {"evaluable": True, "quotes": []}
    ok, reason = validate_extraction_block(claimed, attr)
    assert not ok and "evaluable=true but zero quotes" in reason
    stray = _empty(names)
    stray[names[0]] = {"evaluable": False, "quotes": [PASSAGE]}
    ok, reason = validate_extraction_block(stray, attr)
    assert not ok and "evaluable=false but quotes present" in reason
    unknown = _empty(names)
    unknown["Invented marker"] = unknown.pop(names[0])
    ok, reason = validate_extraction_block(unknown, attr)
    assert not ok and "unknown marker key" in reason


def test_validate_block_rejects_wrong_count_shape_and_types(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    short = _empty(names[:2])
    ok, reason = validate_extraction_block(short, attr)
    assert not ok and "has 2 entries, expected 3" in reason
    cases = {
        "entry is not an object": "text",
        "missing 'evaluable' or 'quotes'": {"quotes": []},
        "must be a list of non-empty strings": {"evaluable": True, "quotes": ["ok", "  "]},
        "'evaluable' must be a bool": {"evaluable": 1, "quotes": []},
    }
    for expected, entry in cases.items():
        block = _empty(names)
        block[names[0]] = entry  # type: ignore[assignment]
        ok, reason = validate_extraction_block(block, attr)
        assert not ok and expected in reason


@pytest.mark.parametrize("block", [[], ["a", "b"], None, "text", 3])
def test_validate_block_rejects_a_non_mapping(toy_rubric: Rubric, block: object) -> None:
    ok, reason = validate_extraction_block(block, toy_rubric.attributes[0])
    assert not ok and reason == f"top-level type {type(block).__name__} is not a JSON object"
