import hashlib
import json

import pytest

from copernicus.coders import definitions_section
from copernicus.errors import RubricValidationError
from copernicus.rubric import (
    Rubric,
    load_rubric,
    rubric_from_dict,
    toy_rubric_path,
    validate_rubric_dict,
)


def test_toy_rubric_loads_with_three_attributes_of_three_markers() -> None:
    r = load_rubric(toy_rubric_path())
    assert r.rubric_id == "constructive_feedback_v1"
    assert len(r.attributes) == 3 and all(len(a.markers) == 3 for a in r.attributes)
    assert all(
        m.level(5).description and m.level(1).description for a in r.attributes for m in a.markers
    )
    assert len(r.content_hash()) == 16


def test_missing_level_and_bad_id_reported_with_paths() -> None:
    data = json.loads(toy_rubric_path().read_text())
    del data["attributes"][0]["markers"][0]["sub_rubric"]["3"]
    data["attributes"][1]["attribute_id"] = ""
    with pytest.raises(RubricValidationError) as e:
        validate_rubric_dict(data)
    msg = str(e.value)
    assert "attributes/0/markers/0/sub_rubric" in msg and "attributes/1/attribute_id" in msg
    assert len(msg.splitlines()) >= 2


def test_duplicate_marker_name_within_attribute_rejected_but_allowed_across() -> None:
    data = json.loads(toy_rubric_path().read_text())
    data["attributes"][0]["markers"][1]["name"] = data["attributes"][0]["markers"][0]["name"]
    with pytest.raises(RubricValidationError, match="marker name"):
        validate_rubric_dict(data)
    data = json.loads(toy_rubric_path().read_text())
    data["attributes"][1]["markers"][0]["name"] = data["attributes"][0]["markers"][0]["name"]
    validate_rubric_dict(data)  # no error


def test_unknown_top_level_key_rejected() -> None:
    data = json.loads(toy_rubric_path().read_text())
    data["extra"] = 1
    with pytest.raises(RubricValidationError, match="extra"):
        validate_rubric_dict(data)


def test_duplicate_attribute_and_marker_ids_rejected() -> None:
    data = json.loads(toy_rubric_path().read_text())
    data["attributes"][1]["attribute_id"] = data["attributes"][0]["attribute_id"]
    data["attributes"][2]["markers"][0]["marker_id"] = data["attributes"][0]["markers"][0][
        "marker_id"
    ]
    with pytest.raises(RubricValidationError) as e:
        validate_rubric_dict(data)
    lines = str(e.value).splitlines()
    assert lines == [
        "attributes/1/attribute_id: id 'specificity' collides with another attribute or "
        "marker id ('specificity'); ids are compared without regard to case",
        "attributes/2/markers/0/marker_id: id 'spec_examples' collides with another attribute "
        "or marker id ('spec_examples'); ids are compared without regard to case",
    ]


def test_accessors_and_level_range() -> None:
    r = load_rubric(toy_rubric_path())
    attribute = r.attribute("tone")
    assert attribute.name
    assert r.marker_ids("tone") == ["tone_respect", "tone_balance", "tone_blame"]
    assert len(r.marker_names("tone")) == 3
    marker = attribute.markers[0]
    assert [marker.level(n).level for n in range(1, 6)] == [1, 2, 3, 4, 5]
    assert marker.level(1).label == "Clearly low" and marker.level(5).label == "Clearly high"
    for bad in (0, 6):
        with pytest.raises(KeyError, match="level"):
            marker.level(bad)
    with pytest.raises(KeyError, match="no_such_attribute"):
        r.attribute("no_such_attribute")


def test_rubric_is_frozen_and_hash_is_stable_across_load_paths() -> None:
    data = json.loads(toy_rubric_path().read_text())
    from_dict = rubric_from_dict(data)
    from_file = load_rubric(toy_rubric_path())
    assert isinstance(from_dict, Rubric)
    assert from_dict.content_hash() == from_file.content_hash()
    assert from_dict.definitions["overall standard"]
    with pytest.raises(AttributeError):
        from_dict.concept = "changed"  # type: ignore[misc]
    data["concept"] = "Another construct"
    assert rubric_from_dict(data).content_hash() != from_file.content_hash()


def test_load_rubric_reports_unparseable_and_missing_files(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(RubricValidationError, match="broken.json"):
        load_rubric(broken)
    latin = tmp_path / "latin.json"
    latin.write_bytes(b'{"rubric_id": "caf\xe9"}')
    with pytest.raises(RubricValidationError, match="latin.json: not a UTF-8 JSON file"):
        load_rubric(latin)
    with pytest.raises(FileNotFoundError):
        load_rubric(tmp_path / "absent.json")


def test_rubric_from_dict_validates_first() -> None:
    data = json.loads(toy_rubric_path().read_text())
    del data["definitions"]
    with pytest.raises(RubricValidationError, match="definitions"):
        rubric_from_dict(data)


@pytest.mark.parametrize("content", ["[1, 2]", '"just a string"', "42", "null"])
def test_non_object_top_level_raises_rubric_validation_error(tmp_path, content: str) -> None:
    path = tmp_path / "wrong_shape.json"
    path.write_text(content)
    with pytest.raises(RubricValidationError, match="<root>"):
        load_rubric(path)


def test_duplicate_marker_name_without_attribute_id_names_the_index() -> None:
    data = json.loads(toy_rubric_path().read_text())
    data["attributes"][1]["attribute_id"] = ""
    data["attributes"][1]["markers"][1]["name"] = data["attributes"][1]["markers"][0]["name"]
    with pytest.raises(RubricValidationError) as e:
        validate_rubric_dict(data)
    msg = str(e.value)
    assert "in attribute at index 1" in msg and "'None'" not in msg


def test_content_hash_follows_its_documented_definition() -> None:
    raw = json.loads(toy_rubric_path().read_text(encoding="utf-8"))
    raw["definitions"] = [[term, meaning] for term, meaning in raw["definitions"].items()]
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert load_rubric(toy_rubric_path()).content_hash() == expected


def without_attribute_sources() -> dict:
    """The toy rubric with its sources given for the rubric as a whole only."""
    data = json.loads(toy_rubric_path().read_text(encoding="utf-8"))
    for attribute in data["attributes"]:
        del attribute["literature_sources"]
    return data


def test_sources_may_be_given_for_the_rubric_alone_and_stay_required_there() -> None:
    data = without_attribute_sources()
    rubric = rubric_from_dict(data)
    assert rubric.literature_sources == tuple(data["literature_sources"])
    assert all(attribute.literature_sources == () for attribute in rubric.attributes)
    each = load_rubric(toy_rubric_path())
    assert all(len(attribute.literature_sources) > 0 for attribute in each.attributes)
    del data["literature_sources"]
    with pytest.raises(RubricValidationError) as raised:
        validate_rubric_dict(data)
    assert str(raised.value) == "<root>: 'literature_sources' is a required property"


def test_an_attribute_without_sources_is_written_with_an_empty_list_and_one_hash() -> None:
    data = without_attribute_sources()
    rubric = rubric_from_dict(data)
    written = rubric.to_dict()
    assert [attribute["literature_sources"] for attribute in written["attributes"]] == [[]] * 3
    assert rubric_from_dict(written) == rubric
    assert rubric_from_dict(written).content_hash() == rubric.content_hash()
    data["attributes"][0]["literature_sources"] = ["A source of the first attribute"]
    assert rubric_from_dict(data).content_hash() != rubric.content_hash()


def test_glossary_order_changes_the_hash_as_it_changes_the_prompt() -> None:
    data = json.loads(toy_rubric_path().read_text(encoding="utf-8"))
    data["definitions"] = dict(reversed(list(data["definitions"].items())))
    reordered = rubric_from_dict(data)
    original = load_rubric(toy_rubric_path())
    assert dict(reordered.definitions) == dict(original.definitions)
    assert reordered.content_hash() != original.content_hash()
    assert definitions_section(reordered) != definitions_section(original)
