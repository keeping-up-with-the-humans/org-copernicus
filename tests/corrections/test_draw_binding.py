"""A human-sample draw binds to the table it was drawn from: the fingerprint and the checks."""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import copernicus.corrections.data as data_module
import copernicus.corrections.sampling as sampling_module
from copernicus.corrections import (
    CorrectionData,
    SampleDraw,
    attach_draw,
    draw_from_data,
    draw_groups,
    draw_human_sample,
    mean_ppi,
    population_hash_of,
    repeated_study,
)
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError

pytestmark = pytest.mark.filterwarnings("error")

IDS = np.arange(8).astype(str)
SOURCE_GROUPS = np.repeat(["a0", "a1", "a2", "a3"], 2)
SENTENCE = (
    "the draw was made on a different table than the data (different texts or groups): draw on "
    "the table the correction data are built from, then attach that draw"
)
DIFFERENT_TABLE = "^" + re.escape(SENTENCE) + "; "
Population = tuple[np.ndarray, np.ndarray]


def _data(ids: np.ndarray, groups: np.ndarray) -> CorrectionData:
    """Correction data over the given texts and groups, without gold values."""
    values = np.arange(ids.size, dtype=float)
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": groups,
            "score": values / 2,
            "gold": np.nan,
            "log_chars": values,
        }
    )
    return CorrectionData(frame, ("score",))


def _values(ids: np.ndarray) -> pd.Series:
    return pd.Series(np.arange(ids.size, dtype=float), index=ids)


def _source_draw() -> SampleDraw:
    """Two of the four source groups: the texts 4 to 7, each with pi = 1/2."""
    return draw_groups(IDS, SOURCE_GROUPS, IDS, 4, seed=0)


def _added(ids: np.ndarray, groups: np.ndarray) -> Population:
    return np.append(ids, "8"), np.append(groups, "a0")


def _removed(ids: np.ndarray, groups: np.ndarray) -> Population:
    return ids[1:], groups[1:]


def _moved(ids: np.ndarray, groups: np.ndarray) -> Population:
    regrouped = groups.copy()
    regrouped[1] = "a1"
    return ids, regrouped


def _renamed(ids: np.ndarray, groups: np.ndarray) -> Population:
    return ids, np.char.replace(groups, "a", "c")


CHANGES: dict[str, Callable[[np.ndarray, np.ndarray], Population]] = {
    "one text added": _added,
    "one text removed": _removed,
    "one text moved to another group": _moved,
    "every group renamed": _renamed,
}


def test_a_draw_attached_to_another_partition_of_its_texts_is_refused_first() -> None:
    draw = _source_draw()
    assert (draw.ids.tolist(), draw.groups.tolist()) == (["4", "5", "6", "7"], ["a2", "a3"])
    assert draw.pi == 0.5
    target = _data(IDS, np.tile(["b0", "b1", "b2", "b3"], 2))
    with pytest.raises(ConfigurationError, match=DIFFERENT_TABLE):
        attach_draw(target, _values(IDS), draw)
    # The table is checked before any gold value: values that would be refused are not read.
    with pytest.raises(ConfigurationError, match=DIFFERENT_TABLE):
        attach_draw(target, pd.Series(dtype=float), draw)
    attached = attach_draw(_data(IDS, SOURCE_GROUPS), _values(IDS), draw)
    assert attached.labelled_ids.tolist() == ["4", "5", "6", "7"]
    assert attached.pi is not None and np.all(attached.pi == 0.5)
    assert mean_ppi(attached).details["clustered"] is True


@pytest.mark.parametrize("change", list(CHANGES), ids=list(CHANGES))
def test_a_table_changed_after_the_draw_is_refused(change: str) -> None:
    ids, groups = CHANGES[change](IDS, SOURCE_GROUPS)
    assert set(_source_draw().ids) <= set(ids)
    with pytest.raises(ConfigurationError, match=DIFFERENT_TABLE):
        attach_draw(_data(ids, groups), _values(ids), _source_draw())


@pytest.mark.parametrize(
    ("change", "counts"),
    [
        ("one text added", "the data hold 9 texts in 4 groups"),
        ("one text removed", "the data hold 7 texts in 4 groups"),
        ("one text moved to another group", "the data hold 8 texts in 4 groups"),
    ],
    ids=["added", "removed", "moved"],
)
def test_the_refusal_of_another_table_gives_the_recorded_and_the_actual_counts(
    change: str, counts: str
) -> None:
    ids, groups = CHANGES[change](IDS, SOURCE_GROUPS)
    with pytest.raises(ConfigurationError) as raised:
        attach_draw(_data(ids, groups), _values(ids), _source_draw())
    recorded = "the draw records n_units = 8 and n_groups = 4"
    assert str(raised.value) == f"{SENTENCE}; {recorded}, {counts}"


def test_the_refusal_of_another_table_names_the_first_ten_drawn_texts_it_lacks() -> None:
    ids = np.array([f"t{number:02d}" for number in range(30)])
    groups = np.repeat(["g0", "g1", "g2"], 10)
    draw = draw_groups(ids, groups, ids, 20, seed=0)
    kept = ~np.isin(ids, draw.ids)
    with pytest.raises(ConfigurationError) as raised:
        attach_draw(_data(ids[kept], groups[kept]), _values(ids[kept]), draw)
    absent = ", ".join(repr(text_id) for text_id in sorted(draw.ids.tolist())[:10])
    assert str(raised.value) == (
        f"{SENTENCE}; the draw records n_units = 30 and n_groups = 3, the data hold 10 texts "
        f"in 1 group; drawn texts not in the data: {absent}, ... (10 more)"
    )
    removed = IDS != "7"
    lacking = r"n_groups = 4, the data hold 7 texts in 4 groups; drawn texts not in the data: '7'$"
    with pytest.raises(ConfigurationError, match=lacking):
        attach_draw(_data(IDS[removed], SOURCE_GROUPS[removed]), _values(IDS), _source_draw())


def test_a_draw_whose_ids_repeat_is_refused() -> None:
    draw = _source_draw()
    repeated = replace(draw, ids=np.append(draw.ids, draw.ids[0]))
    with pytest.raises(ConfigurationError) as raised:
        attach_draw(_data(IDS, SOURCE_GROUPS), _values(IDS), repeated)
    assert str(raised.value) == (
        "the draw does not describe the table it was made on: drawn ids repeat: '4'"
    )


def _count_hashing(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the number of texts of every fingerprint the data or a draw computes."""
    lengths: list[int] = []
    for module in (data_module, sampling_module):
        real: Callable[..., str] = module.population_hash_of

        def counted(text_id: object, group_id: object, real: Callable[..., str] = real) -> str:
            lengths.append(len(np.asarray(text_id)))
            return real(text_id, group_id)

        monkeypatch.setattr(module, "population_hash_of", counted)
    return lengths


def test_the_data_compute_their_fingerprint_once_per_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths = _count_hashing(monkeypatch)
    data = _data(IDS, SOURCE_GROUPS)
    first = data.population_hash
    assert data.population_hash is first and lengths == [8]
    assert first == population_hash_of(IDS, SOURCE_GROUPS)
    same = replace(data, frame=data.frame.copy())
    assert same is not data and same.population_hash == first and lengths == [8, 8]
    fewer = replace(data, frame=data.frame.iloc[1:])
    assert fewer.population_hash == population_hash_of(IDS[1:], SOURCE_GROUPS[1:])
    assert fewer.population_hash != first


def test_a_draw_of_the_data_equals_the_draw_of_its_arrays_and_carries_their_fingerprint() -> None:
    data = _data(IDS, SOURCE_GROUPS)
    for quota, seed in ((4, 0), (2, 7), (8, 3)):
        own = draw_from_data(data, quota, seed=seed)
        arrays = draw_groups(data.ids, data.groups, data.ids, quota, seed=seed)
        assert own.ids.tolist() == arrays.ids.tolist()
        assert own.groups.tolist() == arrays.groups.tolist()
        assert (own.pi, own.m, own.n_groups, own.n_units, own.quota, own.seed) == (
            arrays.pi,
            arrays.m,
            arrays.n_groups,
            arrays.n_units,
            arrays.quota,
            arrays.seed,
        )
        assert own.population_hash == arrays.population_hash == data.population_hash
        assert own.population_hash == population_hash_of(data.ids, data.groups)


def test_draw_groups_takes_no_fingerprint_from_the_caller() -> None:
    assert "population_hash" not in inspect.signature(draw_groups).parameters


def test_a_draw_of_the_data_and_the_repeated_study_reuse_the_fingerprint_of_the_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data(IDS, SOURCE_GROUPS)
    lengths = _count_hashing(monkeypatch)
    draw = draw_from_data(data, 4, seed=0)
    assert draw.population_hash == population_hash_of(IDS, SOURCE_GROUPS)
    attach_draw(data, _values(IDS), draw)
    assert lengths == [8]
    known = data.frame.assign(truth=np.arange(8.0), log_chars=np.arange(8.0) % 3)
    study = CorrectionData(frame=known, score_columns=("score",))
    lengths.clear()
    repeated_study(study, truth_gold="truth", sizes=(4,), draws=5, estimators=("raw",))
    assert lengths == [8]


def test_the_population_hash_is_the_digest_of_the_sorted_lines_of_the_texts() -> None:
    pairs = zip(IDS, SOURCE_GROUPS, strict=True)
    lines = sorted(f"{text_id}\t{group}\n" for text_id, group in pairs)
    expected = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    found = population_hash_of(IDS, SOURCE_GROUPS)
    assert found == expected and re.fullmatch(r"[0-9a-f]{64}", found)
    order = np.random.default_rng(1).permutation(IDS.size)
    assert population_hash_of(IDS[order], SOURCE_GROUPS[order]) == found
    assert population_hash_of(np.arange(8), pd.Series(SOURCE_GROUPS)) == found
    changed = {population_hash_of(*change(IDS, SOURCE_GROUPS)) for change in CHANGES.values()}
    assert len(changed) == len(CHANGES) and found not in changed
    with pytest.raises(ConfigurationError, match="equal length"):
        population_hash_of(IDS, SOURCE_GROUPS[:-1])


def test_a_draw_records_the_hash_of_the_table_it_was_drawn_from(toy_corpus: TextTable) -> None:
    frame = toy_corpus.frame
    draw = draw_human_sample(toy_corpus, n=10, seed=11)
    expected = population_hash_of(frame["text_id"].astype(str), frame["group_id"].astype(str))
    assert draw.population_hash == expected
    assert _source_draw().population_hash == population_hash_of(IDS, SOURCE_GROUPS)
    subset = draw_groups(IDS[2:], SOURCE_GROUPS[2:], IDS[2:], 2, seed=0)
    assert subset.population_hash == population_hash_of(IDS[2:], SOURCE_GROUPS[2:])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ids": np.array(["4", "5", "6", "7", "x"])}, r"drawn texts not in the data: 'x'"),
        ({"groups": np.array(["a2", "a3", "z9"])}, r"drawn groups not in the data: 'z9'"),
        (
            {"ids": np.array(["2", "4", "5", "6", "7"])},
            r"drawn texts whose group in the data is not a drawn group: '2'",
        ),
        (
            {"ids": np.array(["4", "5", "6"])},
            r"texts of drawn groups missing from the drawn ids: '7'",
        ),
        ({"m": 3}, r"m = 3, n_groups = 4 and pi = 0.5; .* give m = 2, n_groups = 4 and pi = 0.5"),
        ({"n_groups": 5}, r"m = 2, n_groups = 5 and pi = 0.5; .* give m = 2, n_groups = 4 and"),
        (
            {"pi": 0.25},
            r"m = 2, n_groups = 4 and pi = 0.25; .* give m = 2, n_groups = 4 and pi = 0.5",
        ),
    ],
    ids=["text", "group", "misgrouped", "left out", "m", "n_groups", "pi"],
)
def test_a_draw_of_the_table_that_does_not_describe_it_is_refused(
    changes: dict[str, object], message: str
) -> None:
    edited = replace(_source_draw(), **changes)
    values = _values(np.append(IDS, "x"))
    with pytest.raises(ConfigurationError, match=r"^the draw does not describe .*" + message):
        attach_draw(_data(IDS, SOURCE_GROUPS), values, edited)


@pytest.mark.parametrize("value", ["", "abc", "a" * 63, "A" * 64, "g" * 64, None, 5])
def test_a_draw_without_a_population_hash_is_refused(value: object) -> None:
    with pytest.raises(ConfigurationError, match="earlier version.*draw again"):
        replace(_source_draw(), population_hash=value)
    recorded = {
        name: getattr(_source_draw(), name)
        for name in ("ids", "groups", "pi", "m", "n_groups", "n_units", "quota", "seed")
    }
    with pytest.raises(TypeError, match="population_hash"):
        SampleDraw(**recorded)
