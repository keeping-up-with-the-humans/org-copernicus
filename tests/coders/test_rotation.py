import random
from pathlib import Path

import pytest

from copernicus.coders import (
    Assignment,
    WordingPool,
    assign,
    decode_seed,
    load_wording_pool,
    marker_order,
    select,
    text_key,
)
from copernicus.errors import PromptContractError

ROLE_HEADER = "variant_id,enabled,extraction,scoring,one_stage"
LEVEL_HEADER = "variant_id,enabled,scoring,one_stage"


def _write_pool(directory: Path, role_rows: list[str], level_header: str = LEVEL_HEADER) -> None:
    (directory / "wording_role.csv").write_text("\n".join([ROLE_HEADER, *role_rows]) + "\n")
    (directory / "wording_task.csv").write_text(f"{ROLE_HEADER}\ntask_a,true,e,s,o\n")
    (directory / "wording_level.csv").write_text(f"{level_header}\nlevel_a,true,s,o\n")


def test_pool_loads_three_three_four_and_texts_filled() -> None:
    pool = load_wording_pool()
    assert [w.variant_id for w in pool.role] == [
        "role_baseline",
        "role_calibrated",
        "role_analytical",
    ]
    assert len(pool.task) == 3 and len(pool.level) == 4
    assert all(
        w.texts["extraction"] and w.texts["scoring"] and w.texts["one_stage"]
        for w in pool.role + pool.task
    )
    assert all(w.texts["scoring"] and w.texts["one_stage"] for w in pool.level)
    assert all(w.enabled for w in pool.role + pool.task + pool.level)


def test_assignment_formula_matches_reference_rng() -> None:
    pool = load_wording_pool()
    a = assign(pool, text_key=17, run_number=2, seed=42)
    rng = random.Random(42 + 17 * 1000 + 2)
    assert a.role_id == rng.choice(pool.role).variant_id
    assert a.task_id == rng.choice(pool.task).variant_id
    assert a.level_id == rng.choice(pool.level).variant_id
    assert a.shuffle_seed == 42 + 17 * 1000 + 2


def test_pinning_and_unknown_ids() -> None:
    pool = select(
        load_wording_pool(), enable=["role_baseline", "task_baseline", "level_independent"]
    )
    assert assign(pool, 1, 1, 42) == Assignment(
        "role_baseline", "task_baseline", "level_independent", 42 + 1000 + 1
    )
    with pytest.raises(PromptContractError, match="role_x"):
        select(load_wording_pool(), enable=["role_x"])
    with pytest.raises(PromptContractError, match="task_x"):
        select(load_wording_pool(), disable=["task_x"])


def test_pinning_one_dimension_leaves_the_others_untouched() -> None:
    pool = select(load_wording_pool(), enable=["role_calibrated"])
    assert [w.variant_id for w in pool.role] == ["role_calibrated"]
    assert len(pool.task) == 3 and len(pool.level) == 4


def test_disable_removes_ids_and_an_emptied_dimension_is_rejected() -> None:
    pool = select(load_wording_pool(), disable=["level_topdown", "level_bottomup"])
    assert [w.variant_id for w in pool.level] == ["level_midpoint", "level_independent"]
    with pytest.raises(PromptContractError, match="role"):
        select(load_wording_pool(), disable=["role_baseline", "role_calibrated", "role_analytical"])


def test_disabled_rows_are_kept_by_the_loader_and_dropped_by_select(tmp_path: Path) -> None:
    _write_pool(tmp_path, ["role_a,true,e,s,o", "role_b,FALSE,e,s,o"])
    pool = load_wording_pool(tmp_path)
    assert load_wording_pool(str(tmp_path)) == pool
    assert [(w.variant_id, w.enabled) for w in pool.role] == [("role_a", True), ("role_b", False)]
    assert [w.variant_id for w in select(pool).role] == ["role_a"]
    assert [w.variant_id for w in select(pool, enable=["role_b"]).role] == ["role_b"]
    assert select(pool, enable=["role_b"]).role[0].enabled
    assert {assign(pool, key, 1, 7).role_id for key in range(30)} == {"role_a"}


def test_assign_without_enabled_wording_in_a_dimension_fails(tmp_path: Path) -> None:
    _write_pool(tmp_path, ["role_a,false,e,s,o"])
    with pytest.raises(PromptContractError, match="role"):
        assign(load_wording_pool(tmp_path), 1, 1, 7)


def test_malformed_wording_files_are_rejected(tmp_path: Path) -> None:
    _write_pool(tmp_path, ["role_a,maybe,e,s,o"])
    with pytest.raises(PromptContractError, match="maybe"):
        load_wording_pool(tmp_path)
    _write_pool(tmp_path, ["role_a,true,e,s,o"], level_header="variant_id,enabled,scoring")
    with pytest.raises(PromptContractError, match="one_stage"):
        load_wording_pool(tmp_path)


def test_get_returns_the_wording_or_fails() -> None:
    pool = load_wording_pool()
    assert pool.get("level", "level_midpoint").variant_id == "level_midpoint"
    with pytest.raises(PromptContractError, match="role_x"):
        pool.get("role", "role_x")
    with pytest.raises(PromptContractError, match="colour"):
        pool.get("colour", "role_baseline")


def test_assignment_is_deterministic_and_varies_across_texts_and_runs() -> None:
    pool = load_wording_pool()
    assert assign(pool, 5, 1, 42) == assign(pool, 5, 1, 42)
    seen = {
        (a.role_id, a.task_id, a.level_id)
        for a in (assign(pool, key, run, 42) for key in range(20) for run in range(1, 4))
    }
    assert len(seen) > 5


def test_text_key_stable_for_strings_and_identity_for_ints() -> None:
    assert (
        text_key(12) == 12 and text_key("abc") == text_key("abc") and 0 <= text_key("abc") < 2**31
    )
    assert text_key(2**31) == text_key(str(2**31)) and 0 <= text_key(2**31) < 2**31
    assert text_key(-3) == text_key("-3") and 0 <= text_key(-3) < 2**31
    assert text_key("abc") != text_key("abd")
    # Golden value: pins the digest size, the byte order, and the 31-bit mask.
    assert text_key("abc") == 1670406728


def test_marker_order_is_seeded_and_unshuffled_when_off() -> None:
    names = ["a", "b", "c", "d"]
    assert marker_order(names, False, 5) == names
    shuffled = marker_order(names, True, 5)
    assert sorted(shuffled) == names
    assert shuffled == marker_order(names, True, 5)
    assert shuffled != names
    assert names == ["a", "b", "c", "d"]
    assert marker_order(["only"], True, 5) == ["only"]
    assert marker_order([], True, 5) == []


def test_decode_seed_adds_the_run_number() -> None:
    assert decode_seed(42, 3) == 45


def test_pool_type_is_exported() -> None:
    assert isinstance(load_wording_pool(), WordingPool)


def test_select_rejects_a_bare_string_other_than_all() -> None:
    with pytest.raises(PromptContractError, match="role_baseline"):
        select(load_wording_pool(), enable="role_baseline")
    assert len(select(load_wording_pool(), enable="all").role) == 3


def test_short_or_blank_rows_are_rejected_with_file_row_and_column(tmp_path: Path) -> None:
    _write_pool(tmp_path, ["role_a,true,e,s,o", "role_b,true"])
    with pytest.raises(PromptContractError, match=r"wording_role\.csv.*role_b.*'extraction'"):
        load_wording_pool(tmp_path)
    _write_pool(tmp_path, ["role_a,true,e,s,o", "role_b"])
    with pytest.raises(PromptContractError, match="role_b.*'enabled'"):
        load_wording_pool(tmp_path)
    _write_pool(tmp_path, ["role_a,true,e,  ,o"])
    with pytest.raises(PromptContractError, match="role_a.*'scoring'"):
        load_wording_pool(tmp_path)
    _write_pool(tmp_path, [",true,e,s,o"])
    with pytest.raises(PromptContractError, match="row #1.*'variant_id'"):
        load_wording_pool(tmp_path)


def test_variant_id_repeated_across_dimensions_is_rejected(tmp_path: Path) -> None:
    _write_pool(tmp_path, ["shared,true,e,s,o"])
    (tmp_path / "wording_level.csv").write_text(f"{LEVEL_HEADER}\nshared,true,s,o\n")
    with pytest.raises(PromptContractError, match="shared.*role.*level"):
        load_wording_pool(tmp_path)
