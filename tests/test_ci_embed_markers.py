"""Every test that needs the encoder carries the embed marker.

The two listed module-level tests may build an encoder without the marker. Their bodies must
patch the class loader and pooling to the stand-ins before embedding, then only assert results.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ci_support import EMBED_MARKER, TESTS

EMBED_DECORATORS = frozenset({"needs_minilm"})
MINILM_NAMES = frozenset({"minilm_cached", "MINILM_AVAILABLE"})
# Identities include the path relative to tests and any containing class or function names.
LOAD_REPLACED_TESTS = frozenset(
    {
        "embeddings/test_encoder_texts::test_the_tokenizer_receives_each_text_as_given",
        "embeddings/test_encoder_texts::"
        "test_the_set_records_the_digest_of_the_texts_in_the_order_given",
    }
)


@dataclass(frozen=True)
class Needing:
    """A test or a module that needs the encoder, and whether it carries the marker."""

    where: str
    marked: bool


def marker_names(node: ast.AST) -> set[str]:
    """Collect the ``pytest.mark.<name>`` names and the bare decorator names below a node."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Attribute):
            if child.value.attr == "mark":
                names.add(child.attr)
        if isinstance(child, ast.Name) and child.id in EMBED_DECORATORS:
            names.add(child.id)
    return names


def pytestmark_statements(tree: ast.Module) -> list[ast.stmt]:
    """Return the module-level ``pytestmark`` assignments of a module."""
    return [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in statement.targets
        )
    ]


def module_marks(tree: ast.Module) -> set[str]:
    """Return the markers that a module sets for all its tests through ``pytestmark``."""
    return set().union(*(marker_names(statement) for statement in pytestmark_statements(tree)))


def is_marked(names: set[str]) -> bool:
    """Tell whether a set of names holds the embed marker or a decorator that applies it."""
    return EMBED_MARKER in names or bool(EMBED_DECORATORS & names)


def imports_torch(node: ast.AST) -> bool:
    """Tell whether a node holds an ``import torch`` statement of any form."""
    return any(
        (
            isinstance(child, ast.Import)
            and any(a.name.split(".")[0] == "torch" for a in child.names)
        )
        or (isinstance(child, ast.ImportFrom) and (child.module or "").split(".")[0] == "torch")
        for child in ast.walk(node)
    )


def top_level_torch_import(tree: ast.Module) -> bool:
    """Tell whether a module imports torch, or skips itself without it, at module level."""
    for statement in tree.body:
        if isinstance(statement, ast.Import | ast.ImportFrom) and imports_torch(statement):
            return True
        for call in ast.walk(statement):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "importorskip"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "torch"
            ):
                return True
    return False


def local_imports(tree: ast.Module) -> set[str]:
    """Return the names of the modules that a module imports at its top level."""
    found: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            found |= {alias.name.split(".")[0] for alias in statement.names}
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            found.add(statement.module.split(".")[0])
    return found


def parse_tests() -> dict[str, ast.Module]:
    """Parse every Python file below tests, keyed by its relative path without the suffix."""
    return {
        path.relative_to(TESTS).with_suffix("").as_posix(): ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for path in sorted(TESTS.rglob("*.py"))
    }


def torch_modules(trees: dict[str, ast.Module]) -> set[str]:
    """Return the modules that need torch: those that import it and those that import them."""
    needing = {stem for stem, tree in trees.items() if top_level_torch_import(tree)}
    changed = True
    while changed:
        changed = False
        for stem, tree in trees.items():
            imported = {path.rsplit("/", 1)[-1] for path in needing}
            if stem not in needing and local_imports(tree) & imported:
                needing.add(stem)
                changed = True
    return needing


def pytestmark_names(tree: ast.Module) -> set[str]:
    """Collect every name used by the ``pytestmark`` assignments of a module."""
    return {
        node.id
        for statement in pytestmark_statements(tree)
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }


def functions_of_tests(
    tree: ast.Module,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """List test functions with their containing class and function names."""
    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def collect(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            parents = (*parents, node.name)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
                "test"
            ):
                found.append(("::".join(parents), node))
        for child in ast.iter_child_nodes(node):
            collect(child, parents)

    collect(tree)
    return found


def encoder_names(tree: ast.Module) -> frozenset[str]:
    """List the names under which a module binds the ``Encoder`` class, aliases included."""
    names = {"Encoder"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names if a.name == "Encoder"}
    return frozenset(names)


def uses_encoder(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    names: frozenset[str] = frozenset({"Encoder"}),
    *,
    count_builds: bool = True,
) -> bool:
    """Tell whether a test builds an ``Encoder``, imports torch, or takes a MiniLM fixture.

    The encoder counts when it is called by a name that the module bound to it (aliases
    included) or as an attribute, as in ``embeddings.Encoder(...)``; with ``count_builds``
    false, building it does not count.
    """
    builds = count_builds and any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in names)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Encoder")
        )
        for node in ast.walk(function)
    )
    fixtures = any("minilm" in argument.arg for argument in function.args.args)
    return builds or fixtures or imports_torch(function)


def statement_is(node: ast.stmt, source: str) -> bool:
    """Compare statement syntax, ignoring formatting and source positions."""
    return ast.dump(node) == ast.dump(ast.parse(source).body[0])


def replaces_load(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Recognize the direct stand-in setup, one embed call and result assertions.

    This deliberately accepts a small syntax pattern, not arbitrary ways to patch Python.
    """
    body = list(function.body)
    if not isinstance(function, ast.FunctionDef) or function.decorator_list or not body:
        return False
    args = function.args
    if (
        [argument.arg for argument in args.args] != ["monkeypatch"]
        or args.posonlyargs
        or args.kwonlyargs
        or args.defaults
        or args.vararg
        or args.kwarg
    ):
        return False
    tokenizer = "WordTokenizer()"
    if statement_is(body[0], "tokenizer = RecordingTokenizer()"):
        body.pop(0)
        tokenizer = "tokenizer"
    expected = [
        f"monkeypatch.setattr(Encoder, '_load', lambda self: runtime_with({tokenizer}))",
        "monkeypatch.setattr(Encoder, '_pool_fragments', one_vector_per_chunk)",
    ]
    if len(body) < 3 or not all(
        statement_is(node, text) for node, text in zip(body[:2], expected, strict=True)
    ):
        return False
    embed = "Encoder(SPEC).embed(TEXTS, IDS)"
    if not any(statement_is(body[2], text) for text in (embed, f"result = {embed}")):
        return False
    if not all(isinstance(node, ast.Assert) for node in body[3:]):
        return False
    for assertion in body[3:]:
        for node in ast.walk(assertion):
            if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name)
                and node.func.id in {"all", "sorted", "documented_texts_hash"}
            ):
                return False
            if isinstance(node, ast.Attribute) and ast.unparse(node) not in {
                "tokenizer.seen",
                "result.sidecar",
            }:
                return False
            if isinstance(node, ast.NamedExpr | ast.Lambda | ast.Await | ast.Yield | ast.YieldFrom):
                return False
    return True


def encoder_needs(
    trees: dict[str, ast.Module], load_replaced: frozenset[str] = LOAD_REPLACED_TESTS
) -> list[Needing]:
    """List every test module and test that needs the encoder, with its marker status.

    Building an encoder does not count for a test named in ``load_replaced`` (as
    ``<relative path>::<qualified test name>``) that uses the stand-in pattern; its imports of
    torch and its MiniLM fixtures still count, and so does building one in every other test.
    """
    torch_based = torch_modules(trees)
    found: list[Needing] = []
    for stem, tree in trees.items():
        module_marked = is_marked(module_marks(tree))
        tests = functions_of_tests(tree)
        module_name = stem.rsplit("/", 1)[-1]
        if stem in torch_based and module_name.startswith("test_"):
            found.append(Needing(f"{stem} (imports torch)", module_marked))
        if MINILM_NAMES & pytestmark_names(tree) and module_name.startswith("test_"):
            found.append(Needing(f"{stem} (skips on the MiniLM cache)", module_marked))
        for name, function in tests:
            where = f"{stem}::{name}"
            let_off = where in load_replaced and function in tree.body and replaces_load(function)
            if uses_encoder(function, encoder_names(tree), count_builds=not let_off):
                decorated = any(is_marked(marker_names(item)) for item in function.decorator_list)
                found.append(Needing(where, module_marked or decorated))
    return found


def test_every_test_that_imports_torch_or_builds_an_encoder_carries_the_embed_marker() -> None:
    needs = encoder_needs(parse_tests())
    assert len(needs) > 5
    unmarked = [need.where for need in needs if not need.marked]
    assert unmarked == []


def test_the_encoder_guard_sees_an_unmarked_torch_module_and_an_unmarked_encoder_test() -> None:
    sources = {
        "test_a": 'import pytest\ntorch = pytest.importorskip("torch")\n\ndef test_x(): ...\n',
        "test_b": "import pytest\nfrom test_a import x\n\ndef test_y(): ...\n",
        "test_c": "def test_z():\n    Encoder('m')\n",
        "test_d": (
            "import pytest\npytestmark = pytest.mark.embed\ndef test_w():\n    Encoder('m')\n"
        ),
        "test_e": "@needs_minilm\ndef test_v():\n    Encoder('m')\n",
        "test_f": "def test_u(minilm_files): ...\n",
        "test_g": ("from copernicus.embeddings import Encoder as E\n\ndef test_t():\n    E('m')\n"),
        "test_h": "import copernicus as cop\n\ndef test_s():\n    cop.Encoder('m')\n",
        "test_i": "def test_r():\n    import torch\n",
        "test_j": "def test_q():\n    from torch import nn\n",
        "test_k": "def helper():\n    import torch\n\ndef test_p(): ...\n",
    }
    trees = {name: ast.parse(text) for name, text in sources.items()}
    needs = {need.where: need.marked for need in encoder_needs(trees)}
    assert needs == {
        "test_a (imports torch)": False,
        "test_b (imports torch)": False,
        "test_c::test_z": False,
        "test_d::test_w": True,
        "test_e::test_v": True,
        "test_f::test_u": False,
        "test_g::test_t": False,
        "test_h::test_s": False,
        "test_i::test_r": False,
        "test_j::test_q": False,
    }


def test_only_the_direct_stand_in_setup_is_exempted() -> None:
    patch = "monkeypatch.setattr(Encoder, '_load', lambda self: runtime_with(WordTokenizer()))\n"
    pool = "monkeypatch.setattr(Encoder, '_pool_fragments', one_vector_per_chunk)\n"
    embed = "Encoder(SPEC).embed(TEXTS, IDS)\n"
    good = patch + pool + embed
    bodies = {
        "patched": good,
        "unlisted": good,
        "missing": embed,
        "other": patch.replace("setattr(Encoder,", "setattr(Other,") + pool + embed,
        "instance": "other = Encoder(SPEC)\nother._load = lambda: None\n" + embed,
        "dead": "if False:\n    " + patch + pool + embed,
        "nested": "def unused():\n    " + patch + pool + embed,
        "late": embed + patch + pool,
        "undone": patch + pool + "monkeypatch.undo()\n" + embed,
        "noop": "monkeypatch.setattr(Encoder, '_load', Encoder._load)\n" + pool + embed,
        "delegate": patch.replace("runtime_with(WordTokenizer())", "original(self)") + pool + embed,
        "extra": good + embed,
        "assert_load": good + "assert Encoder(SPEC)._load()\n",
        "torch": good + "import torch\n",
    }
    source = "".join(
        f"def test_{name}(monkeypatch):\n" + "".join(f"    {line}\n" for line in body.splitlines())
        for name, body in bodies.items()
    )
    listed = frozenset(f"test_m::test_{name}" for name in bodies if name != "unlisted")
    needs = encoder_needs({"test_m": ast.parse(source)}, listed)
    assert {need.where for need in needs} == {
        f"test_m::test_{name}" for name in bodies if name != "patched"
    }
    assert all(not need.marked for need in needs)


def test_the_exemption_does_not_follow_a_basename_or_a_class_method() -> None:
    name = "test_the_tokenizer_receives_each_text_as_given"
    body = (
        "    monkeypatch.setattr(Encoder, '_load', lambda self: runtime_with(WordTokenizer()))\n"
        "    monkeypatch.setattr(Encoder, '_pool_fragments', one_vector_per_chunk)\n"
        "    Encoder(SPEC).embed(TEXTS, IDS)\n"
    )
    source = f"def {name}(monkeypatch):\n{body}"
    method = "class TestOther:\n" + "".join(f"    {line}\n" for line in source.splitlines())
    trees = {
        "embeddings/test_encoder_texts": ast.parse(source + method),
        "other/test_encoder_texts": ast.parse(source),
    }
    assert {need.where for need in encoder_needs(trees)} == {
        f"embeddings/test_encoder_texts::TestOther::{name}",
        f"other/test_encoder_texts::{name}",
    }


def test_every_listed_test_exists_builds_an_encoder_and_replaces_its_load() -> None:
    listed = [
        (f"{stem}::{name}", tree, function)
        for stem, tree in parse_tests().items()
        for name, function in functions_of_tests(tree)
        if f"{stem}::{name}" in LOAD_REPLACED_TESTS
    ]
    assert len(listed) == len(LOAD_REPLACED_TESTS)
    assert {where for where, _, _ in listed} == LOAD_REPLACED_TESTS
    for where, tree, function in listed:
        assert function in tree.body, where
        assert uses_encoder(function, encoder_names(tree)), where
        assert replaces_load(function), where
