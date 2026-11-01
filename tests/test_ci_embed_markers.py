"""Every test that needs the encoder carries the embed marker."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ci_support import EMBED_MARKER, TESTS

EMBED_DECORATORS = frozenset({"needs_minilm"})
MINILM_NAMES = frozenset({"minilm_cached", "MINILM_AVAILABLE"})


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
    """Parse every python file below the tests folder, keyed by its stem."""
    return {
        path.stem: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(TESTS.rglob("*.py"))
    }


def torch_modules(trees: dict[str, ast.Module]) -> set[str]:
    """Return the modules that need torch: those that import it and those that import them."""
    needing = {stem for stem, tree in trees.items() if top_level_torch_import(tree)}
    changed = True
    while changed:
        changed = False
        for stem, tree in trees.items():
            if stem not in needing and local_imports(tree) & needing:
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


def functions_of_tests(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """List the test functions of a module, methods of test classes included."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test")
    ]


def encoder_names(tree: ast.Module) -> frozenset[str]:
    """List the names under which a module binds the ``Encoder`` class, aliases included."""
    names = {"Encoder"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names if a.name == "Encoder"}
    return frozenset(names)


def uses_encoder(
    function: ast.FunctionDef | ast.AsyncFunctionDef, names: frozenset[str] = frozenset({"Encoder"})
) -> bool:
    """Tell whether a test builds an ``Encoder``, imports torch, or takes a MiniLM fixture.

    The encoder counts when it is called by a name that the module bound to it (aliases
    included) or as an attribute, as in ``embeddings.Encoder(...)``.
    """
    builds = any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in names)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Encoder")
        )
        for node in ast.walk(function)
    )
    fixtures = any("minilm" in argument.arg for argument in function.args.args)
    return builds or fixtures or imports_torch(function)


def encoder_needs(trees: dict[str, ast.Module]) -> list[Needing]:
    """List every test module and test that needs the encoder, with its marker status."""
    torch_based = torch_modules(trees)
    found: list[Needing] = []
    for stem, tree in trees.items():
        module_marked = is_marked(module_marks(tree))
        tests = functions_of_tests(tree)
        if stem in torch_based and stem.startswith("test_"):
            found.append(Needing(f"{stem} (imports torch)", module_marked))
        if MINILM_NAMES & pytestmark_names(tree) and stem.startswith("test_"):
            found.append(Needing(f"{stem} (skips on the MiniLM cache)", module_marked))
        for function in tests:
            if uses_encoder(function, encoder_names(tree)):
                decorated = any(is_marked(marker_names(item)) for item in function.decorator_list)
                found.append(Needing(f"{stem}::{function.name}", module_marked or decorated))
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
