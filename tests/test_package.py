import ast
import copy
import pickle
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import copernicus
from copernicus import errors
from copernicus.assets import asset_path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 reads the file with the tomli backport.
    import tomli as tomllib

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
# The import name of a declared distribution, where the two differ.
IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
# Declared without an import of the package: pandas loads it to read and write parquet files.
LOADED_BY_PANDAS = {"pyarrow"}
# The folders whose modules are checked, and whether a module there may import the modules of its
# own folder tree by their bare names: pytest puts the test folders on the path, and a script of
# the examples runs with its own folder on the path.
IMPORTING_FOLDERS = {"src/copernicus": False, "tests": True, "examples": True}
# A module of the standard library from Python 3.11 on. On Python 3.10, the oldest version that
# the library supports, the modules that import it fall back to the tomli backport of the dev extra.
LATER_STANDARD_LIBRARY = {"tomllib"}
# A module of the package that imports the packages of an extra when it is loaded, with that extra:
# an encoder loads it only after it has imported torch (copernicus.embeddings.encoder).
EXTRA_MODULES = {"src/copernicus/embeddings/pooling.py": "embed"}
# The fields of a statement that hold blocks of statements of their own.
BLOCK_FIELDS = ("body", "orelse", "finalbody")
NESTED_FIELDS = (*BLOCK_FIELDS, "handlers", "cases")


def imported_top_level_names(folder: Path) -> set[str]:
    """Return the top-level names that the modules below a folder import absolutely."""
    names: set[str] = set()
    for path in sorted(folder.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def requirement_names(requirements: list[str]) -> set[str]:
    """Return the lower-case distribution names that requirement strings name."""
    return {
        re.match(r"[A-Za-z0-9_.\-]+", requirement)[0].lower()  # type: ignore[index]
        for requirement in requirements
    }


def import_names(requirements: list[str]) -> set[str]:
    """Return the names under which the distributions of requirement strings are imported."""
    return {IMPORT_NAMES.get(name, name) for name in requirement_names(requirements)}


def child_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return the blocks of a statement that run when the module is loaded; a function has none."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    blocks = [getattr(statement, field) for field in BLOCK_FIELDS if getattr(statement, field, [])]
    for field in ("handlers", "cases"):
        blocks += [part.body for part in getattr(statement, field, [])]
    return blocks


def expression_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Yield a node and the nodes below it, without the body of a lambda, which runs later."""
    yield node
    for field, child in ast.iter_fields(node):
        if isinstance(node, ast.Lambda) and field == "body":
            continue
        for value in child if isinstance(child, list) else [child]:
            if isinstance(value, ast.AST):
                yield from expression_nodes(value)


def own_nodes(statement: ast.stmt) -> Iterator[ast.AST]:
    """Yield the nodes of a statement outside its blocks, such as its test or its value.

    The exception type of a handler and the guard of a case are among them: they are evaluated
    with the statement, not as part of the handler's or the case's block.
    """
    for field, child in ast.iter_fields(statement):
        if field not in NESTED_FIELDS:
            for value in child if isinstance(child, list) else [child]:
                if isinstance(value, ast.AST):
                    yield from expression_nodes(value)
    heads = [handler.type for handler in getattr(statement, "handlers", [])]
    heads += [case.guard for case in getattr(statement, "cases", [])]
    for head in heads:
        if head is not None:
            yield from expression_nodes(head)


def import_module_package(node: ast.AST) -> str | None:
    """Return the package of an ``importlib.import_module`` call; ``None`` for a relative name.

    The scan reads the name from the source, so a name that the call computes is refused.
    """
    if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "importlib.import_module"):
        return None
    name = node.args[0] if node.args else None
    assert isinstance(name, ast.Constant) and isinstance(name.value, str), ast.unparse(node)
    return None if name.value.startswith(".") else name.value.split(".")[0]


def importorskip_package(node: ast.AST) -> str | None:
    """Return the package of a ``pytest.importorskip`` call, or ``None`` for any other node."""
    if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "pytest.importorskip"):
        return None
    name = node.args[0]
    assert isinstance(name, ast.Constant) and isinstance(name.value, str), ast.unparse(node)
    return name.value.split(".")[0]


def absolute_imports(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Return the top-level names that an import statement imports; none for a relative one."""
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    return [node.module.split(".")[0]] if node.level == 0 and node.module else []


def statement_skip(statement: ast.stmt) -> str | None:
    """Return the package of a ``pytest.importorskip`` that is the whole value of a statement."""
    if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)) and statement.value:
        return importorskip_package(statement.value)
    return None


def block_imports(statements: list[ast.stmt], skipped: frozenset[str]) -> list[tuple[int, str]]:
    """List the imports that a block of statements runs, as line and top-level name.

    A ``pytest.importorskip`` that a statement of the block runs exempts the imports of its
    package in the statements after it, also in their blocks, but not after its own block ends.
    """
    found: list[tuple[int, str]] = []
    for statement in statements:
        names = [
            (node.lineno, import_module_package(node))
            for node in own_nodes(statement)
            if isinstance(node, ast.Call)
        ]
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            names += [(statement.lineno, name) for name in absolute_imports(statement)]
        found += [(line, name) for line, name in names if name is not None and name not in skipped]
        for block in child_blocks(statement):
            found += block_imports(block, skipped)
        package = statement_skip(statement)
        if package is not None:
            skipped = skipped | {package}
    return found


def import_time_imports(path: Path) -> list[tuple[int, str]]:
    """List the imports that run when a module is loaded, as line and top-level name.

    An import counts wherever it stands outside a function, also inside a ``try``, an ``if`` or a
    class body, and so does an ``importlib.import_module`` call with an absolute name, unless a
    ``pytest.importorskip`` of its package runs before it (see `block_imports`).
    """
    return block_imports(ast.parse(path.read_text(encoding="utf-8")).body, frozenset())


def test_version() -> None:
    assert copernicus.__version__ == "0.2.0"


def test_every_declared_dependency_is_imported_by_a_module_of_the_package() -> None:
    parsed = tomllib.loads((LIBRARY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = requirement_names(parsed["project"]["dependencies"])
    imported = imported_top_level_names(LIBRARY_ROOT / "src" / "copernicus")
    unused = sorted(
        name for name in declared - LOADED_BY_PANDAS if IMPORT_NAMES.get(name, name) not in imported
    )
    assert unused == []


def test_no_module_imports_a_package_at_import_time_that_the_library_does_not_declare() -> None:
    """Every package that a module imports outside a function is one the library accounts for.

    That is the standard library, the library itself, a module of the module's own folder tree,
    or a distribution of the dependencies or of the dev extra; a module of ``EXTRA_MODULES`` may
    also import the distributions of its extra.
    """
    parsed = tomllib.loads((LIBRARY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project, extras = parsed["project"], parsed["project"]["optional-dependencies"]
    allowed = (
        import_names([*project["dependencies"], *extras["dev"]])
        | sys.stdlib_module_names
        | LATER_STANDARD_LIBRARY
        | {"copernicus"}
    )
    undeclared: list[str] = []
    for folder, local_imports in IMPORTING_FOLDERS.items():
        paths = sorted((LIBRARY_ROOT / folder).rglob("*.py"))
        local = {path.stem for path in paths} if local_imports else set()
        for path in paths:
            relative = path.relative_to(LIBRARY_ROOT).as_posix()
            extra = (
                import_names(extras[EXTRA_MODULES[relative]])
                if relative in EXTRA_MODULES
                else set()
            )
            undeclared += [
                f"{relative}:{line} {name}"
                for line, name in import_time_imports(path)
                if name not in allowed | local | extra
            ]
    assert undeclared == []


SKIP = 'pytest.importorskip("missing_package")'
SCANNED_SOURCES = {
    "a literal import_module call": (
        'import importlib\nimportlib.import_module("missing_package.part")\n',
        [(1, "importlib"), (2, "missing_package")],
    ),
    "an import_module call inside a function": (
        'import importlib\ndef load():\n    return importlib.import_module("missing_package")\n',
        [(1, "importlib")],
    ),
    "a relative import_module call": (
        'import importlib\nimportlib.import_module(".sibling", "package")\n',
        [(1, "importlib")],
    ),
    "a skip inside an if before the block ends": (
        f"if False:\n    {SKIP}\nimport missing_package\n",
        [(3, "missing_package")],
    ),
    "a skip inside a try before the block ends": (
        f"try:\n    {SKIP}\nexcept ImportError:\n    pass\nimport missing_package\n",
        [(5, "missing_package")],
    ),
    "a skip inside a with before the block ends": (
        f'with open("x") as handle:\n    {SKIP}\nimport missing_package\n',
        [(3, "missing_package")],
    ),
    "a skip followed by an import in its own block": (
        f"if True:\n    {SKIP}\n    import missing_package\n",
        [],
    ),
    "an unconditional skip followed by an import in a later block": (
        f"{SKIP}\nif True:\n    import missing_package\n",
        [],
    ),
    "an import before its skip": (
        f"import missing_package\n{SKIP}\n",
        [(1, "missing_package")],
    ),
    "an import_module call in the exception type of a handler": (
        "import importlib\ntry:\n    pass\n"
        'except importlib.import_module("missing_package").Error:\n    pass\n',
        [(1, "importlib"), (4, "missing_package")],
    ),
    "an import_module call in the guard of a case": (
        "import importlib\nmatch 1:\n"
        '    case _ if importlib.import_module("missing_package"):\n'
        "        pass\n",
        [(1, "importlib"), (3, "missing_package")],
    ),
}


@pytest.mark.parametrize("case", list(SCANNED_SOURCES))
def test_import_time_imports_reads_import_module_calls_and_where_a_skip_reaches(
    case: str, tmp_path: Path
) -> None:
    source, expected = SCANNED_SOURCES[case]
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    assert import_time_imports(path) == expected


def test_import_time_imports_refuses_an_import_module_call_whose_name_it_cannot_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "module.py"
    path.write_text('import importlib\nNAME = "x"\nimportlib.import_module(NAME)\n', "utf-8")
    with pytest.raises(AssertionError, match="import_module"):
        import_time_imports(path)


def test_error_hierarchy() -> None:
    for name in [
        "RubricValidationError",
        "TextTableError",
        "BackendError",
        "PromptContractError",
        "EvidenceContractError",
        "StoreError",
        "EstimatorError",
        "NotApplicableError",
        "RunFailureCeilingError",
        "SettingsError",
        "ConfigurationError",
    ]:
        assert name in errors.__all__
        assert issubclass(getattr(errors, name), errors.CopernicusError)
    assert issubclass(errors.SettingsError, ValueError)
    err = errors.BackendError("no json", attempts=3, last_output="xyz")
    assert err.attempts == 3 and err.last_output == "xyz" and "no json" in str(err)


def test_asset_path_missing_is_loud() -> None:
    with pytest.raises(FileNotFoundError, match="does_not_exist.txt"):
        asset_path("does_not_exist.txt")


def test_asset_path_existing_file() -> None:
    assert asset_path("rubric/schema.json").exists()


def test_asset_path_refuses_paths_outside_the_package() -> None:
    with pytest.raises(FileNotFoundError, match="outside the package"):
        asset_path("../../pyproject.toml")
    with pytest.raises(FileNotFoundError, match="outside the package"):
        asset_path("/etc/hosts")


def test_run_failure_ceiling_error_survives_pickling_and_copying() -> None:
    summary = {"n_texts": 10}
    error = errors.RunFailureCeilingError("12 of 30 attribute calls failed", summary)  # type: ignore[arg-type]
    for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error)):
        assert isinstance(clone, errors.RunFailureCeilingError)
        assert str(clone) == "12 of 30 attribute calls failed"
        assert clone.summary == summary
