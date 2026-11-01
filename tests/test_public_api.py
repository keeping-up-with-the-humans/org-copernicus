import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import copernicus

EXPECTED_NAMES = [
    "load_rubric",
    "Rubric",
    "TextTable",
    "OpenAICompatibleBackend",
    "ModelCapabilities",
    "VLLMBackend",
    "FakeBackend",
    "ScriptedResponder",
    "cue_responder",
    "TwoStageConfig",
    "OneStageConfig",
    "CodingSettings",
    "CodingSummary",
    "code_texts",
    "export_scores",
    "export_construct_scores",
    "export_text_scores",
    "export_evidence",
    "export_provenance",
    "merge_stores",
    "Encoder",
    "EncoderSpec",
    "ENCODERS",
    "EmbeddingSet",
    "GeneratorSettings",
    "generate_references",
    "ReferenceSet",
    "ReferenceEmbeddings",
    "embed_references",
    "SimilaritySettings",
    "score_similarity",
    "run_grid",
    "PAPER_GRID",
    "score_hybrid",
    "validity_panel",
    "PanelSettings",
    "ValidityPanel",
    "association",
    "score_errors",
    "cronbach_alpha",
    "reliability_ceiling",
    "cluster_bootstrap_ci",
    "BaselineSettings",
    "run_ridge_baseline",
    "curves",
    "CorrectionData",
    "draw_human_sample",
    "plan_sample_size",
    "get_correction",
    "list_corrections",
    "cross_fit_predictions",
    "pseudo_values",
    "mean_ppi",
    "group_difference_ppi",
    "ols_outcome_ppi",
    "ols_predictor_ppi",
    "moment_subtraction",
    "simex",
    "oriv",
    "repeated_study",
    "run_battery",
    "summarize_battery",
    "BatterySettings",
    "errors",
    "__version__",
]

HOME_MODULES = {
    "load_rubric": "copernicus.rubric.loader",
    "Rubric": "copernicus.rubric.model",
    "TextTable": "copernicus.data.table",
    "OpenAICompatibleBackend": "copernicus.llm.openai_compatible",
    "ModelCapabilities": "copernicus.llm.base",
    "VLLMBackend": "copernicus.llm.vllm_backend",
    "FakeBackend": "copernicus.llm.fake",
    "ScriptedResponder": "copernicus.llm.fake",
    "cue_responder": "copernicus.llm.fake",
    "TwoStageConfig": "copernicus.coders.two_stage_types",
    "OneStageConfig": "copernicus.coders.one_stage",
    "CodingSettings": "copernicus.coders.runner_settings",
    "CodingSummary": "copernicus.coders.runner_settings",
    "code_texts": "copernicus.coders.runner",
    "export_scores": "copernicus.coders.results",
    "export_construct_scores": "copernicus.coders.results",
    "export_text_scores": "copernicus.coders.results",
    "export_evidence": "copernicus.coders.results",
    "export_provenance": "copernicus.coders.results",
    "merge_stores": "copernicus.coders.store",
    "Encoder": "copernicus.embeddings.encoder",
    "EncoderSpec": "copernicus.embeddings.registry",
    "ENCODERS": "copernicus.embeddings.registry",
    "EmbeddingSet": "copernicus.embeddings.sets",
    "GeneratorSettings": "copernicus.references.settings",
    "generate_references": "copernicus.references.generator",
    "ReferenceSet": "copernicus.references.reference_set",
    "ReferenceEmbeddings": "copernicus.similarity.references",
    "embed_references": "copernicus.similarity.references",
    "SimilaritySettings": "copernicus.similarity.settings",
    "score_similarity": "copernicus.similarity.scorer",
    "run_grid": "copernicus.similarity.grid",
    "PAPER_GRID": "copernicus.similarity.grid",
    "score_hybrid": "copernicus.hybrid.coder",
    "validity_panel": "copernicus.validity.panel",
    "PanelSettings": "copernicus.validity.panel",
    "ValidityPanel": "copernicus.validity.panel",
    "association": "copernicus.validity.metrics",
    "score_errors": "copernicus.validity.anatomy",
    "cronbach_alpha": "copernicus.validity.reliability",
    "reliability_ceiling": "copernicus.validity.reliability",
    "cluster_bootstrap_ci": "copernicus.validity.bootstrap",
    "BaselineSettings": "copernicus.baseline.runner",
    "run_ridge_baseline": "copernicus.baseline.runner",
    "curves": "copernicus.baseline.runner",
    "CorrectionData": "copernicus.corrections.data",
    "draw_human_sample": "copernicus.corrections.sampling",
    "plan_sample_size": "copernicus.corrections.planning",
    "get_correction": "copernicus.corrections.families._registry",
    "list_corrections": "copernicus.corrections.families._registry",
    "cross_fit_predictions": "copernicus.corrections.crossfit",
    "pseudo_values": "copernicus.corrections.families.pseudo",
    "mean_ppi": "copernicus.corrections.estimates",
    "group_difference_ppi": "copernicus.corrections.estimates",
    "ols_outcome_ppi": "copernicus.corrections.estimates_ols",
    "ols_predictor_ppi": "copernicus.corrections.estimates_ols",
    "moment_subtraction": "copernicus.corrections.estimates_classical",
    "simex": "copernicus.corrections.estimates_classical",
    "oriv": "copernicus.corrections.estimates_oriv",
    "repeated_study": "copernicus.corrections.study",
    "run_battery": "copernicus.corrections.battery",
    "summarize_battery": "copernicus.corrections.battery",
    "BatterySettings": "copernicus.corrections.battery_settings",
}


def all_packages() -> list[ModuleType]:
    packages = [copernicus]
    for info in pkgutil.walk_packages(copernicus.__path__, prefix="copernicus."):
        if info.ispkg:
            packages.append(importlib.import_module(info.name))
    return packages


def test_all_lists_exactly_the_public_names() -> None:
    assert copernicus.__all__ == EXPECTED_NAMES


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_every_exported_name_is_an_attribute(name: str) -> None:
    assert hasattr(copernicus, name)


@pytest.mark.parametrize("name", sorted(HOME_MODULES))
def test_exported_objects_are_the_ones_from_their_home_module(name: str) -> None:
    home = importlib.import_module(HOME_MODULES[name])
    assert getattr(copernicus, name) is getattr(home, name)


def test_errors_is_the_errors_module() -> None:
    assert copernicus.errors is importlib.import_module("copernicus.errors")
    assert "SettingsError" in copernicus.errors.__all__


def test_the_package_ships_its_type_marker() -> None:
    assert (Path(copernicus.__file__).parent / "py.typed").is_file()


def test_every_package_defines_a_resolvable_all() -> None:
    packages = all_packages()
    assert {package.__name__ for package in packages} >= {
        "copernicus",
        "copernicus.coders",
        "copernicus.data",
        "copernicus.embeddings",
        "copernicus.hybrid",
        "copernicus.llm",
        "copernicus.references",
        "copernicus.similarity",
        "copernicus.rubric",
        "copernicus.validity",
        "copernicus.baseline",
        "copernicus.cli",
        "copernicus.corrections",
        "copernicus.corrections.families",
    }
    for package in packages:
        exported = getattr(package, "__all__", None)
        assert isinstance(exported, list) and exported, package.__name__
        assert all(isinstance(name, str) for name in exported), package.__name__
        missing = [name for name in exported if not hasattr(package, name)]
        assert not missing, f"{package.__name__} exports unresolvable names {missing}"


def test_importing_the_package_does_not_import_vllm() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import copernicus, sys; print('vllm' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False"


def test_importing_the_package_does_not_import_torch_or_transformers() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import copernicus, sys; print('torch' in sys.modules, 'transformers' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False False"


def test_importing_the_package_does_not_import_sklearn() -> None:
    code = (
        "import copernicus, copernicus.cli.app, sys; "
        "print(any(name == 'sklearn' or name.startswith('sklearn.') for name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False"


def test_the_package_and_the_cli_import_none_of_the_heavy_libraries() -> None:
    heavy = ("torch", "transformers", "vllm", "sklearn", "tiktoken")
    code = (
        "import copernicus, copernicus.cli.app, sys; "
        f"print([name for name in {heavy!r} if any(module == name or "
        "module.startswith(name + '.') for module in sys.modules)])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"
