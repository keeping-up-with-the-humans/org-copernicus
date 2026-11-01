import copy
import pickle

import pytest

import copernicus
from copernicus import errors
from copernicus.assets import asset_path


def test_version() -> None:
    assert copernicus.__version__ == "0.1.0"


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
