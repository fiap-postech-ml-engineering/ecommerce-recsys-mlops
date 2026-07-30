from unittest.mock import patch

from pydantic import SecretStr, ValidationError
import pytest
from scripts.validate_env import _missing_or_placeholder, main

from src.config import Settings


@pytest.mark.unit
def test_missing_or_placeholder_flags_none_values():
    settings = Settings(DATABRICKS_HOST=None, DATABRICKS_TOKEN=None)

    result = _missing_or_placeholder(settings, ("DATABRICKS_HOST", "DATABRICKS_TOKEN"))

    assert result == ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]


@pytest.mark.unit
def test_missing_or_placeholder_flags_placeholder_secret():
    settings = Settings(
        DATABRICKS_HOST="https://workspace.cloud.databricks.com",
        DATABRICKS_TOKEN=SecretStr("__set_me__"),
    )

    result = _missing_or_placeholder(settings, ("DATABRICKS_HOST", "DATABRICKS_TOKEN"))

    assert result == ["DATABRICKS_TOKEN"]


@pytest.mark.unit
def test_missing_or_placeholder_returns_empty_when_all_configured():
    settings = Settings(
        DATABRICKS_HOST="https://workspace.cloud.databricks.com",
        DATABRICKS_TOKEN=SecretStr("a-real-token"),
    )

    result = _missing_or_placeholder(settings, ("DATABRICKS_HOST", "DATABRICKS_TOKEN"))

    assert result == []


@pytest.mark.unit
def test_main_exits_with_error_when_settings_invalid():
    with (
        patch(
            "scripts.validate_env._load_settings",
            side_effect=ValidationError.from_exception_data("Settings", []),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1


@pytest.mark.unit
def test_main_runs_without_raising_when_settings_valid():
    with patch("scripts.validate_env._load_settings", return_value=Settings()):
        main()
