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


@pytest.mark.unit
def test_main_skips_databricks_check_when_tracking_uri_is_local():
    settings = Settings(MLFLOW_TRACKING_URI="local", DATABRICKS_HOST=None, DATABRICKS_TOKEN=None)
    with patch("scripts.validate_env._load_settings", return_value=settings):
        main()


@pytest.mark.unit
def test_main_warns_when_databricks_tracking_uri_missing_credentials():
    settings = Settings(
        MLFLOW_TRACKING_URI="databricks", DATABRICKS_HOST=None, DATABRICKS_TOKEN=None
    )
    with (
        patch("scripts.validate_env._load_settings", return_value=settings),
        patch("scripts.validate_env.logger") as mock_logger,
    ):
        main()

    formatted_messages = [
        call.args[0] % call.args[1:] for call in mock_logger.warning.call_args_list
    ]
    assert any("DATABRICKS_HOST" in msg for msg in formatted_messages)
