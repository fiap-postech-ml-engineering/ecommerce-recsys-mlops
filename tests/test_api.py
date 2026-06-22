from pydantic import ValidationError
import pytest

from src.api.app import app, home
from src.config import Settings


def test_home_returns_service_status():
    assert home() == {
        "service": "E-commerce RecSys API",
        "version": "0.1.0",
        "env": "development",
        "status": "ok",
    }


def test_app_metadata_uses_model_version():
    assert app.title == "E-commerce RecSys API"
    assert app.version == "0.1.0"


def test_settings_reject_invalid_app_env():
    with pytest.raises(ValidationError):
        Settings(APP_ENV="local")
