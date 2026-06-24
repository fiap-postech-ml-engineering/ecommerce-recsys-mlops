from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from src.api.app import app, home
from src.api.middleware import REQUEST_ID_HEADER
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


def test_response_includes_request_id_header():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]


def test_response_reuses_client_provided_request_id():
    client = TestClient(app)

    response = client.get("/", headers={REQUEST_ID_HEADER: "my-custom-id"})

    assert response.headers[REQUEST_ID_HEADER] == "my-custom-id"
