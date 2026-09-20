"""Tests for application settings loaded from the environment."""

from __future__ import annotations

import os

import pytest

from data_assistant.settings import Settings, load_settings


def test_load_settings_requires_google_cloud_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/app")

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        load_settings()


def test_load_settings_applies_defaults_and_optional_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/data_assistant")
    monkeypatch.setenv("VERTEX_AI_LOCATION", "europe-west1")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash-001")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    settings = load_settings()

    assert isinstance(settings, Settings)
    assert settings.google_cloud_project == "my-gcp-project"
    assert settings.database_url == "postgresql://u:p@db:5432/data_assistant"
    assert settings.vertex_ai_location == "europe-west1"
    assert settings.gemini_model == "gemini-2.0-flash-001"
    assert settings.langfuse_public_key == "pk-test"
    assert settings.langfuse_secret_key == "sk-test"
    assert settings.langfuse_host == "https://cloud.langfuse.com"


def test_load_settings_defaults_to_gemini_2_5_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/data_assistant")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert load_settings().gemini_model == "gemini-2.5-flash"


@pytest.mark.parametrize(
    "model",
    [
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.5-flash",
        "gemini-3-pro-preview",
    ],
)
def test_load_settings_accepts_supported_gemini_models(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/data_assistant")
    monkeypatch.setenv("GEMINI_MODEL", model)

    assert load_settings().gemini_model == model


@pytest.mark.parametrize(
    "model",
    [
        "gemini-1.5-flash",
        "gemini-1.0-pro",
        "text-bison",
        "gemini-latest",
        "gemini-unknown-flash",
        "publishers/google/models/gemini-2.5-flash",
    ],
)
def test_load_settings_rejects_unsupported_gemini_models(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/data_assistant")
    monkeypatch.setenv("GEMINI_MODEL", model)

    with pytest.raises(ValueError, match="GEMINI_MODEL.*Gemini 2.0\\+"):
        load_settings()


def test_settings_module_does_not_read_env_on_import() -> None:
    """Importing settings must not require GOOGLE_CLOUD_PROJECT."""
    original = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
    try:
        import importlib

        import data_assistant.settings as settings_module

        importlib.reload(settings_module)
    finally:
        if original is not None:
            os.environ["GOOGLE_CLOUD_PROJECT"] = original

