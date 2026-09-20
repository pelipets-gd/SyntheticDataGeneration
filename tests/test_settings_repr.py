"""Tests that Settings string forms do not leak secrets."""

from __future__ import annotations

import pytest

from data_assistant.settings import load_settings


def test_settings_repr_redacts_database_url_and_langfuse_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
    monkeypatch.setenv("DATABASE_URL", "postgresql://dbuser:dbpass@postgres:5432/data_assistant")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "lf-secret-value")

    settings = load_settings()
    rendered = repr(settings)

    assert "dbpass" not in rendered
    assert "dbuser:dbpass" not in rendered
    assert "lf-secret-value" not in rendered
    assert "my-gcp-project" in rendered


def test_settings_str_redacts_database_url_and_langfuse_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
    monkeypatch.setenv("DATABASE_URL", "postgresql://dbuser:dbpass@postgres:5432/data_assistant")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "lf-secret-value")

    settings = load_settings()
    rendered = str(settings)

    assert "dbpass" not in rendered
    assert "lf-secret-value" not in rendered
