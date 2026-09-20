"""Tests for PostgreSQL health checks."""

from __future__ import annotations

from typing import Any

import pytest

from data_assistant.health import PostgresHealthStatus, check_postgres_health


class _HealthyConnection:
    def execute(self, _query: str) -> None:
        return None

    def __enter__(self) -> _HealthyConnection:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class _FailingConnection:
    def execute(self, _query: str) -> None:
        raise RuntimeError("connection refused")

    def __enter__(self) -> _FailingConnection:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def test_check_postgres_health_ok_when_select_one_succeeds() -> None:
    def connect() -> _HealthyConnection:
        return _HealthyConnection()

    status = check_postgres_health(connect=connect)

    from data_assistant.health import POSTGRES_HEALTH_OK

    assert status == PostgresHealthStatus(ok=True, detail=POSTGRES_HEALTH_OK)


def test_check_postgres_health_not_ok_when_query_fails() -> None:
    def connect() -> _FailingConnection:
        return _FailingConnection()

    status = check_postgres_health(connect=connect)

    assert status.ok is False
    assert status.detail == "PostgreSQL is unavailable"
    assert "connection refused" not in status.detail


def test_check_postgres_health_not_ok_when_connect_raises() -> None:
    def connect() -> _HealthyConnection:
        raise OSError("password authentication failed for user secret")

    status = check_postgres_health(connect=connect)

    assert status.ok is False
    assert status.detail == "PostgreSQL is unavailable"
    assert "password" not in status.detail
    assert "secret" not in status.detail
