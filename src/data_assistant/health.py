"""Health checks for runtime dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Self


class _Connection(Protocol):
    def execute(self, query: str) -> Any: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class PostgresHealthStatus:
    ok: bool
    detail: str


ConnectFactory = Callable[[], _Connection]

_HEALTH_QUERY = "SELECT 1"
POSTGRES_HEALTH_OK = "PostgreSQL is reachable"
POSTGRES_HEALTH_UNAVAILABLE = "PostgreSQL is unavailable"


def check_postgres_health(*, connect: ConnectFactory) -> PostgresHealthStatus:
    """Verify PostgreSQL accepts a simple query using an injected connection factory."""
    try:
        with connect() as conn:
            conn.execute(_HEALTH_QUERY)
    except Exception:  # noqa: BLE001 — map any failure to a stable public message
        return PostgresHealthStatus(ok=False, detail=POSTGRES_HEALTH_UNAVAILABLE)
    return PostgresHealthStatus(ok=True, detail=POSTGRES_HEALTH_OK)


def default_postgres_connect(database_url: str) -> ConnectFactory:
    """Return a psycopg connection factory for the given URL (lazy import)."""
    import psycopg

    def connect() -> psycopg.Connection[Any]:
        return psycopg.connect(database_url)

    return connect
