"""Compose file safety checks (no runtime Docker required)."""

from __future__ import annotations

import re
from pathlib import Path


def test_postgres_host_port_bound_to_localhost_only() -> None:
    compose_path = Path(__file__).resolve().parent.parent / "docker-compose.yml"
    text = compose_path.read_text(encoding="utf-8")

    postgres_block = re.search(
        r"^\s{2}postgres:\n(.*?)(?=^\s{2}\w|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert postgres_block is not None
    block = postgres_block.group(1)

    port_lines = [line.strip() for line in block.splitlines() if line.strip().startswith("- ")]
    postgres_port_lines = [
        line
        for line in port_lines
        if "5432" in line and "pg_isready" not in line and "CMD" not in line
    ]
    if not postgres_port_lines:
        return

    for line in postgres_port_lines:
        assert "127.0.0.1:" in line or "localhost:" in line, (
            f"postgres port must bind to localhost only, got {line!r}"
        )


def test_compose_defaults_to_gemini_2_5_flash() -> None:
    compose_path = Path(__file__).resolve().parent.parent / "docker-compose.yml"

    assert "GEMINI_MODEL: ${GEMINI_MODEL:-gemini-2.5-flash}" in compose_path.read_text(
        encoding="utf-8"
    )
