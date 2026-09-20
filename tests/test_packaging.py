"""Packaging metadata uses setuptools src layout with requirements.txt."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_uses_setuptools_src_layout_and_dynamic_deps() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    build = data.get("build-system", {})
    assert build.get("build-backend") == "setuptools.build_meta"

    setuptools = data.get("tool", {}).get("setuptools", {})
    find = setuptools.get("packages", {}).get("find", {})
    assert find.get("where") == ["src"]

    dynamic = data.get("project", {}).get("dynamic", [])
    assert "dependencies" in dynamic

    dep_file = data.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get(
        "dependencies", {}
    ).get("file")
    assert dep_file == ["requirements.txt"]
