"""Application settings loaded explicitly from the process environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


_GEMINI_MODEL_PATTERN = re.compile(
    r"^gemini-(?P<major>\d+)(?:\.(?P<minor>\d+))?-[a-z0-9]+(?:-[a-z0-9]+)*$"
)


@dataclass(frozen=True, slots=True)
class Settings:
    google_cloud_project: str
    database_url: str
    vertex_ai_location: str = "us-central1"
    gemini_model: str = "gemini-2.5-flash"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    def __post_init__(self) -> None:
        match = _GEMINI_MODEL_PATTERN.fullmatch(self.gemini_model)
        if match is None:
            raise ValueError(
                "GEMINI_MODEL must be a Gemini 2.0+ model id such as "
                "'gemini-2.5-flash'"
            )
        major = int(match.group("major"))
        minor = match.group("minor")
        if major < 2 or (major == 2 and minor is None):
            raise ValueError(
                "GEMINI_MODEL must be a Gemini 2.0+ model id such as "
                "'gemini-2.5-flash'"
            )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def __repr__(self) -> str:
        return _settings_repr(self)

    def __str__(self) -> str:
        return _settings_repr(self)


def _settings_repr(settings: Settings) -> str:
    secret = "***" if settings.langfuse_secret_key else None
    return (
        f"Settings("
        f"google_cloud_project={settings.google_cloud_project!r}, "
        f"database_url='***', "
        f"vertex_ai_location={settings.vertex_ai_location!r}, "
        f"gemini_model={settings.gemini_model!r}, "
        f"langfuse_public_key={settings.langfuse_public_key!r}, "
        f"langfuse_secret_key={secret!r}, "
        f"langfuse_host={settings.langfuse_host!r})"
    )


def load_settings(environ: os._Environ[str] | None = None) -> Settings:
    """Build settings from environment variables (call at runtime, not import)."""
    env = os.environ if environ is None else environ

    project = env.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT is required")

    database_url = env.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL is required")

    return Settings(
        google_cloud_project=project,
        database_url=database_url,
        vertex_ai_location=(
            env.get("VERTEX_AI_LOCATION", "us-central1").strip() or "us-central1"
        ),
        gemini_model=(
            env.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
            or "gemini-2.5-flash"
        ),
        langfuse_public_key=_optional(env, "LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_optional(env, "LANGFUSE_SECRET_KEY"),
        langfuse_host=_optional(env, "LANGFUSE_HOST"),
    )


def _optional(env: os._Environ[str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
