"""Runtime settings, read once from MEDIA_GEN_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOME = Path.home() / "Library" / "Application Support" / "media-gen"


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _read_secret(path: Path | None) -> str | None:
    if path is None:
        return None
    value = path.read_text().strip()
    if not value:
        raise ValueError(f"{path} is empty")
    return value


@dataclass(frozen=True)
class Settings:
    home: Path
    host: str = "127.0.0.1"
    port: int = 9112
    # When set, every /v1 request needs `Authorization: Bearer <key>`.
    api_key: str | None = None
    # oMLX on the same Mac: media-gen unloads its idle models to make room.
    omlx_url: str | None = None
    omlx_api_key: str | None = None
    # oMLX models used within this many seconds are never unloaded.
    omlx_busy_seconds: int = 300
    # Memory kept free beyond a model's own floor.
    memory_reserve_gib: int = 16
    # A loaded model is released after this long without work.
    idle_unload_seconds: int = 600
    video_retention_hours: int = 24
    max_queued_jobs: int = 16
    download_limit_mbps: int = 200

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.home / "outputs"

    @property
    def database(self) -> Path:
        return self.home / "jobs.sqlite3"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            home=_path("MEDIA_GEN_HOME") or DEFAULT_HOME,
            host=os.environ.get("MEDIA_GEN_HOST", "127.0.0.1"),
            port=_int("MEDIA_GEN_PORT", 9112),
            api_key=_read_secret(_path("MEDIA_GEN_API_KEY_FILE")),
            omlx_url=os.environ.get("MEDIA_GEN_OMLX_URL") or None,
            omlx_api_key=_read_secret(_path("MEDIA_GEN_OMLX_API_KEY_FILE")),
            omlx_busy_seconds=_int("MEDIA_GEN_OMLX_BUSY_SECONDS", 300),
            memory_reserve_gib=_int("MEDIA_GEN_MEMORY_RESERVE_GIB", 16),
            idle_unload_seconds=_int("MEDIA_GEN_IDLE_UNLOAD_SECONDS", 600),
            video_retention_hours=_int("MEDIA_GEN_VIDEO_RETENTION_HOURS", 24),
            max_queued_jobs=_int("MEDIA_GEN_MAX_QUEUED_JOBS", 16),
            download_limit_mbps=_int("MEDIA_GEN_DOWNLOAD_LIMIT_MBPS", 200),
        )
