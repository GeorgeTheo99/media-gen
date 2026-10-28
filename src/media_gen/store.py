"""Durable video job records (SQLite). Image requests are synchronous and not stored."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    seconds INTEGER NOT NULL,
    size TEXT NOT NULL,
    params TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    expires_at INTEGER,
    error_code TEXT,
    error_message TEXT,
    output_path TEXT
);
CREATE INDEX IF NOT EXISTS videos_created ON videos (created_at);
"""

ACTIVE = ("queued", "in_progress")


@dataclass
class VideoRecord:
    id: str
    model: str
    status: str
    progress: int
    seconds: int
    size: str
    params: dict[str, Any]
    created_at: int
    started_at: int | None = None
    completed_at: int | None = None
    expires_at: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    output_path: str | None = None

    def to_api(self) -> dict[str, Any]:
        error = None
        if self.error_code:
            error = {"code": self.error_code, "message": self.error_message or ""}
        return {
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "expires_at": self.expires_at,
            "seconds": str(self.seconds),
            "size": self.size,
            "error": error,
            "remixed_from_video_id": None,
        }


class VideoStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> VideoRecord:
        data = dict(row)
        data["params"] = json.loads(data["params"])
        return VideoRecord(**data)

    def insert(self, record: VideoRecord) -> None:
        data = asdict(record)
        data["params"] = json.dumps(data["params"])
        columns = ", ".join(data)
        placeholders = ", ".join(f":{k}" for k in data)
        with self._lock:
            self._db.execute(f"INSERT INTO videos ({columns}) VALUES ({placeholders})", data)

    def update(self, video_id: str, **fields: Any) -> None:
        assignments = ", ".join(f"{k} = :{k}" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE videos SET {assignments} WHERE id = :id", {**fields, "id": video_id})

    def get(self, video_id: str) -> VideoRecord | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return self._record(row) if row else None

    def list(self, limit: int, after: str | None, order: str) -> tuple[list[VideoRecord], bool]:
        direction = "DESC" if order == "desc" else "ASC"
        comparison = "<" if order == "desc" else ">"
        query = "SELECT * FROM videos"
        args: list[Any] = []
        if after:
            anchor = self.get(after)
            if anchor is None:
                return [], False
            query += f" WHERE (created_at, id) {comparison} (?, ?)"
            args += [anchor.created_at, anchor.id]
        query += f" ORDER BY created_at {direction}, id {direction} LIMIT ?"
        args.append(limit + 1)
        with self._lock:
            rows = self._db.execute(query, args).fetchall()
        records = [self._record(r) for r in rows]
        return records[:limit], len(records) > limit

    def delete(self, video_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM videos WHERE id = ?", (video_id,))

    def by_status(self, *statuses: str) -> list[VideoRecord]:
        marks = ", ".join("?" for _ in statuses)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM videos WHERE status IN ({marks}) ORDER BY created_at, id", statuses
            ).fetchall()
        return [self._record(r) for r in rows]

    def expired(self, now: int | None = None) -> list[VideoRecord]:
        now = int(time.time()) if now is None else now
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM videos WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
            ).fetchall()
        return [self._record(r) for r in rows]
