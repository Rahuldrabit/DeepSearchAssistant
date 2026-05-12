"""SQLite metadata store for indexed files and user feedback.

Schema:
  files  (file_id, path, hash, size, mtime, indexed_at, chunk_count, status)
  feedback (id, query, answer, rating, comment, created_at)
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS files (
    file_id     TEXT PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    hash        TEXT NOT NULL,
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    indexed_at  REAL NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'indexed'
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL,
    answer      TEXT NOT NULL,
    rating      INTEGER NOT NULL,   -- 1 (thumbs up) or -1 (thumbs down)
    comment     TEXT,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
"""


class MetadataDB:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        log.info("Metadata DB opened: %s", self._path)

    # ------------------------------------------------------------------ #
    # Files                                                                #
    # ------------------------------------------------------------------ #

    def file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_indexed(self, path: Path) -> bool:
        """Return True if file is already indexed and hasn't changed."""
        row = self._conn.execute(
            "SELECT hash, mtime FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        if row is None:
            return False
        stat = path.stat()
        if abs(row["mtime"] - stat.st_mtime) < 0.01:
            return True
        # mtime changed — check hash to be sure
        return row["hash"] == self.file_hash(path)

    def upsert_file(
        self,
        file_id: str,
        path: Path,
        chunk_count: int,
        status: str = "indexed",
    ) -> None:
        stat = path.stat()
        h = self.file_hash(path)
        self._conn.execute(
            """
            INSERT INTO files (file_id, path, hash, size, mtime, indexed_at, chunk_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                hash=excluded.hash, size=excluded.size, mtime=excluded.mtime,
                indexed_at=excluded.indexed_at, chunk_count=excluded.chunk_count,
                status=excluded.status
            """,
            (file_id, str(path), h, stat.st_size, stat.st_mtime, time.time(), chunk_count, status),
        )
        self._conn.commit()

    def get_file(self, path: Path) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        return dict(row) if row else None

    def get_file_by_id(self, file_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_file(self, file_id: str) -> None:
        self._conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        self._conn.commit()

    def list_files(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM files ORDER BY indexed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    @property
    def file_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    # ------------------------------------------------------------------ #
    # Feedback                                                             #
    # ------------------------------------------------------------------ #

    def add_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        comment: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT INTO feedback (query, answer, rating, comment, created_at) VALUES (?,?,?,?,?)",
            (query, answer, rating, comment, time.time()),
        )
        self._conn.commit()

    def get_feedback(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
