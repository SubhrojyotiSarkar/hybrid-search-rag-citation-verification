"""SQLite connection handling and schema.

SQLite is the source of truth for documents and chunks. The FAISS index (Step 1c) stores
only vectors keyed by chunks.vector_id, so it can always be rebuilt from this database.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    sha256        TEXT NOT NULL UNIQUE,
    size_bytes    INTEGER NOT NULL,
    status        TEXT NOT NULL,
    page_count    INTEGER,
    empty_pages   TEXT NOT NULL DEFAULT '[]',
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    error_code    TEXT,
    error_message TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- AUTOINCREMENT guarantees a vector_id is never reused after a delete, so a stale
-- FAISS vector can never point at a different chunk.
CREATE TABLE IF NOT EXISTS chunks (
    vector_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    page_label  TEXT,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text        TEXT NOT NULL,
    context     TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""

# Columns added after the first release: (table, column, SQL type). CREATE TABLE IF NOT
# EXISTS does not alter existing tables, so these are added to older databases on startup.
ADDED_COLUMNS = [
    ("chunks", "page_label", "TEXT"),
    ("chunks", "context", "TEXT"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, sql_type in ADDED_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            # WAL lets the frontend's reads proceed while an ingestion is writing.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            _migrate(conn)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection for one unit of work: commit on success, roll back on error."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # needed for ON DELETE CASCADE
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
