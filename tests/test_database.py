import sqlite3

from backend.storage.database import Database

# The chunks table as created before page labels were added.
OLD_CHUNKS_TABLE = """
CREATE TABLE documents (document_id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL, status TEXT NOT NULL, page_count INTEGER,
    empty_pages TEXT NOT NULL DEFAULT '[]', chunk_count INTEGER NOT NULL DEFAULT 0, error_code TEXT,
    error_message TEXT, metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE chunks (vector_id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, page_number INTEGER NOT NULL,
    start_char INTEGER NOT NULL, end_char INTEGER NOT NULL, token_count INTEGER NOT NULL, text TEXT NOT NULL);
INSERT INTO chunks (chunk_id, document_id, chunk_index, page_number, start_char, end_char, token_count, text)
    VALUES ('c1', 'd1', 0, 1, 0, 4, 1, 'text');
"""


def test_initialize_adds_missing_columns_and_keeps_rows(tmp_path):
    path = tmp_path / "app.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(OLD_CHUNKS_TABLE)

    Database(path).initialize()

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        rows = conn.execute("SELECT chunk_id, page_label FROM chunks").fetchall()
    assert {"page_label", "context"} <= columns
    assert rows == [("c1", None)]


def test_initialize_is_idempotent(tmp_path):
    db = Database(tmp_path / "app.db")
    db.initialize()
    db.initialize()  # must not fail by re-adding existing columns
