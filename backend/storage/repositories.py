"""Read/write access to documents and chunks. All SQL lives here."""

import json
import sqlite3
from datetime import UTC, datetime

from backend.schemas import Chunk, DocumentRecord, DocumentStatus
from backend.storage.database import Database


class DuplicateHashError(Exception):
    """A document with the same file hash already exists."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    data = dict(row)
    data["empty_pages"] = json.loads(data["empty_pages"])
    data["metadata"] = json.loads(data["metadata"])
    return DocumentRecord(**data)


class DocumentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, document_id: str, filename: str, sha256: str, size_bytes: int) -> None:
        now = _now()
        try:
            with self.db.connect() as conn:
                conn.execute(
                    """INSERT INTO documents
                       (document_id, filename, sha256, size_bytes, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (document_id, filename, sha256, size_bytes, DocumentStatus.PROCESSING, now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateHashError(sha256) from exc

    def get(self, document_id: str) -> DocumentRecord | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def get_by_sha256(self, sha256: str) -> DocumentRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return _row_to_document(row) if row else None

    def list_all(self) -> list[DocumentRecord]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [_row_to_document(row) for row in rows]

    def save_chunks(
        self,
        document_id: str,
        chunks: list[Chunk],
        page_count: int,
        empty_pages: list[int],
        metadata: dict[str, str],
    ) -> None:
        """Store all chunks and mark the document CHUNKED in a single transaction."""
        with self.db.connect() as conn:
            conn.executemany(
                """INSERT INTO chunks
                   (chunk_id, document_id, chunk_index, page_number, start_char, end_char,
                    token_count, text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (c.chunk_id, c.document_id, c.chunk_index, c.page_number, c.start_char,
                     c.end_char, c.token_count, c.text)
                    for c in chunks
                ],
            )
            conn.execute(
                """UPDATE documents
                   SET status = ?, page_count = ?, empty_pages = ?, chunk_count = ?,
                       metadata = ?, error_code = NULL, error_message = NULL, updated_at = ?
                   WHERE document_id = ?""",
                (DocumentStatus.CHUNKED, page_count, json.dumps(empty_pages), len(chunks),
                 json.dumps(metadata), _now(), document_id),
            )

    def mark_failed(
        self,
        document_id: str,
        error_code: str,
        error_message: str,
        page_count: int | None = None,
        empty_pages: list[int] | None = None,
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE documents
                   SET status = ?, error_code = ?, error_message = ?,
                       page_count = COALESCE(?, page_count), empty_pages = ?, updated_at = ?
                   WHERE document_id = ?""",
                (DocumentStatus.FAILED, error_code, error_message, page_count,
                 json.dumps(empty_pages or []), _now(), document_id),
            )

    def list_chunks(self, document_id: str) -> list[Chunk]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT c.chunk_id, c.document_id, d.filename, c.page_number, c.chunk_index,
                          c.start_char, c.end_char, c.token_count, c.text
                   FROM chunks c JOIN documents d ON d.document_id = c.document_id
                   WHERE c.document_id = ?
                   ORDER BY c.chunk_index""",
                (document_id,),
            ).fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def delete(self, document_id: str) -> bool:
        """Delete a document and (via ON DELETE CASCADE) its chunks. Returns False if absent."""
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cursor.rowcount > 0
