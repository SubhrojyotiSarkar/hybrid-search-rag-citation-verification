"""Read/write access to documents and chunks. All SQL lives here."""

import json
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime

from backend.schemas import Chunk, DocumentRecord, DocumentStatus, index_text
from backend.storage.database import Database

# Stay well below SQLite's limit on the number of "?" parameters in one statement.
SQL_BATCH_SIZE = 500

CHUNK_COLUMNS = """c.vector_id, c.chunk_id, c.document_id, d.filename, c.page_number,
                   c.page_label, c.chunk_index, c.start_char, c.end_char, c.token_count, c.text,
                   c.context"""


class DuplicateHashError(Exception):
    """A document with the same file hash already exists."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _batched(values: Iterable, size: int = SQL_BATCH_SIZE) -> Iterator[list]:
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    data = dict(row)
    data.pop("vector_id", None)
    return Chunk(**data)


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
                   (chunk_id, document_id, chunk_index, page_number, page_label, start_char,
                    end_char, token_count, text, context)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (c.chunk_id, c.document_id, c.chunk_index, c.page_number, c.page_label,
                     c.start_char, c.end_char, c.token_count, c.text, c.context)
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

    def mark_indexed(self, document_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, updated_at = ? WHERE document_id = ?",
                (DocumentStatus.INDEXED, _now(), document_id),
            )

    def mark_chunked_as_indexed(self) -> int:
        """Promote CHUNKED documents (ingestion interrupted after chunking) once their
        vectors have been restored. Returns the number of documents promoted."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = ?, updated_at = ? WHERE status = ?",
                (DocumentStatus.INDEXED, _now(), DocumentStatus.CHUNKED),
            )
        return cursor.rowcount

    def fail_interrupted(self) -> int:
        """Mark documents left in PROCESSING by a crash or restart as FAILED."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """UPDATE documents SET status = ?, error_code = 'interrupted',
                       error_message = 'Processing was interrupted. Delete and re-upload the file.',
                       updated_at = ?
                   WHERE status = ?""",
                (DocumentStatus.FAILED, _now(), DocumentStatus.PROCESSING),
            )
        return cursor.rowcount

    def list_chunks(self, document_id: str) -> list[Chunk]:
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""SELECT {CHUNK_COLUMNS}
                    FROM chunks c JOIN documents d ON d.document_id = c.document_id
                    WHERE c.document_id = ?
                    ORDER BY c.chunk_index""",
                (document_id,),
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def vector_ids_for_document(self, document_id: str) -> list[int]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT vector_id FROM chunks WHERE document_id = ? ORDER BY vector_id", (document_id,)
            ).fetchall()
        return [row["vector_id"] for row in rows]

    # The three methods below return (vector_id, index text) pairs: the text that is
    # embedded and BM25-indexed, i.e. table context (if any) followed by the chunk text.

    def chunk_texts_for_document(self, document_id: str) -> list[tuple[int, str]]:
        """Index texts for embedding one document's chunks."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT vector_id, context, text FROM chunks WHERE document_id = ? ORDER BY vector_id",
                (document_id,),
            ).fetchall()
        return [(row["vector_id"], index_text(row["context"], row["text"])) for row in rows]

    def chunk_texts(self, vector_ids: Iterable[int]) -> list[tuple[int, str]]:
        """Index texts for re-embedding specific chunks during reconciliation."""
        pairs: list[tuple[int, str]] = []
        with self.db.connect() as conn:
            for batch in _batched(sorted(vector_ids)):
                rows = conn.execute(
                    f"SELECT vector_id, context, text FROM chunks WHERE vector_id IN ({_placeholders(len(batch))})",
                    batch,
                ).fetchall()
                pairs.extend((row["vector_id"], index_text(row["context"], row["text"])) for row in rows)
        return pairs

    def indexed_chunk_texts(self) -> list[tuple[int, str]]:
        """Index texts of every chunk of an INDEXED document; used to build BM25."""
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT c.vector_id, c.context, c.text FROM chunks c JOIN documents d ON d.document_id = c.document_id
                   WHERE d.status = ? ORDER BY c.vector_id""",
                (DocumentStatus.INDEXED,),
            ).fetchall()
        return [(row["vector_id"], index_text(row["context"], row["text"])) for row in rows]

    def indexable_vector_ids(self) -> set[int]:
        """vector_ids that should be in FAISS: chunks of CHUNKED or INDEXED documents."""
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT c.vector_id FROM chunks c JOIN documents d ON d.document_id = c.document_id
                   WHERE d.status IN (?, ?)""",
                (DocumentStatus.CHUNKED, DocumentStatus.INDEXED),
            ).fetchall()
        return {row["vector_id"] for row in rows}

    def indexed_vector_ids(self, document_ids: list[str]) -> set[int]:
        """vector_ids of the given documents, restricted to INDEXED ones."""
        ids: set[int] = set()
        with self.db.connect() as conn:
            for batch in _batched(document_ids):
                rows = conn.execute(
                    f"""SELECT c.vector_id FROM chunks c JOIN documents d ON d.document_id = c.document_id
                        WHERE d.status = ? AND c.document_id IN ({_placeholders(len(batch))})""",
                    [DocumentStatus.INDEXED, *batch],
                ).fetchall()
                ids.update(row["vector_id"] for row in rows)
        return ids

    def get_indexed_chunks(self, vector_ids: list[int]) -> dict[int, Chunk]:
        """Map vector_id -> Chunk, only for chunks belonging to INDEXED documents."""
        chunks: dict[int, Chunk] = {}
        with self.db.connect() as conn:
            for batch in _batched(vector_ids):
                rows = conn.execute(
                    f"""SELECT {CHUNK_COLUMNS}
                        FROM chunks c JOIN documents d ON d.document_id = c.document_id
                        WHERE d.status = ? AND c.vector_id IN ({_placeholders(len(batch))})""",
                    [DocumentStatus.INDEXED, *batch],
                ).fetchall()
                chunks.update((row["vector_id"], _row_to_chunk(row)) for row in rows)
        return chunks

    def delete(self, document_id: str) -> bool:
        """Delete a document and (via ON DELETE CASCADE) its chunks. Returns False if absent."""
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cursor.rowcount > 0
