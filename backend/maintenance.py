"""Maintenance commands for the document collection and its indexes.

Run from the project root, with the backend STOPPED (two processes writing the same FAISS
file would overwrite each other's work):

    .\\.venv\\Scripts\\python.exe -m backend.maintenance status
    .\\.venv\\Scripts\\python.exe -m backend.maintenance rebuild-index
    .\\.venv\\Scripts\\python.exe -m backend.maintenance reprocess --all
    .\\.venv\\Scripts\\python.exe -m backend.maintenance reprocess doc_3fa2f0eb7ba0

status         Documents, chunk and vector counts, and whether SQLite and FAISS agree.
rebuild-index  Re-embed every indexed chunk from SQLite into a fresh FAISS index (use after
               changing EMBEDDING_MODEL_NAME, or if search results look stale).
reprocess      Re-extract and re-chunk stored PDFs in place, keeping document IDs (use after
               the chunking logic or CHUNK_SIZE_TOKENS changed). A single document can also
               be re-processed from the Documents tab while the backend is running.
"""

import argparse
import sys

from backend.config import get_settings


def index_status(app) -> dict:
    from backend.storage.repositories import DocumentRepository

    repository = DocumentRepository(app.state.db)
    services = app.state.retrieval
    documents = repository.list_all()
    expected = repository.indexable_vector_ids()
    present = services.vector_store.ids() if services else set()
    by_status: dict[str, int] = {}
    for document in documents:
        status = str(document.status.value)
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "documents": len(documents),
        "by_status": by_status,
        "chunks_expected_in_index": len(expected),
        "faiss_vectors": len(present),
        "bm25_chunks": services.bm25_index.count if services else 0,
        "missing_from_faiss": len(expected - present),
        "orphan_vectors": len(present - expected),
        "consistent": services is not None and expected == present,
        "embedding_model": get_settings().embedding_model_name,
    }


def rebuild_index(app) -> dict:
    """Drop every vector and re-embed all indexable chunks from SQLite."""
    from backend.ingestion.pipeline import reconcile_index
    from backend.storage.repositories import DocumentRepository

    services = app.state.retrieval
    store = services.vector_store
    store.remove(store.ids())
    report = reconcile_index(DocumentRepository(app.state.db), store, services.embedder)
    store.save()
    return {"vectors": store.count, "re_embedded": report.missing_vectors_added}


def reprocess_documents(app, document_ids: list[str] | None) -> list[dict]:
    """Re-chunk the given documents (all, if None). One failure does not stop the rest."""
    from backend.ingestion.pipeline import IngestionError, IngestionPipeline
    from backend.storage.repositories import DocumentRepository

    settings = get_settings()
    services = app.state.retrieval
    repository = DocumentRepository(app.state.db)
    pipeline = IngestionPipeline(settings, repository, app.state.tokenizer,
                                 services.embedder, services.vector_store, services.bm25_index)
    targets = document_ids or [d.document_id for d in repository.list_all()]
    results = []
    for document_id in targets:
        before = repository.get(document_id)
        try:
            after = pipeline.reprocess(document_id)
            results.append({"document_id": document_id, "filename": after.filename, "ok": True,
                            "chunks_before": before.chunk_count if before else None, "chunks_after": after.chunk_count})
        except IngestionError as exc:
            results.append({"document_id": document_id, "ok": False, "error": f"{exc.code}: {exc.message}"})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("rebuild-index")
    reprocess = commands.add_parser("reprocess")
    reprocess.add_argument("document_ids", nargs="*")
    reprocess.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "reprocess" and not (args.all or args.document_ids):
        parser.error("reprocess needs document IDs or --all")

    from fastapi.testclient import TestClient

    from backend.main import create_app

    if args.command != "status":
        print("Make sure the backend is not running: two processes writing the same index lose updates.")
    app = create_app()
    with TestClient(app):  # startup: loads models, reconciles and builds the indexes
        if app.state.retrieval is None:
            print("Retrieval is unavailable (embedding model or index failed to load); see the log above.", file=sys.stderr)
            return 1
        if args.command == "status":
            for key, value in index_status(app).items():
                print(f"{key:26} {value}")
        elif args.command == "rebuild-index":
            result = rebuild_index(app)
            print(f"Rebuilt the FAISS index: {result['re_embedded']} chunk(s) re-embedded, {result['vectors']} vector(s).")
        else:
            results = reprocess_documents(app, None if args.all else args.document_ids)
            for result in results:
                if result["ok"]:
                    print(f"OK   {result['document_id']} {result['filename']}: "
                          f"{result['chunks_before']} -> {result['chunks_after']} chunks")
                else:
                    print(f"FAIL {result['document_id']}: {result['error']}")
            if not all(r["ok"] for r in results):
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
