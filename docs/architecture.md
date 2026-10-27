# Architecture

This document explains how the system is put together and why. For setup, see the
[README](../README.md); for how it is measured, see [evaluation.md](evaluation.md).

## Components

| Layer | Module | Responsibility |
|---|---|---|
| API | `backend/api/` | FastAPI routers; request validation; errors as `{"detail": {"code", "message"}}` |
| Ingestion | `backend/ingestion/` | PDF extraction, chunking, ingestion and re-processing |
| Storage | `backend/storage/` | SQLite schema with additive migrations; all SQL in `repositories.py` |
| Retrieval | `backend/retrieval/` | embeddings, FAISS store, BM25, RRF fusion, cross-encoder reranking |
| Generation | `backend/generation/` | provider-agnostic LLM interface, Gemini client, answer generation |
| Verification | `backend/verification/` | claim-level citation verifier, bounded recovery and final status |
| Evaluation | `backend/evaluation/` | datasets, metrics, runners, background jobs |
| Frontend | `frontend/` | Streamlit UI; talks to the backend only over HTTP |

Long-lived objects (database, tokenizer, embedder, FAISS store, BM25 index, reranker, LLM
client, job manager) are created once at startup in `backend/main.py` and handed to
request handlers by `backend/api/dependencies.py`. `backend/generation/factory.py` builds
the answer pipeline from settings, so the API and the evaluation runners use exactly the
same configuration.

## Ingestion

1. **Validate**: extension, `%PDF-` signature, size limit, SHA-256 duplicate check. The
   stored file is named after the generated document ID, never the user's filename.
2. **Extract** each page with PyMuPDF. Text is whitespace-normalized only, so citation
   snippets match the PDF. Each page records its physical number (for `#page=N` links)
   and its **printed page label**: a journal article whose first page is printed "183"
   cites "p. 186", not "page 4". Pages with fewer than 20 visible characters are flagged
   empty (typically scanned images); a document with no text fails with a clear reason.
3. **Chunk** with the embedding model's own tokenizer: 220-token windows, 40-token
   overlap, snapped to word starts, never crossing a page. MiniLM reads at most 256
   tokens, so larger chunks would be silently truncated for dense retrieval. Chunk text
   is sliced from the page string by character offsets, so it is an exact substring.
4. **Table context.** When a table is split across chunks, only the first chunk contains
   its caption and column headers; the rest would be bare numbers. Chunks that continue
   a table store the caption and header row as `context`. It is prefixed to the text that
   is embedded, BM25-indexed, reranked and shown to the LLM, but it is never part of the
   cited text. Chunks on table pages are shortened by the context length so the prefix
   still fits the 256-token embedding window.
5. **Persist**: chunks are committed to SQLite (`chunked`), vectors added to FAISS and
   saved atomically, then the document becomes `indexed`.

### Consistency between SQLite and FAISS

SQLite is the source of truth; FAISS stores only vectors keyed by `chunks.vector_id`
(`AUTOINCREMENT`, so an ID is never reused after a delete). At every startup,
`reconcile_index` compares the two: chunks missing from FAISS (index lost or corrupt, a
crash mid-ingestion, a changed embedding model) are re-embedded; vectors without a chunk
(a crash mid-delete) are removed. BM25 is rebuilt from SQLite at startup, so there is no
second persisted index to drift. Deletion removes SQLite rows first, so a partial failure
can only leave unreachable vectors, which reconciliation cleans up.

Re-processing (`POST /api/documents/{id}/reprocess`, or `python -m backend.maintenance
reprocess`) re-chunks a stored PDF in place, keeping the document ID. New chunks are built
before anything is replaced and swapped in one transaction, so a failure leaves the old
chunks searchable.

## Retrieval

- **Dense**: MiniLM embeddings, L2-normalized, exact inner-product search in FAISS
  (`IndexIDMap2(IndexFlatIP)`), with document filters applied inside FAISS.
- **BM25**: lowercased tokens without stopwords; technical compounds are kept and split
  ("f1-score" matches "F1 score"), plain decimals stay whole ("0.06248" does not emit a
  "0" token that would match every number in every table). The IDF is Lucene's
  `log(1 + (N - n + 0.5) / (n + 0.5))`: rank-bm25's default gives a term in exactly half
  the chunks an IDF of 0, silently ignoring real query terms in small collections.
- **Hybrid**: each retriever returns 20 candidates; Reciprocal Rank Fusion
  (`score = Σ 1 / (60 + rank)`) merges them by rank, because BM25 scores and cosine
  similarities are not comparable. Results keep each component's rank and score.
- **Reranking** (`RETRIEVAL_MODE=hybrid_rerank`): a cross-encoder rescores the top 30
  fused candidates with the query and passage read together. It loads lazily; if it
  fails, the hybrid order is returned labelled `hybrid`, so a fallback is visible.

## Answer generation and verification

1. Retrieve the top 8 passages; if none, abstain without an LLM call.
2. **Generate.** Passages are labelled `S1…Sn` inside `<source>` tags marked as untrusted
   (tag-like text inside documents is neutralized). Gemini returns, under a JSON schema,
   atomic claims with the labels that support each one, plus an answer status. Labels the
   model invents are recorded as invalid citations, never shown as sources. The answer
   text is assembled from the claims, so every sentence carries its citations.
3. **Verify.** A separate call sees each claim with only its own cited passages, not the
   question, other passages or the generator's status. It returns supported /
   contradicted / insufficient_evidence with an explanation. The output is validated:
   every claim needs exactly one verdict (otherwise `verification_failed`), supporting
   labels must be the claim's own, and a supported or contradicted verdict must quote
   text that code finds in a cited passage, otherwise it is downgraded.
4. **Recover** (at most `MAX_RECOVERY_ATTEMPTS`, default 1): retrieve 12 passages and
   regenerate with the rejected claims and reasons as feedback; verify again; keep the
   best fully verified attempt.
5. **Finalize.** Unsupported claims are removed from the answer and returned in
   `removed_claims` with reasons (policy `remove`) or kept and marked (policy `flag`).
   No supported claim left means abstention. A technical failure produces
   `verification_failed`, never a verified status.

The LLM is behind a small interface (`generate_structured(system, user, schema)`); the
Gemini client maps every failure to a stable code (rate limit, outage, refusal,
truncation, invalid output) and checks the finish reason before parsing.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/documents/upload`, `GET /api/documents`, `GET/DELETE /api/documents/{id}` | document management |
| `GET /api/documents/{id}/chunks`, `GET /api/documents/{id}/file`, `POST /api/documents/{id}/reprocess` | inspection, original PDF, re-chunking |
| `POST /api/retrieval/search` | retrieval only, any mode |
| `POST /api/chat/ask` | full answer with claims, citations, verification |
| `GET /api/evaluation/runs`, `GET /api/evaluation/runs/{id}`, `PUT /api/evaluation/runs/{id}/reviews` | runs, metrics, human reviews |
| `POST /api/evaluation/run`, `GET /api/evaluation/jobs/{id}` | background evaluation jobs |
| `GET /health` | configuration and readiness (never secrets) |

## Operations

- Every request gets an `X-Request-ID` (accepted from the caller if safe), included on
  every log line and in server-error bodies; one access log line per request with latency.
- Unhandled exceptions become a JSON 500 with the request ID; stack traces stay in logs.
- If the embedding model or index cannot load, the backend still starts (`degraded`) and
  retrieval endpoints return 503; a missing or invalid LLM configuration disables only
  question answering.
- Secrets come from `.env` (`SecretStr`, never logged or returned); `.env` is gitignored
  and excluded from the Docker image.

## Trade-offs and known limitations

- **Gemini free tier.** No cost, but 5 requests per minute (when tested), occasional
  overload errors (503/504), and free-tier data may be used by Google. Verification
  doubles the calls per answer. The LLM interface makes a provider switch a single class.
- **RRF discards score magnitude.** A decisive BM25 #1 that dense retrieval cannot see
  (numeric table rows) can fall outside the top 8. Reranking fixes the observed case;
  score-aware fusion is an alternative to evaluate. A strict expected-failure test
  documents this.
- **Reranking costs ~0.6-1 s per question on CPU**, so it is opt-in until the evaluation
  shows it pays off across questions.
- **The quote check is strict**: it rejects correct verdicts whose quote is slightly off,
  which costs a recovery attempt (observed once in a real run). The rejected quote is
  recorded so this can be measured.
- **Single process.** Jobs and BM25 live in memory; FAISS is a flat index (exact search,
  fine for tens of thousands of chunks). Multi-user use would need authentication,
  per-user collections and a shared job queue.
- **Text PDFs only**; table structure is recovered heuristically from extracted text.
