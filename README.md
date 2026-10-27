# Hybrid-Search RAG with Citation Verification

A document question-answering system that answers only from uploaded PDFs, cites the exact
page and passage behind every claim, and **independently checks each citation** before
showing it. Claims whose cited passage does not support them are removed or flagged, and
the system abstains when the documents do not contain the answer.

The goal is not "a chatbot over PDFs" but a measurable answer to: *does each claim have
identifiable supporting evidence?* Retrieval quality and citation reliability are both
evaluated against human-reviewed data.

## What it does

| Stage | Implementation |
|---|---|
| Ingestion | PyMuPDF page-by-page extraction, printed page labels (e.g. "p. 186" in a journal article), token-bounded overlapping chunks that never cross pages, table caption/header context carried into continuation chunks |
| Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2`, persistent FAISS (`IndexIDMap2`), reconciled against SQLite at startup |
| Keyword retrieval | BM25 (rank-bm25 with Lucene IDF), technical-term-aware tokenizer |
| Fusion | Reciprocal Rank Fusion of dense + BM25 |
| Reranking | Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` over the fused candidates (optional mode) |
| Generation | Google Gemini (`gemini-3.8-flash`, free tier) with schema-constrained output: atomic claims, each with source labels |
| Verification | A separate LLM call per answer judges every claim against only its own cited passages (supported / contradicted / insufficient evidence); its evidence quote must be found in the passage by code |
| Recovery | One bounded retry with more evidence and the rejected claims as feedback; unsupported claims removed (listed separately) or the system abstains |
| Evaluation | Retrieval metrics (hit@k, recall@k, MRR), answer/citation metrics, human review UI, background evaluation jobs, dashboard |

Distinct answer states are never collapsed into one green check:
`verified_within_provided_evidence`, `contains_unresolved_claims`, `verification_failed`
(technical failure, never shown as verified), `insufficient_evidence`, `unverified`.

## Architecture

```
PDF ──► extract (PyMuPDF) ──► chunk (+ table context) ──► SQLite (source of truth)
                                                     ├──► MiniLM ──► FAISS
                                                     └──► BM25 (rebuilt from SQLite)

question ──► dense ─┐
             BM25 ──┴► RRF ──► [cross-encoder] ──► top-k passages
                                                     │
              Gemini: claims + citations ◄───────────┘
                         │
              Gemini verifier (claim + its cited passages only) ──► code quote check
                         │
              retry once with feedback, or remove / abstain ──► answer + citations + statuses
```

FastAPI backend (all logic and state) and a Streamlit frontend that talks to it only over
HTTP. Details and design decisions: [docs/architecture.md](docs/architecture.md).

## Results so far (preliminary)

The evaluation dataset currently has **2 questions** (1 human-reviewed), so these numbers
illustrate the pipeline; they are **not** evidence that one configuration is better in
general. Measured on one 19-page paper (136 chunks), CPU, from
`evaluation/results/retrieval_20261024T052226Z.json`:

| Retrieval | MRR@20 | hit@8 | Mean latency |
|---|---|---|---|
| dense | 0.10 | 0.5 | 20 ms |
| BM25 | 1.00 | 1.0 | 3 ms |
| hybrid (RRF) | 0.55 | 0.5 | 23 ms |
| hybrid + cross-encoder rerank (30 candidates) | 1.00 | 1.0 | 966 ms |

The failure behind these numbers is documented as a case study in
[docs/evaluation.md](docs/evaluation.md): a numeric table row that dense retrieval cannot
match and RRF ranks 10th, which reranking moves to 1st. Answer-level and citation metrics
need the full dataset and human reviews; see the methodology there.

## Setup (Windows, PowerShell)

Requirements: Python 3.11, about 2 GB disk (PyTorch CPU + models), a free
[Google AI Studio](https://aistudio.google.com/apikey) API key. No billing account is needed.

Run every command from the project root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env   # set GEMINI_API_KEY (never put it in .env.example)
```

Start the backend and the frontend in two terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open http://localhost:8501. The API documentation is at http://localhost:8000/docs.
The first start downloads the embedding model (~90 MB); the cross-encoder (~90 MB) is
downloaded on the first reranked query.

On macOS/Linux use `python3.11 -m venv .venv` and `.venv/bin/python` instead of
`.\.venv\Scripts\python.exe`.

### Docker

```powershell
Copy-Item .env.example .env   # and set GEMINI_API_KEY
docker compose up --build
```

`data/` and `evaluation/` are mounted from the host; models are cached in a volume.
**The Docker setup has not been built or run yet** (Docker was not available on the
development machine); treat it as a starting point and report issues.

## Using it

1. **Documents**: upload PDFs (text-based; scanned PDFs are reported as having no
   extractable text). *Re-process* re-chunks a stored PDF after settings change.
2. **Ask**: answers show each claim with its citations (document, printed page, snippet,
   *open PDF* at the page), which citation supports it, the verifier's evidence quote,
   and any removed claims with reasons.
3. **Retrieval inspector**: compare dense, BM25, hybrid and reranked results side by side.
4. **Evaluation**: dataset health, start runs, metrics, and human review of claims.

Important settings (all in `.env`, documented in `.env.example`):

| Setting | Default | Notes |
|---|---|---|
| `RETRIEVAL_MODE` | `hybrid` | `hybrid_rerank` is more accurate in our tests but adds ~0.6-1 s per question on CPU |
| `VERIFICATION_ENABLED` | `true` | 1 extra LLM call per answer |
| `MAX_RECOVERY_ATTEMPTS` | `1` | each attempt adds 2 calls; use `0` to stay within the free quota |
| `UNSUPPORTED_CLAIM_POLICY` | `remove` | or `flag` (keep and mark) |
| `GEMINI_MODEL` | `gemini-3.8-flash` | has a free tier; check the pricing page before changing |

**Free-tier limits.** `gemini-3.8-flash` allowed 5 requests per minute when tested, and a
verified answer uses 2-4 requests, so expect 1-2 questions per minute. Google states that
free-tier usage may be used to improve its products: do not upload confidential documents.

## Evaluation

```powershell
# Retrieval comparison (no LLM calls, seconds)
.\.venv\Scripts\python.exe -m backend.evaluation.run_retrieval_eval
# Full answer pipeline, paced to the free-tier limit (minutes per question)
.\.venv\Scripts\python.exe -m backend.evaluation.run_answer_eval --limit 3
```

Both can also be started from the Evaluation tab. Dataset format, metric definitions and
the human review protocol: [docs/evaluation.md](docs/evaluation.md).

## Maintenance

With the backend stopped:

```powershell
.\.venv\Scripts\python.exe -m backend.maintenance status          # SQLite / FAISS consistency
.\.venv\Scripts\python.exe -m backend.maintenance rebuild-index   # after changing the embedding model
.\.venv\Scripts\python.exe -m backend.maintenance reprocess --all # after changing chunking settings
```

Every API response carries an `X-Request-ID` header; server errors show it in the UI and
every backend log line includes it.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

242 passing tests, plus 1 opt-in real-model test (skipped unless `RUN_MODEL_TESTS=1`) and
1 expected failure that documents the known RRF limitation. The suite uses fake embedders, rerankers
and LLMs, so it needs no network access, no API key and no model downloads.

## Project structure

```
backend/
  api/            FastAPI routers: documents, retrieval, chat, evaluation
  ingestion/      PDF extraction, chunking (incl. table context), ingestion pipeline
  retrieval/      embeddings, FAISS store, BM25, RRF hybrid, cross-encoder reranker
  generation/     LLM client (Gemini), answer generation, pipeline factory
  verification/   citation verifier, bounded recovery
  evaluation/     dataset, metrics, runners, background jobs
  storage/        SQLite schema/migrations, repositories
  maintenance.py  status / rebuild-index / reprocess
frontend/         Streamlit app and evaluation dashboard
evaluation/       datasets/ (questions), results/ (runs), reviews/ (human labels)
docs/             architecture, evaluation methodology, demo script
tests/            pytest suite
```

## Limitations

- Text-based PDFs only (no OCR); table structure is recovered heuristically from text.
- Verification is an LLM judgment: probabilistic, checked against human reviews but not
  guaranteed. "Supported" means *supported by the cited passage*, not true in the world.
- English-oriented models (MiniLM, ms-marco cross-encoder); quality on other languages is untested.
- Single user, no authentication; evaluation jobs live in memory.
- The evaluation dataset is still small; no general claims about improvements are made yet.

More in [docs/evaluation.md](docs/evaluation.md#limitations) and
[docs/architecture.md](docs/architecture.md#trade-offs-and-known-limitations).
