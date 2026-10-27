# Demo script (about 10 minutes)

A walkthrough that shows the system working **and** failing honestly. Prepare it once:
upload the demo PDF, wait for `indexed`, and check that the sidebar shows `LLM: gemini`.
On the free tier, leave about a minute between questions (5 requests per minute).

Suggested `.env` for the demo: `RETRIEVAL_MODE=hybrid_rerank`, `MAX_RECOVERY_ATTEMPTS=1`.

## 1. Ingestion and page fidelity (1 min)

- **Documents** tab: show the page count, chunk count and any empty pages.
- *Show chunks*: point out a chunk cited as **p. 194 (PDF page 12)**, the printed page
  label of a journal article, not just the file position, and a table chunk carrying
  **Table context** (caption and column headers) that is used for search but not cited.

## 2. A grounded, verified answer (2 min)

Ask a factual question with a clear answer in the document, e.g.
*"What were the linear kernel RMSE and MAPE for GOOGL on the daily price test data?"*

- Green banner: **verified within the provided evidence**, and explain that it means the
  cited passages support it, not that it is true in general.
- Open *Claims and citations*: each claim, which citation **supports** it, the verifier's
  **evidence quote** (checked by code to occur in the passage), and *open PDF* at the page.
- Caption: LLM calls, time per stage (generation vs verification).

## 3. Abstention (1 min)

Ask something the document does not cover, e.g. *"What is the capital of France?"*

- Blue banner: not enough evidence; no invented answer and no claims.

## 4. Catching an overstatement (2 min)

Ask a leading question, e.g. *"Does SVR with a linear kernel always outperform the
random walk model?"*

- Look at **Removed claims**: statements the verifier rejected, with its reason. If the
  model answered carefully and nothing was removed, show that instead and say so: the
  verifier is a safety net, not a guaranteed catch.
- Point out the verification summary: attempts, removals, quote-check rejections.

## 5. Retrieval, and an honest failure (2 min)

**Retrieval inspector**, *Compare all*, query `GOOGL linear kernel RMSE 0.06248 MAPE 0.03752`:

- BM25 finds the Table 7 row first (exact numbers); dense does not (rank ~25; embeddings
  do not match literal numbers); hybrid RRF ranks it ~10th (outside the 8 passages the
  LLM sees); reranking puts it first. Show each result's component ranks.
- Explain the trade-off: reranking costs ~1 s per question on CPU, which is why it is a
  setting decided by evaluation, not a default.

## 6. Evaluation (2 min)

**Evaluation** tab:

- Dataset health (and that it is still small; the dashboard says so).
- A retrieval run: hit@k per configuration, MRR, latency; table views.
- An answer run: verifier-judged numbers labelled *not ground truth*, then **Review
  claims**: judge one claim live and watch the human-judged citation precision update.

## Failure cases worth showing (if time allows)

| Case | What to show |
|---|---|
| Provider quota / overload | An answer run with a rate-limited question: recorded as an error with Gemini's message, never as a wrong answer |
| Verifier outage | `verification_failed` (red banner): claims shown but explicitly **not** checked |
| Scanned PDF | Upload an image-only PDF: fails with `no_extractable_text` and an explanation |
| Prompt injection | A PDF containing "ignore previous instructions": sources are delimited and marked untrusted; the verifier is a second line of defence (not a guarantee) |

## Talking points

- Every claim traces to a document, a printed page and an exact passage.
- The verifier is independent (it sees only the claim and its cited passages) and is
  itself checked by code (its quote must exist in the passage).
- Technical failure, missing evidence and unsupported claims are distinct states.
- Each fix (page labels, table context, Lucene IDF, whole decimals, reranking) came from a
  measured failure, with a regression test.
