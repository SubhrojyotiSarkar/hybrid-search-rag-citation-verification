# Evaluation methodology

Evaluation is part of the system, not an afterthought: every improvement in this project
was motivated by a measured failure, and no improvement is claimed without data. This
document defines the dataset, the metrics, the human review protocol, the current
results and the limits of all of them.

## Dataset

`evaluation/datasets/questions.jsonl`, one JSON object per line:

```json
{"id": "q001", "type": "keyword", "question": "...", "document": "paper.pdf",
 "evidence": [{"page": "194", "quote": "GOOGL 0.06248 0.03752"}],
 "key_facts": ["..."], "should_not_claim": ["..."], "reviewed": true, "notes": "..."}
```

- `type` is `keyword` (exact terms, numbers), `paraphrase` (different wording from the
  source), `factual`, or `unanswerable` (the documents do not answer it; no evidence).
- **Relevance is defined by evidence quotes, not chunk IDs.** Chunk IDs change whenever a
  document is re-chunked; a short exact quote does not. A retrieved chunk is relevant if
  its cited text contains a quote (case-insensitive, whitespace collapsed). Quotes found
  in no chunk are reported and the question is excluded, rather than silently counted as
  a retrieval miss.
- `should_not_claim` lists tempting overstatements, for checking answers by hand.
- Questions must be written from the documents, **not** from what the retriever returns,
  or the test is biased towards the current system. `reviewed: true` marks questions a
  person has checked.

## Retrieval metrics

Per answerable question, the rank of the first retrieved chunk that contains each quote.

- **hit@k**: share of questions with a relevant chunk in the top k.
- **recall@k**: share of a question's evidence quotes found in the top k (averaged).
- **MRR**: mean of 1 / rank of the first relevant chunk (0 if not in the retrieved depth).
- **Latency**: mean and max per configuration, after a warm-up query (model loading is
  excluded).

Configurations: dense, BM25, hybrid (RRF; pool size and `k` can be swept), hybrid +
reranking (candidate pool can be swept). Each run records the configuration, dataset
fingerprint and git commit.

## Answer and citation metrics

Two kinds of numbers are kept strictly separate.

**Verifier-judged** (automatic, *not* ground truth): the share of first-attempt claims the
verifier rejected, quote-check rejections and verification failures. Useful for
monitoring; the verifier is grading its own pipeline, so it cannot validate itself.

**Human-judged** (ground truth, only from reviewed claims). A reviewer sees the claim, its
cited passages and the verifier's verdict, and judges *whether the cited passages
support the claim*, not whether it is true in the world.

- **Citation precision** = claims whose citations support them / reviewed claims;
  **unsupported-claim rate** = 1 − citation precision. Measured at claim level.
- **Before vs after verification**: *before* = the first attempt's claims as generated;
  *after* = the claims shown in the final answer. The difference is what verification
  and recovery contribute.
- **Verifier quality**: agreement with the human, precision and recall of its
  "supported" verdict, false positives (verifier supported, human did not) and false
  negatives, and the full confusion matrix. Claims whose verification failed technically
  are excluded.
- **Answer level**: correct abstentions on unanswerable questions, false abstentions on
  answerable ones, the share of answers that needed correction, the status distribution,
  and latency, LLM calls and tokens per question by stage.

Reviews are stored in `evaluation/reviews/<run_id>.json`, separately from the run, so
re-running never overwrites them.

## Running

```powershell
.\.venv\Scripts\python.exe -m backend.evaluation.run_retrieval_eval --rerank-candidates 20,30
.\.venv\Scripts\python.exe -m backend.evaluation.run_answer_eval --limit 3
```

Or start either from the Evaluation tab and follow its progress there. Answer runs are
paced to `EVAL_REQUESTS_PER_MINUTE` (5) and retry rate limits and provider overloads
twice; remaining failures are recorded with the provider's message. With verification on,
a question uses 2-4 LLM calls, so 25 questions take roughly 15-20 minutes. Answer-run
files contain document text and are gitignored; retrieval runs and reviews are tracked.

## Current results

**Dataset: 2 questions, 1 human-reviewed, 1 document (a 19-page journal article). These
results describe this dataset only and support no general conclusion.**

Retrieval (`evaluation/results/retrieval_20261024T052226Z.json`; CPU; rank of the first
relevant chunk, blank = not in the top 20):

| Configuration | q001 (exact numbers) | q002 (natural question) | MRR@20 | Mean latency |
|---|---|---|---|---|
| dense | – | 5 | 0.10 | 20 ms |
| BM25 | 1 | 1 | 1.00 | 3 ms |
| hybrid (pool 20, k 60) | 10 | 1 | 0.55 | 23 ms |
| hybrid + rerank (30) | 1 | 1 | 1.00 | 966 ms |

Answer runs so far were dominated by provider limits: one run hit the free-tier quota on
both questions; in another, q001 abstained (its evidence was ranked 10th by the default
hybrid retrieval, outside the 8 passages sent to the LLM) and q002's verification call
failed and was correctly reported as `verification_failed`. A manual end-to-end check
verified both claims of q002 against printed page 194 after one recovery attempt, in
4 LLM calls and about 58 s.

### Case study: a numeric table row

Query: `GOOGL linear kernel RMSE 0.06248 MAPE 0.03752` (Table 7, printed page 194).

1. **Extraction was correct, but the table was split**: the row was in a chunk without
   the table's caption and headers. A natural-language version of the question could not
   retrieve it at all (not in the top 50 for any retriever). *Fix*: carry table context
   into continuation chunks. The natural question then ranked it 1st (hybrid).
2. **BM25 number noise**: decimals were split into `0` + digits, so every table chunk
   matched `0`. *Fix*: keep decimals whole.
3. **RRF limitation**: for the exact-number query, dense retrieval ranks the row 25th
   (MiniLM cannot match literal numbers), so RRF credits only BM25's #1, and Table 11's
   GOOGL row (ranked highly by both) wins. RRF behaves as designed. *Not* fixed by
   tuning `k` or the pool size to this query; reranking moves it to 1st (measured), at
   ~1 s per question.

## Limitations

- **Sample size.** Two questions cannot distinguish configurations; the runner warns below
  20 evaluated questions. The target is 50-100 questions across several documents.
- **One document, one domain.** Results may not transfer to other documents or languages.
- **Human review is the bottleneck and has its own error.** One reviewer, no
  inter-annotator agreement yet.
- **Verifier judgments are LLM outputs.** They are probabilistic and vary between runs;
  only human-reviewed claims count as evidence of verifier quality.
- **Provider instability** (rate limits, overloads) affects answer runs; failures are
  recorded and excluded from answer metrics, which biases results if failures correlate
  with question difficulty.
- **Latency** is measured on one Windows laptop CPU and includes provider latency for the
  LLM stages.

## Next steps

1. Grow the dataset to 25+ reviewed questions, including 3-5 unanswerable ones, then
   re-run the retrieval comparison and decide on the default `RETRIEVAL_MODE`.
2. Run the answer evaluation and review a sample of claims to report citation precision
   before and after verification, and the verifier's false-positive rate.
3. Evaluate score-aware fusion against RRF, and an NLI model as a second verifier.
