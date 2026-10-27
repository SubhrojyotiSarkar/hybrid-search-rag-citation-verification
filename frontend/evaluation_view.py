"""Evaluation tab: dataset health, starting runs, retrieval and answer metrics, human review.

Data comes from the backend's /api/evaluation endpoints. Chart colors use the validated
reference palette (fixed categorical order; light and dark steps checked with the palette
validator), and every chart has a table view, so no reading depends on color alone.
Verifier-judged numbers are always labelled as such; only human reviews are ground truth.
"""

import altair as alt
import pandas as pd
import streamlit as st

MIN_QUESTIONS = 20

# Categorical slots in fixed order (never cycled), light and dark steps.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
# Reserved status colors (never reused as series colors), always shown with text labels.
STATUS_COLORS = {
    "verified_within_provided_evidence": "#0ca30c",   # good
    "contains_unresolved_claims": "#fab219",          # warning
    "unverified": "#ec835a",                          # serious
    "verification_failed": "#d03b3b",                 # critical
    "insufficient_evidence": "#8a8984",               # neutral: an abstention is not an error
}
VERDICT_LABELS = {"supported": "Supported", "contradicted": "Contradicted",
                  "insufficient_evidence": "Insufficient evidence"}


def _theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def _pct(value) -> str:
    return "–" if value is None else f"{value:.0%}"


# ---- dataset + running ---------------------------------------------------------------------

def render_dataset(api_request) -> dict:
    response = api_request("GET", "/api/evaluation/dataset", timeout=10.0)
    dataset = response.json() if response is not None and response.is_success else {}
    st.subheader("Evaluation dataset")
    if dataset.get("error"):
        st.error(f"The dataset file is invalid: {dataset['error']}")
        return dataset
    by_type = dataset.get("by_type", {})
    cols = st.columns(4)
    cols[0].metric("Questions", dataset.get("questions", 0))
    cols[1].metric("Human-reviewed", dataset.get("reviewed", 0))
    cols[2].metric("Unanswerable", by_type.get("unanswerable", 0))
    cols[3].metric("Keyword / paraphrase / factual",
                   f"{by_type.get('keyword', 0)} / {by_type.get('paraphrase', 0)} / {by_type.get('factual', 0)}")
    if dataset.get("questions", 0) < MIN_QUESTIONS:
        st.warning(f"Only {dataset.get('questions', 0)} question(s). Results are anecdotal below about "
                   f"{MIN_QUESTIONS}: don't change defaults or report improvements from them.")
    if not by_type.get("unanswerable"):
        st.info("No unanswerable questions yet, so abstention behaviour cannot be measured.")
    st.caption(f"File: {dataset.get('path', 'evaluation/datasets/questions.jsonl')}")
    return dataset


def render_run_controls(api_request, error_message, dataset: dict) -> None:
    with st.expander("Start a new evaluation run"):
        with st.form("start_run"):
            kind = st.radio("Run", ["retrieval", "answers"], horizontal=True,
                            help="retrieval: no LLM calls, fast. answers: full pipeline with verification, "
                                 "paced to the Gemini free-tier limit (slow).")
            modes = st.multiselect("Retrievers (retrieval runs)", ["dense", "bm25", "hybrid", "hybrid_rerank"],
                                   default=["dense", "bm25", "hybrid", "hybrid_rerank"])
            limit = st.number_input("Questions (answer runs, 0 = all)", min_value=0, value=0, step=1)
            questions = dataset.get("questions", 0) if not limit else min(limit, dataset.get("questions", 0))
            st.caption(f"Answer runs use about 2-4 LLM calls per question at 5 calls/minute: "
                       f"roughly {max(1, round(questions * 3 / 5))} minute(s) for {questions} question(s).")
            if st.form_submit_button("Start"):
                payload = {"kind": kind, "modes": modes or ["hybrid"]}
                if kind == "answers" and limit:
                    payload["limit"] = int(limit)
                response = api_request("POST", "/api/evaluation/run", json=payload, timeout=30.0)
                if response is not None and response.status_code == 202:
                    st.success(f"Started {kind} job {response.json()['job_id']}.")
                elif response is not None:
                    st.error(error_message(response))
    render_jobs(api_request)


@st.fragment(run_every="5s")
def render_jobs(api_request) -> None:
    """Auto-refreshes every 5 s, only this fragment (not the whole page)."""
    response = api_request("GET", "/api/evaluation/jobs", timeout=10.0)
    jobs = response.json() if response is not None and response.is_success else []
    if not jobs:
        return
    job = jobs[0]
    label = f"Latest job {job['job_id']} ({job['kind']}): {job['status']}"
    if job["status"] == "running":
        fraction = job["done"] / job["total"] if job["total"] else 0.0
        st.progress(fraction, text=f"{label} · {job['done']}/{job['total'] or '?'} · {job['message']}")
    elif job["status"] == "failed":
        st.error(f"{label}: {job['error']}")
    else:
        st.success(f"{label} → run {job['run_id']} (select it below)")


# ---- retrieval runs ----------------------------------------------------------------------------

def render_retrieval_run(run: dict) -> None:
    summary = pd.DataFrame(run.get("summary", []))
    data = run.get("dataset", {})
    st.caption(f"{data.get('evaluated', '?')} evaluated question(s) ({data.get('reviewed', '?')} human-reviewed) · "
               f"dataset {data.get('fingerprint', '?')} · git {run.get('git', '?')}")
    if (data.get("evaluated") or 0) < MIN_QUESTIONS:
        st.warning("Too few questions for conclusions; treat differences between configurations as anecdotal.")
    for problem in data.get("excluded", []):
        st.warning(f"Excluded: {problem}")
    for fallback in run.get("fallbacks", []):
        st.warning(f"Reranker fallback: {fallback}")
    if summary.empty:
        st.info("This run has no results.")
        return

    configs = list(summary["config"])
    palette = CATEGORICAL[_theme()]
    hit_columns = [c for c in summary.columns if c.startswith("hit@")]

    left, right = st.columns(2)
    with left:
        st.markdown("**Hit@k: share of questions with relevant evidence in the top k**")
        shown = configs[: len(palette)]
        if len(configs) > len(palette):
            st.caption(f"Chart shows the first {len(palette)} configurations; all are in the table.")
        long = summary[summary["config"].isin(shown)].melt(
            id_vars="config", value_vars=hit_columns, var_name="metric", value_name="hit")
        long["k"] = long["metric"].str.removeprefix("hit@").astype(int)
        color = alt.Color("config:N", scale=alt.Scale(domain=shown, range=palette[: len(shown)]),
                          legend=alt.Legend(title=None, orient="bottom", columns=2))
        base = alt.Chart(long).encode(
            x=alt.X("k:O", title="k (passages)"),
            y=alt.Y("hit:Q", title="hit@k", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=color,
            tooltip=[alt.Tooltip("config:N"), alt.Tooltip("k:O"), alt.Tooltip("hit:Q", format=".0%")],
        )
        st.altair_chart(base.mark_line(strokeWidth=2) + base.mark_point(size=64, filled=True), width="stretch")
    with right:
        mrr_column = next(c for c in summary.columns if c.startswith("MRR@"))
        st.markdown(f"**{mrr_column}: how high the first relevant passage ranks (1.0 = always first)**")
        st.altair_chart(_hbar(summary, mrr_column, "config", configs, palette[0], value_format=".2f",
                              domain=[0, 1]), width="stretch")
    if "mean_ms" in summary:
        st.markdown("**Mean retrieval latency per question (ms)**")
        st.altair_chart(_hbar(summary, "mean_ms", "config", configs, palette[0], value_format=".0f"), width="stretch")

    st.markdown("**All metrics**")
    st.dataframe(summary, hide_index=True, width="stretch")
    ranks = pd.DataFrame(run.get("first_relevant_rank", {})).T
    if not ranks.empty:
        st.markdown("**Rank of the first relevant passage per question** (blank = not in the retrieved depth)")
        st.dataframe(ranks, width="stretch")


def _hbar(frame: pd.DataFrame, value: str, label: str, order: list, color: str, value_format: str, domain=None):
    x = alt.X(f"{value}:Q", title=None, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    base = alt.Chart(frame).encode(
        y=alt.Y(f"{label}:N", sort=order, title=None),
        x=x,
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q", format=value_format)],
    )
    bars = base.mark_bar(color=color, cornerRadiusEnd=4, size=14)
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text(f"{value}:Q", format=value_format))
    return bars + labels


# ---- answer runs ------------------------------------------------------------------------------

def render_answer_metrics(run: dict) -> None:
    metrics = run["metrics"]
    answers, judged, human = metrics["answers"], metrics["verifier_judged"], metrics["human"]
    settings = run.get("settings", {})
    st.caption(f"{answers['questions']} question(s) · model {settings.get('llm_model', '?')} · retrieval "
               f"{settings.get('retrieval_mode', '?')} · verification {'on' if settings.get('verification_enabled') else 'off'} · "
               f"max recovery {settings.get('max_recovery_attempts', '?')} · git {run.get('git', '?')} · "
               f"{run.get('duration_seconds', '?')} s")
    if answers["questions"] < MIN_QUESTIONS:
        st.warning("Too few questions for conclusions; these numbers describe this run only.")

    st.markdown("#### Answers")
    cols = st.columns(4)
    cols[0].metric("Answered", f"{answers['answered']}/{answers['questions']}",
                   help="Questions that produced a response (errors are provider or technical failures).")
    cols[1].metric("Correct abstentions", _pct(answers["unanswerable"]["rate"]),
                   help=f"Unanswerable questions answered with insufficient evidence "
                        f"({answers['unanswerable']['correct_abstentions']}/{answers['unanswerable']['n']}).")
    cols[2].metric("False abstentions", _pct(answers["answerable"]["rate"]),
                   help=f"Answerable questions where the system abstained "
                        f"({answers['answerable']['false_abstentions']}/{answers['answerable']['n']}).")
    cols[3].metric("Needed correction", _pct(answers["required_correction"]["rate"]),
                   help="Answers whose first attempt had a claim the verifier rejected.")

    if answers["overall_status"]:
        status = pd.DataFrame([{"status": k, "answers": v} for k, v in answers["overall_status"].items()])
        domain = [s for s in STATUS_COLORS if s in set(status["status"])]
        chart = alt.Chart(status).encode(
            y=alt.Y("status:N", sort=domain, title=None),
            x=alt.X("answers:Q", title="answers", axis=alt.Axis(tickMinStep=1)),
            tooltip=["status:N", "answers:Q"],
        )
        bars = chart.mark_bar(cornerRadiusEnd=4, size=14).encode(
            color=alt.Color("status:N", scale=alt.Scale(domain=domain, range=[STATUS_COLORS[s] for s in domain]),
                            legend=None))
        st.altair_chart(bars + chart.mark_text(align="left", dx=4).encode(text="answers:Q"), width="stretch")

    st.markdown("#### Verifier-judged (automatic, **not** ground truth)")
    cols = st.columns(4)
    cols[0].metric("First-attempt claims", judged["first_attempt_claims"])
    cols[1].metric("Rejected by verifier", _pct(judged["first_attempt_not_supported_rate"]))
    cols[2].metric("Verification failures", judged["verification_failed_claims"],
                   help="Claims the verifier could not check (technical errors); never counted as supported.")
    cols[3].metric("Quote-check rejections", judged["quote_check_downgrades"],
                   help="Verdicts rejected because the verifier's quote was not in the cited passage.")

    st.markdown("#### Human-judged (ground truth, from your reviews)")
    before, after, agreement = human["before_verification"], human["after_verification"], human["verifier_agreement"]
    st.caption(f"{human['reviewed_claims']} of {human['total_claims']} claim(s) reviewed. Review more below.")
    cols = st.columns(4)
    cols[0].metric("Citation precision before verification", _pct(before["citation_precision"]),
                   help=f"First-attempt claims whose citations support them ({before['reviewed']} reviewed).")
    delta = None
    if before["citation_precision"] is not None and after["citation_precision"] is not None:
        delta = f"{after['citation_precision'] - before['citation_precision']:+.0%}"
    cols[1].metric("Citation precision after verification", _pct(after["citation_precision"]), delta=delta,
                   help=f"Claims shown in final answers ({after['reviewed']} reviewed).")
    cols[2].metric("Verifier agreement", _pct(agreement["accuracy"]), help=f"Over {agreement['n']} reviewed claims.")
    cols[3].metric("Verifier false positives", agreement["false_positives"],
                   help="Claims the verifier marked supported but you did not.")
    if agreement["n"]:
        confusion = pd.DataFrame(agreement["confusion"]).T
        confusion.index = [f"verifier: {VERDICT_LABELS[v]}" for v in confusion.index]
        confusion.columns = [f"human: {VERDICT_LABELS[v]}" for v in confusion.columns]
        st.markdown("**Verifier vs human** (rows: verifier verdict, columns: your verdict)")
        st.dataframe(confusion, width="stretch")
        st.caption(f"Supported precision {_pct(agreement['supported_precision'])} · supported recall "
                   f"{_pct(agreement['supported_recall'])} · false negatives {agreement['false_negatives']}")

    latency = metrics["latency"]
    if latency:
        st.markdown("#### Latency and cost per question")
        rows = [{"measure": field, **values} for field, values in latency.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        stages = pd.DataFrame([{"stage": f.removesuffix("_ms"), "mean_ms": latency[f]["mean"]}
                               for f in ("retrieval_ms", "generation_ms", "verification_ms") if f in latency])
        if not stages.empty:
            st.markdown("**Mean time per stage (ms)**")
            st.altair_chart(_hbar(stages, "mean_ms", "stage", list(stages["stage"]), CATEGORICAL[_theme()][0], ".0f"),
                            width="stretch")

    errors = [q for q in run["questions"] if q.get("error")]
    if errors:
        st.markdown("#### Errors")
        for q in errors:
            detail = q["error"].get("detail")
            st.error(f"{q['question_id']}: {q['error']['code']}" + (f" ({detail})" if detail else ""))


# ---- review -----------------------------------------------------------------------------------

def render_review(run: dict, api_request, error_message, page_display, is_admin: bool = True) -> None:
    rows = run.get("claim_rows", [])
    reviews = run.get("reviews", {})
    st.markdown("#### Review claims")
    if not rows:
        st.info("This run has no claims to review.")
        return
    st.caption("Judge only whether the cited passages support the claim, not whether it is true in the world. "
               "Saved reviews are stored separately from the run and update the human-judged metrics.")
    c1, c2, c3 = st.columns(3)
    states = c1.multiselect("Final state", ["shown", "removed", "superseded"], default=["shown", "removed"])
    verdicts = c2.multiselect("Verifier verdict", sorted({r["claim"]["verification_status"] for r in rows}),
                              default=sorted({r["claim"]["verification_status"] for r in rows}))
    only_unreviewed = c3.checkbox("Only unreviewed", value=True)
    selected = [r for r in rows if r["state"] in states and r["claim"]["verification_status"] in verdicts
                and (not only_unreviewed or r["key"] not in reviews)]
    st.caption(f"{len(selected)} claim(s) match · {len(reviews)} reviewed in total")
    questions = {q["question_id"]: q for q in run["questions"]}

    for row in selected[:20]:
        claim = row["claim"]
        existing = reviews.get(row["key"])
        title = f"{row['key']} · {row['state']} · verifier: {claim['verification_status']}"
        if existing:
            title += f" · you: {existing['human_verdict']}"
        with st.expander(title):
            st.markdown(f"**Question:** {questions[row['question_id']]['question']}")
            st.markdown(f"**Claim:** {claim['text']}")
            if claim.get("verification_reason"):
                st.caption(f"Verifier: {claim['verification_reason']}")
            if claim.get("evidence_quote"):
                st.caption(f'Verifier quote: "{claim["evidence_quote"]}"')
            for passage in row["passages"]:
                st.markdown(f"**[{passage['source_label']}]** {passage['filename']}, {page_display(passage)}")
                if passage.get("context"):
                    st.caption(f"Table context: {passage['context']}")
                st.text(passage["text"])
            if not row["passages"]:
                st.warning("No cited passage was provided for this claim.")
            with st.form(f"review_{row['key']}"):
                options = list(VERDICT_LABELS)
                index = options.index(existing["human_verdict"]) if existing else 0
                verdict = st.radio("Your verdict", options, index=index, format_func=VERDICT_LABELS.get,
                                   horizontal=True, key=f"verdict_{row['key']}")
                notes = st.text_input("Notes (optional)", value=existing["notes"] if existing else "",
                                      key=f"notes_{row['key']}")
                if st.form_submit_button("Save review", disabled=not is_admin):
                    response = api_request("PUT", f"/api/evaluation/runs/{run['run_id']}/reviews",
                                           json={"key": row["key"], "human_verdict": verdict, "notes": notes}, timeout=10.0)
                    if response is not None and response.is_success:
                        st.success("Saved.")
                        st.rerun()
                    elif response is not None:
                        st.error(error_message(response))
    if len(selected) > 20:
        st.caption("Showing the first 20; save or filter to see more.")


def render_question_inspector(run: dict) -> None:
    st.markdown("#### Inspect answers")
    records = run["questions"]
    if not records:
        return
    by_label = {f"{q['question_id']} ({q['type']}): {q['response']['overall_status'] if q['response'] else 'error'}": q
                for q in records}
    record = by_label[st.selectbox("Question", list(by_label))]
    st.markdown(f"**{record['question']}**")
    if record.get("should_not_claim"):
        st.caption("Should not claim: " + "; ".join(record["should_not_claim"]))
    if record["response"] is None:
        st.error(f"{record['error']['code']}: {record['error'].get('detail') or record['error']['message']}")
        return
    st.markdown(record["response"]["answer"])
    for attempt in record["attempts"]:
        with st.expander(f"Attempt {attempt['number']}: {len(attempt['claims'])} claim(s) before final selection"):
            for claim in attempt["claims"]:
                st.markdown(f"- {claim['text']}  \n  *{claim['verification_status']}*: {claim.get('verification_reason') or ''}")


# ---- tab entry point ------------------------------------------------------------------------------

def render_evaluation_tab(api_request, error_message, page_display, is_admin: bool = True) -> None:
    dataset = render_dataset(api_request)
    if is_admin:
        render_run_controls(api_request, error_message, dataset)
    else:
        st.caption("Read-only account: you can view runs and metrics; starting runs and reviewing need an admin.")

    st.subheader("Runs")
    response = api_request("GET", "/api/evaluation/runs", timeout=10.0)
    runs = response.json() if response is not None and response.is_success else []
    if not runs:
        st.info("No evaluation runs yet. Start one above, or run the CLI commands from the README.")
        return
    by_label = {f"{r['run_id']} ({r['kind']}"
                + (f", {r['reviewed_claims']} reviewed" if r["reviewed_claims"] is not None else "") + ")": r["run_id"]
                for r in runs}
    run_id = by_label[st.selectbox("Run", list(by_label))]
    detail = api_request("GET", f"/api/evaluation/runs/{run_id}", timeout=30.0)
    if detail is None or not detail.is_success:
        st.error(error_message(detail) if detail is not None else "Backend unreachable.")
        return
    run = detail.json()
    if run["kind"] == "retrieval":
        render_retrieval_run(run)
    else:
        render_answer_metrics(run)
        render_review(run, api_request, error_message, page_display, is_admin)
        render_question_inspector(run)
