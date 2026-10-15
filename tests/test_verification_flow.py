"""End-to-end citation verification and recovery through /api/chat/ask.

Generation and verification are scripted (FakeLLM); the scripted verifier parses the real
verifier prompt and quotes real passage text, so the code's quote check is exercised on
genuine evidence. Includes deliberately unsupported and contradicted claims.
"""

import re

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.generation.llm_client import LLMError
from backend.schemas import GeneratedAnswer, GeneratedClaim, GeneratedVerdict, GeneratedVerification
from tests.conftest import FakeLLM, make_pdf

ML_PDF = make_pdf([
    "L2 regularization adds a penalty proportional to the squared magnitude of the weights.",
    "Regularization does not guarantee better test accuracy on every dataset.",
    "Dropout randomly disables units during training.",
])

SUPPORTED = "L2 regularization penalizes the squared magnitude of the weights."
OVERSTATED = "L2 regularization always guarantees improved test accuracy."

_CLAIM_BLOCK = re.compile(r'<claim id="(claim_\d+)">\n<text>(.*?)</text>\n(.*?)\n</claim>', re.DOTALL)
_PASSAGE = re.compile(r'<passage label="(S\d+)"[^>]*>\n(?:\[Table context:.*?\]\n)?(.*?)\n</passage>', re.DOTALL)


def claims_in(prompt: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    return [(cid, text, _PASSAGE.findall(body)) for cid, text, body in _CLAIM_BLOCK.findall(prompt)]


def scripted_verifier(rules: dict[str, str]):
    """Verdict per claim: the first rule whose key occurs in the claim text decides
    ("supported", "insufficient_evidence", "contradicted", "fabricated"); default supported.
    Quotes are copied from the claim's first cited passage (or invented for "fabricated")."""

    def verify(prompt: str) -> GeneratedVerification:
        verdicts = []
        for claim_id, text, passages in claims_in(prompt):
            decision = next((v for k, v in rules.items() if k in text), "supported")
            label, passage_text = passages[0]
            quote = " ".join(passage_text.split()[:6])
            if decision == "fabricated":
                verdicts.append(GeneratedVerdict(claim_id=claim_id, verdict="supported", evidence_quote=
                                                 "this sentence is not in any passage", supporting_source_labels=[label],
                                                 explanation="Looks fine."))
            elif decision == "insufficient_evidence":
                verdicts.append(GeneratedVerdict(claim_id=claim_id, verdict=decision, evidence_quote="",
                                                 supporting_source_labels=[], explanation="Not established by the passage."))
            else:
                verdicts.append(GeneratedVerdict(claim_id=claim_id, verdict=decision, evidence_quote=quote,
                                                 supporting_source_labels=[label] if decision == "supported" else [],
                                                 explanation=f"The passage {decision}s this."))
        return GeneratedVerification(verdicts=verdicts)

    return verify


def answer_with(*texts: str):
    """Generation script: the given claims, each citing the first source in the prompt."""

    def generate(prompt: str) -> GeneratedAnswer:
        first = re.search(r'label="(S\d+)"', prompt).group(1)
        return GeneratedAnswer(answer_status="answered", missing_information="",
                               claims=[GeneratedClaim(text=t, source_labels=[first]) for t in texts])

    return generate


def ask(llm: FakeLLM, app_factory, monkeypatch, **env) -> dict:
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    get_settings.cache_clear()
    with TestClient(app_factory(llm_client=llm)) as client:
        client.post("/api/documents/upload", files={"file": ("ml.pdf", ML_PDF, "application/pdf")})
        response = client.post("/api/chat/ask", json={"question": "What does L2 regularization do?"})
    assert response.status_code == 200, response.text
    return response.json()


def test_supported_claims_are_verified(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED), verification=scripted_verifier({}))
    body = ask(llm, app_factory, monkeypatch)

    assert body["overall_status"] == "verified_within_provided_evidence"
    [claim] = body["claims"]
    assert claim["verification_status"] == "supported"
    assert claim["evidence_quote"]
    assert claim["citations"][0]["supports_claim"] is True
    summary = body["verification"]
    assert (summary["enabled"], summary["attempts"], summary["supported"], summary["removed"]) == (True, 1, 1, 0)
    assert body["metadata"]["llm_calls"] == 2  # generation + verification
    assert body["metadata"]["verification_ms"] is not None


def test_unsupported_claim_triggers_recovery_with_feedback(app_factory, monkeypatch):
    llm = FakeLLM(output=[answer_with(SUPPORTED, OVERSTATED), answer_with(SUPPORTED)],
                  verification=scripted_verifier({"always guarantees": "insufficient_evidence"}))
    body = ask(llm, app_factory, monkeypatch)

    assert body["overall_status"] == "verified_within_provided_evidence"
    assert body["verification"]["attempts"] == 2
    assert [c["text"] for c in body["claims"]] == [SUPPORTED]
    assert OVERSTATED not in body["answer"]
    # The regeneration prompt told the model which claim was rejected and why.
    retry_prompt = llm.stage_calls("generation")[1]["user"]
    assert "<rejected_claims>" in retry_prompt and OVERSTATED in retry_prompt
    assert body["metadata"]["llm_calls"] == 4


def test_claim_still_unsupported_after_retry_is_removed_transparently(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED, OVERSTATED),
                  verification=scripted_verifier({"always guarantees": "insufficient_evidence"}))
    body = ask(llm, app_factory, monkeypatch)

    assert body["overall_status"] == "verified_within_provided_evidence"
    assert [c["text"] for c in body["claims"]] == [SUPPORTED]
    [removed] = body["removed_claims"]
    assert removed["text"] == OVERSTATED
    assert removed["verification_status"] == "insufficient_evidence"
    assert removed["verification_reason"]
    assert OVERSTATED not in body["answer"] and "removed" in body["answer"]
    assert body["answer_status"] == "partially_answered"
    assert body["verification"]["removed"] == 1


def test_contradicted_claim_is_removed(app_factory, monkeypatch):
    contradicted = "Regularization guarantees better test accuracy on every dataset."
    llm = FakeLLM(output=answer_with(SUPPORTED, contradicted),
                  verification=scripted_verifier({"guarantees better": "contradicted"}))
    body = ask(llm, app_factory, monkeypatch, MAX_RECOVERY_ATTEMPTS=0)

    [removed] = body["removed_claims"]
    assert removed["verification_status"] == "contradicted"
    assert removed["citations"][0]["supports_claim"] is False


def test_no_supported_claims_means_abstention(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(OVERSTATED), verification=scripted_verifier({"always": "insufficient_evidence"}))
    body = ask(llm, app_factory, monkeypatch)

    assert body["overall_status"] == "insufficient_evidence"
    assert body["claims"] == []
    assert [c["text"] for c in body["removed_claims"]] == [OVERSTATED]
    assert "could not be verified" in body["answer"]


def test_fabricated_verifier_quote_is_rejected(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED, OVERSTATED), verification=scripted_verifier({"always": "fabricated"}))
    body = ask(llm, app_factory, monkeypatch, MAX_RECOVERY_ATTEMPTS=0)

    [removed] = body["removed_claims"]
    assert removed["text"] == OVERSTATED and removed["quote_check_failed"] is True
    assert body["verification"]["quote_check_downgrades"] == 1


def test_verifier_outage_is_reported_as_verification_failed(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED, OVERSTATED),
                  verification_error=LLMError("llm_rate_limited", "Quota reached."))
    body = ask(llm, app_factory, monkeypatch)

    assert body["overall_status"] == "verification_failed"
    assert body["claims"] and all(c["verification_status"] == "verification_failed" for c in body["claims"])
    assert body["verification"]["error"] == "llm_rate_limited"
    assert body["verification"]["supported"] == 0
    # No recovery is attempted when verification itself failed.
    assert len(llm.stage_calls("generation")) == 1


def test_recovery_generation_failure_keeps_the_verified_first_attempt(app_factory, monkeypatch):
    calls = {"n": 0}
    first = answer_with(SUPPORTED, OVERSTATED)

    def generate_then_fail(prompt):
        calls["n"] += 1
        if calls["n"] > 1:
            raise LLMError("llm_unavailable", "Could not reach the Gemini API.")
        return first(prompt)

    llm = FakeLLM(output=generate_then_fail, verification=scripted_verifier({"always": "insufficient_evidence"}))
    body = ask(llm, app_factory, monkeypatch)

    assert body["overall_status"] == "verified_within_provided_evidence"
    assert [c["text"] for c in body["claims"]] == [SUPPORTED]
    assert "recovery attempt 2 failed" in body["verification"]["error"]


def test_flag_policy_keeps_and_marks_unresolved_claims(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED, OVERSTATED), verification=scripted_verifier({"always": "insufficient_evidence"}))
    body = ask(llm, app_factory, monkeypatch, UNSUPPORTED_CLAIM_POLICY="flag", MAX_RECOVERY_ATTEMPTS=0)

    assert body["overall_status"] == "contains_unresolved_claims"
    assert len(body["claims"]) == 2 and body["removed_claims"] == []
    assert f"{OVERSTATED} [S" in body["answer"] and "[not supported by its citations]" in body["answer"]
    assert body["verification"]["unresolved"] == 1


def test_recovery_budget_is_respected(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED, OVERSTATED), verification=scripted_verifier({"always": "insufficient_evidence"}))
    body = ask(llm, app_factory, monkeypatch, MAX_RECOVERY_ATTEMPTS=0)

    assert body["verification"]["attempts"] == 1
    assert len(llm.stage_calls("generation")) == 1


@pytest.mark.parametrize("attempts", [1, 2])
def test_recovery_never_exceeds_the_configured_attempts(app_factory, monkeypatch, attempts):
    llm = FakeLLM(output=answer_with(OVERSTATED), verification=scripted_verifier({"always": "insufficient_evidence"}))
    body = ask(llm, app_factory, monkeypatch, MAX_RECOVERY_ATTEMPTS=attempts)

    assert len(llm.stage_calls("generation")) == 1 + attempts
    assert body["verification"]["attempts"] == 1 + attempts


def test_model_abstention_skips_verification(app_factory, monkeypatch):
    llm = FakeLLM(output=GeneratedAnswer(answer_status="insufficient_evidence", claims=[], missing_information="Not covered."),
                  verification=scripted_verifier({}))
    body = ask(llm, app_factory, monkeypatch)

    assert body["overall_status"] == "insufficient_evidence"
    assert llm.stage_calls("verification") == []


def test_verifier_sees_claims_and_passages_but_not_the_question(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED), verification=scripted_verifier({}))
    ask(llm, app_factory, monkeypatch)

    [call] = llm.stage_calls("verification")
    assert SUPPORTED in call["user"]
    assert "What does L2 regularization do?" not in call["user"]


def test_verification_can_be_disabled(app_factory, monkeypatch):
    llm = FakeLLM(output=answer_with(SUPPORTED), verification=scripted_verifier({}))
    body = ask(llm, app_factory, monkeypatch, VERIFICATION_ENABLED="false")

    assert body["overall_status"] == "unverified"
    assert body["verification"]["enabled"] is False
    assert llm.stage_calls("verification") == []
