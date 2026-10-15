"""Bounded recovery after verification, and the final answer/status decision.

1. Verify the first attempt's claims.
2. While some claims are unsupported or contradicted and the retry budget allows, retrieve
   more passages and regenerate with the rejected claims as feedback; verify again.
3. Keep the best attempt whose verification actually completed.
4. Unsupported claims are removed from the answer (or flagged, by policy); if no supported
   claim remains, abstain.

A technical failure never produces a "verified" status: if no attempt was fully verified,
the answer is returned as verification_failed. A failed recovery generation keeps the
previous (already verified) attempt.
"""

import logging

from backend.generation.answer_generator import AnswerGenerator, Attempt
from backend.generation.llm_client import LLMError, LLMUsage
from backend.schemas import AskResponse, OverallStatus, VerificationStatus, VerificationSummary

logger = logging.getLogger(__name__)


def _verify(generator: AnswerGenerator, attempt: Attempt, usage: LLMUsage) -> None:
    attempt.verified, attempt.quote_check_downgrades, attempt.error = generator.verifier.verify(
        attempt.claims, attempt.passages, usage
    )


def run_recovery(
    generator: AnswerGenerator, question: str, document_ids, first: Attempt, usage: LLMUsage, timings
) -> tuple[list[Attempt], str | None]:
    """Verify, then retry within the budget. Returns all attempts and any recovery error."""
    _verify(generator, first, usage)
    attempts = [first]
    config = generator.recovery
    current = first
    while current.verified and current.unresolved and len(attempts) <= config.max_attempts:
        number = len(attempts) + 1
        logger.info("Attempt %d: %d unresolved claim(s); recovering", number - 1, len(current.unresolved))
        try:
            # More evidence than the first attempt; the same top passages keep the same labels.
            passages = current.passages
            if config.recovery_top_k > len(current.passages):
                passages = generator.retrieve(question, config.recovery_top_k, document_ids, timings) or passages
            attempt = generator.generate(question, passages, number, usage, rejected=current.unresolved)
        except LLMError as exc:
            logger.warning("Recovery attempt %d failed: %s", number, exc.code)
            return attempts, f"recovery attempt {number} failed: {exc.code}"
        if attempt.claims:
            _verify(generator, attempt, usage)
        else:
            attempt.verified = True  # nothing to verify: the model abstained
        attempts.append(attempt)
        current = attempt
    return attempts, None


def choose_best(attempts: list[Attempt]) -> Attempt:
    """The fully verified attempt with the most supported claims (fewest unresolved, latest
    on ties); if none was fully verified, the last attempt."""
    verified = [a for a in attempts if a.verified and a.claims]
    if not verified:
        return next((a for a in reversed(attempts) if a.claims), attempts[-1])
    return max(verified, key=lambda a: (a.count(VerificationStatus.SUPPORTED), -len(a.unresolved), a.number))


def finalize(
    generator: AnswerGenerator, question: str, attempts: list[Attempt], recovery_error: str | None,
    timings, usage: LLMUsage, started: float,
) -> AskResponse:
    config = generator.recovery
    best = choose_best(attempts)
    summary = VerificationSummary(
        enabled=True, attempts=len(attempts), policy=config.policy,
        supported=best.count(VerificationStatus.SUPPORTED),
        failed=best.count(VerificationStatus.VERIFICATION_FAILED),
        quote_check_downgrades=sum(a.quote_check_downgrades for a in attempts),
        error=recovery_error or best.error,
    )

    if not best.verified:
        # Never presented as verified: the claims are shown with their failure status.
        return generator._response(question, best, [], OverallStatus.VERIFICATION_FAILED,
                                   summary, timings, usage, started)

    unresolved = best.unresolved
    if config.policy == "flag":
        summary.unresolved = len(unresolved)
        overall = OverallStatus.UNRESOLVED_CLAIMS if unresolved else OverallStatus.VERIFIED
        return generator._response(question, best, [], overall, summary, timings, usage, started)

    kept = [c for c in best.claims if c.verification_status == VerificationStatus.SUPPORTED]
    summary.removed = len(unresolved)
    if not kept:
        reason = ("The statements generated from the retrieved passages could not be verified against "
                  "their citations, so no answer is given.")
        return generator._response(question, best, unresolved, OverallStatus.INSUFFICIENT_EVIDENCE,
                                   summary, timings, usage, started, abstain_reason=reason)
    best.claims = kept
    return generator._response(question, best, unresolved, OverallStatus.VERIFIED, summary, timings, usage, started)

