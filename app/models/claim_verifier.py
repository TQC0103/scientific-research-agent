"""LLM-backed atomic claim extraction and evidence verification."""

from __future__ import annotations

import json
from typing import Any

from app.models.claims import (
    CLAIM_VERIFICATION_CONTRACT_VERSION,
    ClaimVerificationBundle,
)


def get_llm(**kwargs: Any) -> Any:
    """Load the local Ollama adapter only on the production invocation path."""
    from app.models.llm import get_llm as factory

    return factory(**kwargs)


def answer_body(answer: str) -> str:
    """Remove the deterministic metadata block before claim extraction."""
    return answer.split("\n\nSources:", 1)[0].strip()


def _extract_json(content: Any) -> dict[str, Any]:
    text = str(content).strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Claim verifier did not return a JSON object.")


def build_claim_verifier_prompt(
    answer: str,
    evidence: list[dict[str, Any]],
    question: str | None = None,
) -> str:
    """Build one bounded extraction-and-verification prompt."""
    body = answer_body(answer)
    if not body:
        raise ValueError("Claim verification requires a non-empty answer body.")
    if not evidence:
        raise ValueError("Claim verification requires verifier-approved evidence.")
    excerpts = []
    for number, item in enumerate(evidence, start=1):
        source_id = item.get("versioned_id") or item.get("arxiv_id") or "unknown"
        location = (
            f"page {item['page']}, {item.get('section') or 'Unknown section'}"
            if item.get("page")
            else "Abstract"
        )
        excerpts.append(f"[{number}] {source_id} — {location}\n{item.get('text', '')}")

    schema = json.dumps(
        ClaimVerificationBundle.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    return f"""You are a claim-level evidence verifier. Analyze the answer using only the
verifier-approved evidence below. Do not repair, rewrite, or answer the question.

Split every substantive answer assertion into minimal atomic claims. A compound sentence may
produce multiple claims with the same exact source_text. Include organizational text only when
needed to mark it not_required. Do not omit an unsupported or uncited scientific claim.

For every claim:
- claim_text is one normalized atomic assertion;
- source_text is an exact, contiguous substring copied from the answer;
- claim IDs are claim_1, claim_2, ... in answer order;
- citation_labels are exactly the unique numeric labels visibly present in source_text, in order;
- requires_citation is true for factual, numeric, comparative, methodological, causal, or
  paper-specific assertions, even when the source_text has no citation;
- requires_citation is false only for purely organizational or conversational text, which must
  have no citation labels.

Assess each attached label against the matching numbered passage only. Use entails when that
passage fully establishes the atomic claim, partial when it establishes only a qualified part,
and does_not_support for a topical, contradictory, or wrong citation. Assess every attached
label once in source order. Do not use outside knowledge.

Derive verdicts consistently: any entails link means supported; otherwise any partial link means
partial; otherwise a citation-required claim is unsupported. A required claim with no citation
is unsupported. A claim not requiring citation is not_required and has no evidence links.

Echo the exact answer body and evidence_count={len(evidence)}. Use contract_version
{CLAIM_VERIFICATION_CONTRACT_VERSION}. Return exactly one JSON object matching this JSON Schema,
with no Markdown or commentary:
{schema}

Original question (context only):
{question or "Not supplied."}

Exact answer body:
{body}

Verifier-approved evidence:
{chr(10).join(excerpts)}
"""


def parse_claim_verifier_response(
    content: Any,
    *,
    expected_answer: str,
    evidence_count: int,
) -> ClaimVerificationBundle:
    """Validate model output against immutable invocation inputs."""
    payload = _extract_json(content)
    if payload.get("answer") != expected_answer:
        raise ValueError("Claim verifier changed the answer body.")
    if payload.get("evidence_count") != evidence_count:
        raise ValueError("Claim verifier changed the supplied evidence count.")
    return ClaimVerificationBundle.model_validate(payload)


def verify_answer_claims(
    answer: str,
    evidence: list[dict[str, Any]],
    question: str | None = None,
) -> ClaimVerificationBundle:
    """Extract and verify claims against verifier-approved passages only."""
    body = answer_body(answer)
    prompt = build_claim_verifier_prompt(body, evidence, question)
    try:
        response = get_llm(temperature=0, num_predict=1800).invoke(prompt)
        return parse_claim_verifier_response(
            response.content,
            expected_answer=body,
            evidence_count=len(evidence),
        )
    except Exception as exc:
        raise ValueError(f"Invalid claim-verifier response: {exc}") from exc
