import json
from typing import Any

from pydantic import BaseModel, Field

from app.models.llm import get_llm


class EvidenceVerification(BaseModel):
    sufficient: bool
    reason: str = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)
    suggested_query: str | None = None
    supported_evidence: list[int] = Field(default_factory=list)


def _extract_json(content: Any) -> dict:
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
    raise ValueError("Verifier did not return a JSON object.")


def verify_evidence(
    question: str,
    evidence: list[dict],
    current_query: str | None = None,
    scope_instruction: str | None = None,
) -> EvidenceVerification:
    """Use the local reasoning model to decide whether passages answer the question."""
    if not evidence:
        return EvidenceVerification(
            sufficient=False,
            reason="No evidence was retrieved.",
            missing_information=["Relevant passages that directly address the question."],
            suggested_query=question,
        )

    excerpts = []
    for number, item in enumerate(evidence, start=1):
        location = f"page {item['page']}, {item['section']}" if item.get("page") else "Abstract"
        excerpts.append(f"[{number}] {item['versioned_id']} — {location}\n{item['text']}")

    prompt = f"""You are an evidence sufficiency verifier, not an answer writer.
Decide whether an answer writer can satisfy the verification scope for the user's exact
question using only the supplied passages. For a paper-specific scope in a multi-paper
question, judge only whether that paper supplies its own side; do not require the complete
cross-paper comparison. Judge semantic support, not wording or presentation format. Do not
invent extra requirements beyond the question: a stated mechanism or advantage can answer
"how" or "why" without an experiment; prose can answer a request for a numbered list without
being pre-enumerated. Do not require an explicit comparison unless the user asks for an
empirical comparison. If a passage directly states the requested fact, accept it.

Topical similarity alone is not enough. Numeric, comparative, and multi-part questions need
the requested values, sides, or parts. Negative or exhaustive claims (for example, that a
paper never reports something) need evidence with enough scope; absence from a few retrieved
passages is not proof. Do not use your own knowledge.

Calibration examples:
- Question: "Why use several sensors instead of one?" Evidence: "Several sensors preserve
  different frequency bands; one sensor averages them and loses this information." Decision:
  sufficient=true. This states the requested rationale; no performance experiment is needed.
- Question: "Does the report contain no safety experiment?" Evidence: three passages that do
  not mention safety. Decision: sufficient=false. A few silent passages cannot prove absence
  from the report.
- Question: "How is event time encoded?" Evidence: "Components exchange information globally,
  and a mask hides future events." Decision: sufficient=false. Interaction and masking do not
  state how time itself is encoded; retrieve evidence about timestamps or time encoding.
- Original request: "Compare paper A and paper B." Scope: "paper A only." Evidence from paper A
  states its method and result but never mentions paper B. Decision: sufficient=true for paper A.
  In paper-specific verification, absence of the other paper is irrelevant; each paper is checked
  separately before the final comparison.
- Original request: "Compare device A and B on architecture and energy use." Scope: "device A
  paper only." Evidence explains A's architecture but gives no energy measurement or energy-use
  method. Decision: sufficient=false for A, missing_information=["device A energy use"]. Every
  requested comparison dimension must be supported for that paper; coverage of one dimension
  cannot substitute for another.

First identify exactly what the question requests without strengthening it. Then check whether
each requested element can be stated as a faithful paraphrase of one or more passages. Mark
sufficient=true when all requested elements are supported; do not demand extra detail.

Current retrieval query:
{current_query or question}

Verification scope:
{scope_instruction or "Assess whether the passages support the complete question."}

Question:
{question}

Evidence:
{chr(10).join(excerpts)}

Return exactly one JSON object with this schema:
{{
  "sufficient": true or false,
  "reason": "brief evidence-based reason",
  "missing_information": ["specific missing fact or scope"],
  "suggested_query": "one focused retrieval query, or null if sufficient",
  "supported_evidence": [1, 2]
}}

supported_evidence contains only passage numbers that directly support answering the
question. If sufficient is false, suggested_query must be a materially different,
keyword-focused search (use likely terminology, section/table names, entities, and metrics),
not a restatement of the current retrieval query.
"""
    try:
        payload = _extract_json(get_llm(temperature=0, num_predict=400).invoke(prompt).content)
        result = EvidenceVerification.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Invalid verifier response: {exc}") from exc

    valid = sorted({number for number in result.supported_evidence if 1 <= number <= len(evidence)})
    result.supported_evidence = valid
    if not result.sufficient and not result.missing_information and valid:
        # Repair a contradictory boolean when the structured explanation says
        # nothing is missing and identifies supporting passages.
        result.sufficient = True
        result.suggested_query = None
    if result.sufficient and not valid:
        result.sufficient = False
        result.reason = "Verifier marked evidence sufficient but identified no supporting passage."
        result.missing_information = ["A directly supporting passage."]
        result.suggested_query = result.suggested_query or question
    return result
