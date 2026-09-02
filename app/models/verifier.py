import json
import re
from typing import Any

from pydantic import BaseModel, Field


class EvidenceVerification(BaseModel):
    sufficient: bool
    reason: str = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)
    suggested_query: str | None = None
    supported_evidence: list[int] = Field(default_factory=list)


SEMANTIC_ANCHOR_RULES = (
    (
        "an electrical-energy measurement",
        re.compile(
            r"\b(electrical energy|electricity|energy consumption|power consumption)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(electrical energy|electricity|energy consumption|power consumption|"
            r"kwh|kilowatt(?:-hours?)?|joules?|watt(?:-hours?)?)\b",
            re.IGNORECASE,
        ),
        "electrical energy consumption kWh joules power measurement",
    ),
    (
        "the ImageNet benchmark",
        re.compile(r"\bimagenet\b", re.IGNORECASE),
        re.compile(r"\bimagenet\b", re.IGNORECASE),
        "ImageNet evaluation results table",
    ),
    (
        "a top-1 accuracy measurement",
        re.compile(r"\btop[- ]?1\b.*\baccurac(?:y|ies)\b", re.IGNORECASE),
        re.compile(r"\btop[- ]?1\b.*\baccurac(?:y|ies)\b", re.IGNORECASE),
        "top-1 accuracy percent evaluation",
    ),
)


def get_llm(**kwargs: Any) -> Any:
    """Load the local Ollama adapter only on the production invocation path."""
    from app.models.llm import get_llm as factory

    return factory(**kwargs)


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


def build_verifier_prompt(
    question: str,
    evidence: list[dict],
    current_query: str | None = None,
    scope_instruction: str | None = None,
) -> str:
    """Build the production verifier prompt without invoking a model."""
    excerpts = []
    for number, item in enumerate(evidence, start=1):
        location = f"page {item['page']}, {item['section']}" if item.get("page") else "Abstract"
        excerpts.append(f"[{number}] {item['versioned_id']} — {location}\n{item['text']}")

    return f"""You are an evidence sufficiency verifier, not an answer writer.
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
- Question: "How much electrical energy did training consume?" Evidence reports only training
  duration, GPU count, FLOPs, or a generic training cost. Decision: sufficient=false. Compute,
  elapsed time, and financial cost are not electrical energy and must never be converted into it.
- Question: "What ImageNet top-1 accuracy is reported?" Evidence reports BLEU on WMT translation.
  Decision: sufficient=false. Explaining that BLEU is a different metric does not supply the
  requested benchmark result and does not prove document-wide absence.

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
question. The fields must be internally consistent: sufficient=true requires an empty
missing_information list, while any missing requested fact requires sufficient=false even
when other passages provide partial support. If sufficient is false, suggested_query must be a materially different,
keyword-focused search (use likely terminology, section/table names, entities, and metrics),
not a restatement of the current retrieval query.
"""


def apply_semantic_anchor_guard(
    result: EvidenceVerification,
    question: str,
    evidence: list[dict],
) -> EvidenceVerification:
    """Reject known high-risk metric substitutions after model verification."""
    if not result.sufficient:
        return result
    supported_text = "\n".join(
        str(evidence[number - 1].get("text", ""))
        for number in result.supported_evidence
        if 1 <= number <= len(evidence)
    )
    missing = [
        (description, query_terms)
        for description, question_pattern, evidence_pattern, query_terms in SEMANTIC_ANCHOR_RULES
        if question_pattern.search(question) and not evidence_pattern.search(supported_text)
    ]
    if not missing:
        return result
    descriptions = [description for description, _ in missing]
    result.sufficient = False
    result.reason = (
        "The selected passages do not explicitly report the required semantic anchor(s): "
        + "; ".join(descriptions)
        + ". Related compute, duration, task, or metric evidence cannot substitute for them."
    )
    result.missing_information = list(dict.fromkeys([*result.missing_information, *descriptions]))
    result.suggested_query = (
        f"{question} explicit {' '.join(query_terms for _, query_terms in missing)}"
    )
    result.supported_evidence = []
    return result


def parse_verifier_response(
    content: Any,
    *,
    evidence_count: int,
    fallback_query: str,
) -> EvidenceVerification:
    """Validate and apply the production verifier's fail-closed repairs."""
    payload = _extract_json(content)
    result = EvidenceVerification.model_validate(payload)
    valid = sorted(
        {number for number in result.supported_evidence if 1 <= number <= evidence_count}
    )
    result.supported_evidence = valid
    if result.sufficient and result.missing_information:
        result.sufficient = False
        result.suggested_query = result.suggested_query or fallback_query
    if result.sufficient and not valid:
        result.sufficient = False
        result.reason = "Verifier marked evidence sufficient but identified no supporting passage."
        result.missing_information = ["A directly supporting passage."]
        result.suggested_query = result.suggested_query or fallback_query
    if not result.sufficient:
        if not result.missing_information:
            result.missing_information = ["At least one requested element remains unsupported."]
        result.suggested_query = result.suggested_query or fallback_query
    return result


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

    prompt = build_verifier_prompt(question, evidence, current_query, scope_instruction)
    try:
        result = parse_verifier_response(
            get_llm(temperature=0, num_predict=400).invoke(prompt).content,
            evidence_count=len(evidence),
            fallback_query=question,
        )
        result = apply_semantic_anchor_guard(result, question, evidence)
    except Exception as exc:
        raise ValueError(f"Invalid verifier response: {exc}") from exc
    return result
