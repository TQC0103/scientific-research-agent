import json
from pathlib import Path

import pytest

from app.evaluation.end_to_end import (
    NODE_TRACE_KEY,
    EndToEndReport,
    compare_with_baseline,
    end_to_end_json_schema,
    load_end_to_end_aggregate,
    render_end_to_end_report,
    run_end_to_end,
    write_end_to_end_outputs,
)
from app.evaluation.loader import load_suite
from app.models.claims import (
    CLAIM_VERIFICATION_CONTRACT_VERSION,
    AtomicClaim,
    ClaimAssessment,
    ClaimEvidenceLink,
    ClaimVerificationBundle,
)
from evaluation import run as evaluation_run

SUITE_PATH = Path("evaluation/suites/v0_5/development_10.json")


def _two_case_suite():
    suite = load_suite(SUITE_PATH)
    answer_case = next(case for case in suite.cases if case.expected.decision == "answer")
    abstain_case = next(
        case
        for case in suite.cases
        if case.expected.decision == "abstain" and not case.gold_evidence
    )
    return suite.model_copy(update={"cases": [answer_case, abstain_case]})


def _supported_bundle(answer: str) -> ClaimVerificationBundle:
    return ClaimVerificationBundle(
        contract_version=CLAIM_VERIFICATION_CONTRACT_VERSION,
        answer=answer,
        evidence_count=1,
        claims=[
            AtomicClaim(
                claim_id="claim_1",
                claim_text=answer.split(" [1]", 1)[0],
                source_text=answer,
                requires_citation=True,
                citation_labels=[1],
            )
        ],
        assessments=[
            ClaimAssessment(
                claim_id="claim_1",
                verdict="supported",
                cited_evidence=[
                    ClaimEvidenceLink(
                        citation_label=1,
                        relationship="entails",
                        reason="The approved passage fully supports the answer.",
                    )
                ],
                reason="The claim is supported.",
            )
        ],
    )


def test_production_adapter_records_new_graph_nodes_automatically(monkeypatch) -> None:
    class FakeGraph:
        def stream(self, payload, config, *, stream_mode):
            assert stream_mode == "updates"
            yield {"discover": {"candidate_papers": [{"arxiv_id": "paper"}]}}
            yield {"future_node": {"future_field": 42}}

    monkeypatch.setattr(evaluation_run, "research_graph", FakeGraph())

    state = evaluation_run._invoke_with_node_trace(
        {"user_query": "Question?", "paper_ids": []}, {"recursion_limit": 30}
    )

    assert state["future_field"] == 42
    assert [event["node"] for event in state[NODE_TRACE_KEY]] == [
        "discover",
        "future_node",
    ]


def test_committed_report_schema_matches_versioned_contract() -> None:
    committed = json.loads(
        Path("evaluation/schema/end-to-end-report.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert committed == end_to_end_json_schema()
    assert committed["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert committed["$id"].endswith("end-to-end-report.schema.json")


def test_end_to_end_runner_uses_graph_state_and_preserves_unknown_trace_fields() -> None:
    suite = _two_case_suite()
    answer_case, abstain_case = suite.cases
    reference = answer_case.expected.reference_answer
    body = f"{reference} [1]"
    bundle = _supported_bundle(body)
    evidence = answer_case.gold_evidence[0]

    def invoke(payload, config):
        assert config == {"recursion_limit": 30}
        assert payload["paper_ids"]
        if payload["user_query"] == answer_case.question:
            chunk = {
                "arxiv_id": evidence.paper_id,
                "versioned_id": evidence.versioned_id,
                "page": evidence.page,
                "section": evidence.section,
                "chunk_index": 0,
                "text": evidence.quote,
                "retrieval_score": 1.0,
            }
            return {
                "answer": f"{body}\n\nSources:\n[1] trusted",
                "evidence_sufficient": True,
                "synthesis_citation_valid": True,
                "claim_verification_status": "verified",
                "claim_verification": bundle.model_dump(mode="json"),
                "claim_verification_attempt_count": 1,
                "claim_revision_count": 0,
                "retrieval_attempt_counts": {evidence.paper_id: 1},
                "retrieved_chunks_by_paper": {evidence.paper_id: [chunk]},
                "verified_evidence": [chunk],
                "tool_errors": [],
                "future_graph_field": {"preserved": True},
                NODE_TRACE_KEY: [
                    {"node": "synthesize", "update": {"answer": body}},
                    {"node": "verify_claims", "update": {"status": "verified"}},
                ],
            }
        assert payload["user_query"] == abstain_case.question
        return {
            "answer": "Insufficient evidence to answer without guessing.",
            "evidence_sufficient": False,
            "synthesis_citation_valid": False,
            "claim_verification_status": "not_run",
            "claim_verification": {},
            "claim_verification_attempt_count": 0,
            "claim_revision_count": 0,
            "retrieval_attempt_counts": {abstain_case.papers[0].paper_id: 1},
            "retrieved_chunks_by_paper": {},
            "tool_errors": [],
        }

    report = run_end_to_end(
        suite,
        invoke,
        config_name="hybrid_verified",
        run_id="test-run",
    )

    assert report.aggregate.metrics["decision_accuracy"] == 1.0
    assert report.aggregate.metrics["abstention_accuracy"] == 1.0
    assert report.aggregate.metrics["retrieval_recall_at_k"] == 1.0
    assert report.aggregate.metrics["verifier_supported_claim_rate"] == 1.0
    assert report.aggregate.metrics["citation_complete_claim_rate"] == 1.0
    assert report.aggregate.metrics["llm_calls"] == 4
    assert report.aggregate.metric_directions["decision_accuracy"] == "higher"
    assert report.cases[0].trace["future_graph_field"] == {"preserved": True}
    assert [event.node for event in report.cases[0].node_trace] == [
        "synthesize",
        "verify_claims",
    ]
    assert NODE_TRACE_KEY not in report.cases[0].trace
    assert report.cases[0].llm_calls.total == 3
    assert report.cases[1].llm_calls.total == 1
    assert report.cases[1].retrieval.recall_at_k is None
    assert report.cases[1].failure_reasons == []


def test_execution_error_is_recorded_and_never_counts_as_correct_abstention() -> None:
    suite = _two_case_suite()
    abstain_case = next(case for case in suite.cases if case.expected.decision == "abstain")
    suite = suite.model_copy(update={"cases": [abstain_case]})

    def fail(payload, config):
        raise RuntimeError("model service unavailable")

    report = run_end_to_end(suite, fail, config_name="hybrid_verified", run_id="failed")
    result = report.cases[0]

    assert result.predicted_decision == "abstain"
    assert result.decision_correct is False
    assert result.execution_error == "RuntimeError: model service unavailable"
    assert "execution_error" in result.failure_reasons
    assert report.aggregate.execution_failures == 1
    assert report.aggregate.metrics["execution_failure_rate"] == 1.0


def test_revised_success_counts_two_claim_checks_and_one_repair() -> None:
    suite = _two_case_suite()
    case = suite.cases[0]
    suite = suite.model_copy(update={"cases": [case]})
    reference = case.expected.reference_answer
    body = f"{reference} [1]"
    bundle = _supported_bundle(body)
    evidence = case.gold_evidence[0]

    def invoke(payload, config):
        chunk = {
            "arxiv_id": evidence.paper_id,
            "versioned_id": evidence.versioned_id,
            "page": evidence.page,
            "section": evidence.section,
            "text": evidence.quote,
        }
        return {
            "answer": f"{body}\n\nSources:\n[1] trusted",
            "evidence_sufficient": True,
            "synthesis_citation_valid": True,
            "claim_verification_status": "verified",
            "claim_verification": bundle.model_dump(mode="json"),
            "claim_verification_attempt_count": 2,
            "claim_revision_count": 1,
            "retrieval_attempt_counts": {evidence.paper_id: 1},
            "retrieved_chunks_by_paper": {evidence.paper_id: [chunk]},
            "tool_errors": [],
        }

    report = run_end_to_end(suite, invoke, config_name="repair", run_id="repair")

    assert report.aggregate.metrics["revision_rate"] == 1.0
    assert report.aggregate.metrics["post_revision_success_rate"] == 1.0
    assert report.cases[0].llm_calls.total == 5


def test_outputs_are_reloadable_and_markdown_explains_metric_boundary(tmp_path: Path) -> None:
    suite = _two_case_suite().model_copy(update={"cases": []})
    report = run_end_to_end(suite, lambda payload, config: {}, config_name="empty", run_id="empty")

    write_end_to_end_outputs(report, tmp_path)

    aggregate = load_end_to_end_aggregate(tmp_path / "metrics.json")
    assert aggregate.run_id == "empty"
    full_report = EndToEndReport.model_validate_json(
        (tmp_path / "report.json").read_text(encoding="utf-8")
    )
    assert full_report.aggregate.run_id == "empty"
    assert (tmp_path / "per_case.jsonl").read_text(encoding="utf-8") == ""
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "not independently adjudicated entailment accuracy" in markdown
    assert render_end_to_end_report(report) == markdown
    json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))


def test_baseline_comparison_uses_only_registered_metrics() -> None:
    suite = _two_case_suite().model_copy(update={"cases": []})
    current_report = run_end_to_end(
        suite, lambda payload, config: {}, config_name="same", run_id="current"
    )
    baseline_metrics = dict(current_report.aggregate.metrics)
    baseline_metrics.update(
        {
            "decision_accuracy": 0.5,
            "execution_failure_rate": 0.2,
            "unregistered_future_metric": 999,
        }
    )
    current_metrics = dict(current_report.aggregate.metrics)
    current_metrics.update(
        {
            "decision_accuracy": 0.75,
            "execution_failure_rate": 0.1,
            "unregistered_future_metric": 0,
        }
    )
    baseline = current_report.aggregate.model_copy(
        deep=True, update={"run_id": "baseline", "metrics": baseline_metrics}
    )
    current = current_report.aggregate.model_copy(deep=True, update={"metrics": current_metrics})

    comparison = compare_with_baseline(current, baseline)

    outcomes = {item.metric: item.outcome for item in comparison.deltas}
    assert outcomes["decision_accuracy"] == "improved"
    assert outcomes["execution_failure_rate"] == "improved"
    assert "unregistered_future_metric" not in outcomes


def test_baseline_comparison_rejects_different_suite_shape() -> None:
    suite = _two_case_suite().model_copy(update={"cases": []})
    report = run_end_to_end(
        suite, lambda payload, config: {}, config_name="same", run_id="current"
    )
    baseline = report.aggregate.model_copy(update={"case_count": 1})

    with pytest.raises(ValueError, match="exact same suite cases"):
        compare_with_baseline(report.aggregate, baseline)
