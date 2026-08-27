import json
from pathlib import Path

import pytest

from app.evaluation.verifier import (
    load_verifier_benchmark,
    render_verifier_markdown,
    run_verifier_benchmark,
    write_verifier_outputs,
)

DEFINITION = Path("evaluation/suites/v0_5/verifier_development.json")
SOURCE_SUITE = Path("evaluation/suites/v0_5/development_10.json")


def _response(sufficient: bool, supported: list[int], suffix: str) -> str:
    return json.dumps(
        {
            "sufficient": sufficient,
            "reason": "The controlled snapshot was classified.",
            "missing_information": [] if sufficient else ["The requested fact is missing."],
            "suggested_query": None if sufficient else f"focused recovery query {suffix}",
            "supported_evidence": supported,
        }
    )


class PerfectGenerator:
    def __init__(self, benchmark):
        self.batches = [
            [
                _response(
                    case.initial.expected_sufficient,
                    case.initial.expected_supported_indices,
                    case.case_id,
                )
                for case in benchmark.cases
            ],
            [
                _response(
                    case.recovery.expected_sufficient,
                    case.recovery.expected_supported_indices,
                    case.case_id,
                )
                for case in benchmark.cases
                if not case.initial.expected_sufficient
            ],
        ]
        self.calls = 0

    def __call__(self, prompts: list[str]) -> list[str]:
        responses = self.batches[self.calls]
        self.calls += 1
        assert len(prompts) == len(responses)
        return responses


def test_verifier_benchmark_loads_balanced_controlled_snapshots() -> None:
    benchmark = load_verifier_benchmark(DEFINITION, SOURCE_SUITE)
    assert len(benchmark.cases) == 22
    assert sum(case.initial.expected_sufficient for case in benchmark.cases) == 10
    assert sum(not case.initial.expected_sufficient for case in benchmark.cases) == 12
    assert sum(
        not case.initial.expected_sufficient and case.recovery.expected_sufficient
        for case in benchmark.cases
    ) == 10
    assert sum(not case.recovery.expected_sufficient for case in benchmark.cases) == 2


def test_perfect_verifier_report_covers_recovery_abstention_and_selection(
    tmp_path: Path,
) -> None:
    benchmark = load_verifier_benchmark(DEFINITION, SOURCE_SUITE)
    report = run_verifier_benchmark(
        benchmark,
        PerfectGenerator(benchmark),
        model="fake/perfect",
        model_revision="test",
    )
    assert report.initial_metrics["accuracy"] == 1.0
    assert report.initial_metrics["false_positive_rate"] == 0.0
    assert report.initial_metrics["false_negative_rate"] == 0.0
    assert report.flow_accuracy == 1.0
    assert report.rewrite_metrics == {
        "eligible_cases": 10,
        "proposal_rate": 1.0,
        "execution_rate": 1.0,
        "recovery_rate": 1.0,
    }
    assert report.abstention_metrics == {"eligible_cases": 2, "accuracy": 1.0}
    assert report.supported_selection["precision"] == 1.0
    assert report.supported_selection["recall"] == 1.0
    assert report.model_calls == 34
    assert report.generation_batches == 2
    assert report.parse_failure_count == 0

    write_verifier_outputs(report, tmp_path)
    assert (tmp_path / "verifier_report.json").is_file()
    assert (tmp_path / "verifier_report.md").is_file()
    assert "Development diagnostic only" in render_verifier_markdown(report)


def test_invalid_responses_fail_closed_and_are_counted() -> None:
    benchmark = load_verifier_benchmark(DEFINITION, SOURCE_SUITE)

    def invalid_generator(prompts: list[str]) -> list[str]:
        return ["not json" for _prompt in prompts]

    report = run_verifier_benchmark(benchmark, invalid_generator, model="fake/invalid")
    assert report.parse_failure_count == 44
    assert report.initial_metrics["false_positive_rate"] == 0.0
    assert report.initial_metrics["false_negative_rate"] == 1.0
    assert report.abstention_metrics["accuracy"] == 1.0
    assert report.rewrite_metrics["recovery_rate"] == 0.0


def test_false_positive_does_not_count_as_rewrite_recovery() -> None:
    benchmark = load_verifier_benchmark(DEFINITION, SOURCE_SUITE)

    def approve_everything(prompts: list[str]) -> list[str]:
        return [_response(True, [1], str(index)) for index, _prompt in enumerate(prompts)]

    report = run_verifier_benchmark(
        benchmark, approve_everything, model="fake/unsafe-approve"
    )
    assert report.initial_metrics["false_positive_rate"] == 1.0
    assert report.rewrite_metrics["execution_rate"] == 0.0
    assert report.rewrite_metrics["recovery_rate"] == 0.0
    assert report.abstention_metrics["accuracy"] == 0.0
    assert report.flow_accuracy == pytest.approx(10 / 22)


def test_loader_rejects_wrong_source_suite(tmp_path: Path) -> None:
    payload = json.loads(DEFINITION.read_text(encoding="utf-8"))
    payload["source_suite_id"] = "wrong"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_verifier_benchmark(path, SOURCE_SUITE)
