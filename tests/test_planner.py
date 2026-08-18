import pytest

from app.models import planner


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, prompt: str) -> FakeResponse:
        return FakeResponse(self.content)


def test_planner_returns_validated_multi_paper_plan(monkeypatch) -> None:
    content = """Some preface
    {
      "mode": "multi_paper",
      "search_query": "transformer bert architecture training objective",
      "required_paper_count": 2,
      "comparison_dimensions": ["architecture", "training objective", "architecture"],
      "rationale": "The request compares two methods."
    }
    """
    monkeypatch.setattr(planner, "get_llm", lambda **kwargs: FakeModel(content))

    result = planner.plan_query("Compare Transformer and BERT")

    assert result.mode == "multi_paper"
    assert result.required_paper_count == 2
    assert result.comparison_dimensions == ["architecture", "training objective"]
    assert result.used_fallback is False


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        (
            '{"mode":"single_paper","search_query":"x","required_paper_count":2,'
            '"comparison_dimensions":[],"rationale":"contradiction"}'
        ),
        (
            '{"mode":"multi_paper","search_query":"x","required_paper_count":3,'
            '"comparison_dimensions":[],"rationale":"too many automatic papers"}'
        ),
        (
            '{"mode":"single_paper","search_query":"x","required_paper_count":1,'
            '"comparison_dimensions":[],"rationale":"ok","unexpected":true}'
        ),
    ],
)
def test_invalid_planner_output_uses_safe_single_paper_fallback(
    monkeypatch, content: str
) -> None:
    monkeypatch.setattr(planner, "get_llm", lambda **kwargs: FakeModel(content))

    result = planner.plan_query("Explain attention")

    assert result.mode == "single_paper"
    assert result.search_query == "Explain attention"
    assert result.required_paper_count == 1
    assert result.used_fallback is True


def test_explicit_paper_count_is_a_hard_constraint(monkeypatch) -> None:
    content = (
        '{"mode":"single_paper","search_query":"attention","required_paper_count":1,'
        '"comparison_dimensions":[],"rationale":"incorrect count"}'
    )
    monkeypatch.setattr(planner, "get_llm", lambda **kwargs: FakeModel(content))

    result = planner.plan_query("Compare these papers", forced_paper_count=3)

    assert result.mode == "multi_paper"
    assert result.required_paper_count == 3
    assert result.search_query == "Compare these papers"
    assert result.used_fallback is True


def test_empty_question_is_rejected_before_model_call() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        planner.plan_query("  ")
