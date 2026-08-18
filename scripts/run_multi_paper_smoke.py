"""Run the fixed positive and coverage-gap multi-paper smoke cases."""

import json
from datetime import UTC, datetime

from app.agent.graph import research_graph
from app.config import settings

PAPER_IDS = ["1706.03762", "1810.04805"]
CASES = [
    {
        "name": "positive_self_attention_comparison",
        "question": (
            "Compare how Attention Is All You Need and BERT use self-attention "
            "in their model architectures."
        ),
    },
    {
        "name": "training_objective_coverage_gap",
        "question": (
            "Compare how Attention Is All You Need and BERT use the Transformer "
            "architecture and training objectives."
        ),
    },
]


def main() -> None:
    results = []
    for case in CASES:
        state = research_graph.invoke(
            {"user_query": case["question"], "paper_ids": PAPER_IDS},
            {"recursion_limit": 40},
        )
        result = {
            **case,
            "answer": state["answer"],
            "evidence_sufficient": state.get("evidence_sufficient"),
            "coverage": state.get("evidence_verification"),
            "retrieval_queries": state.get("retrieval_queries"),
            "retrieval_attempt_counts": state.get("retrieval_attempt_counts"),
            "tool_errors": state.get("tool_errors", []),
        }
        results.append(result)
        print(
            f"{case['name']}: verified={result['evidence_sufficient']} "
            f"attempts={result['retrieval_attempt_counts']}",
            flush=True,
        )

    destination = settings.data_dir / "evaluations" / "multi_paper_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "run_at": datetime.now(UTC).isoformat(),
                "paper_ids": PAPER_IDS,
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
