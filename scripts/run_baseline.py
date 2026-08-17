"""Run the fixed six-question verifier baseline and save the workflow outputs."""

import json
from datetime import UTC, datetime

from app.agent.graph import research_graph
from app.config import settings

QUESTIONS = [
    "Why does the Transformer use multi-head attention instead of a single attention function?",
    "How does the Transformer represent token order without recurrence or convolution?",
    "What BLEU scores do the base and big Transformer models achieve on WMT 2014 English-to-German translation?",
    "What are the three ways the Transformer uses multi-head attention?",
    "Does the paper report experimental results on ImageNet?",
    "What limitations of the Transformer are explicitly acknowledged in the paper?",
]


def main() -> None:
    results = []
    for question in QUESTIONS:
        state = research_graph.invoke(
            {"user_query": question, "paper_ids": ["1706.03762"]},
            {"recursion_limit": 30},
        )
        results.append(
            {
                "question": question,
                "answer": state["answer"],
                "evidence_sufficient": state.get("evidence_sufficient"),
                "verification": state.get("evidence_verification"),
                "retrieval_attempt_count": state.get("retrieval_attempt_count"),
                "final_retrieval_query": state.get("retrieval_query"),
                "retrieved_chunks": [
                    {
                        key: chunk.get(key)
                        for key in ("page", "section", "chunk_index", "score", "text")
                    }
                    for chunk in state.get("retrieved_chunks", [])
                ],
                "tool_errors": state.get("tool_errors", []),
            }
        )
        print(
            f"[{len(results)}/{len(QUESTIONS)}] "
            f"verified={state.get('evidence_sufficient')} "
            f"attempts={state.get('retrieval_attempt_count')}",
            flush=True,
        )

    destination = settings.data_dir / "evaluations" / "llm_verifier_baseline.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"run_at": datetime.now(UTC).isoformat(), "paper": "1706.03762v7", "cases": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
