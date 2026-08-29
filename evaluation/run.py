"""Run the internal suite through the compiled production LangGraph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agent.graph import research_graph
from app.evaluation.end_to_end import (
    NODE_TRACE_KEY,
    load_end_to_end_aggregate,
    run_end_to_end,
    write_end_to_end_outputs,
)
from app.evaluation.loader import load_suite


def _invoke_with_node_trace(payload: dict, config: dict) -> dict:
    state = dict(payload)
    events = []
    for update in research_graph.stream(payload, config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node, values in update.items():
            node_update = dict(values) if isinstance(values, dict) else {}
            events.append({"node": str(node), "update": node_update})
            state.update(node_update)
    state[NODE_TRACE_KEY] = events
    return state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a versioned end-to-end evaluation over production LangGraph."
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--config", dest="config_name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retrieval-k-per-paper", type=int, default=5)
    parser.add_argument("--quote-token-recall-threshold", type=float, default=0.8)
    parser.add_argument("--recursion-limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    suite = load_suite(args.suite)
    if args.limit:
        if args.limit < 1:
            raise ValueError("--limit must be positive when supplied.")
        suite = suite.model_copy(update={"cases": suite.cases[: args.limit]})
    baseline = load_end_to_end_aggregate(args.baseline) if args.baseline else None
    report = run_end_to_end(
        suite,
        _invoke_with_node_trace,
        config_name=args.config_name,
        retrieval_k_per_paper=args.retrieval_k_per_paper,
        quote_token_recall_threshold=args.quote_token_recall_threshold,
        recursion_limit=args.recursion_limit,
        baseline=baseline,
    )
    write_end_to_end_outputs(report, args.output_dir)
    print(
        json.dumps(
            {
                "full_report": str(args.output_dir / "report.json"),
                "metrics": str(args.output_dir / "metrics.json"),
                "per_case": str(args.output_dir / "per_case.jsonl"),
                "report": str(args.output_dir / "report.md"),
                **report.aggregate.model_dump(
                    mode="json", exclude={"runtime", "baseline_comparison"}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
