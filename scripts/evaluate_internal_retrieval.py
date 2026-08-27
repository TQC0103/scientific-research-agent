import argparse
import json
from pathlib import Path

from app.evaluation.loader import load_suite
from app.evaluation.retrieval import evaluate_retrieval, load_retrieval_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ranked internal-suite retrieval output against gold evidence."
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--retrievals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quote-token-recall-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = load_suite(args.suite)
    retrievals = load_retrieval_jsonl(args.retrievals)
    report = evaluate_retrieval(
        suite,
        retrievals,
        config_name=args.config_name,
        top_k=args.top_k,
        quote_token_recall_threshold=args.quote_token_recall_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "metrics.json"
    per_case_path = args.output_dir / "per_case.jsonl"
    aggregate_path.write_text(
        json.dumps(report.aggregate.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    per_case_path.write_text(
        "".join(
            json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for case in report.cases
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "metrics": str(aggregate_path),
                "per_case": str(per_case_path),
                **report.aggregate.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
