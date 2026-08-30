"""Run gold-hidden internal retrieval rankings, then score and report them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.internal_retrieval_runner import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_DENSE_REVISION,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    SUPPORTED_MODES,
    run_internal_retrieval_ablation,
    write_ablation_outputs,
)
from app.evaluation.loader import load_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=SUPPORTED_MODES, default=SUPPORTED_MODES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--dense-revision", default=DEFAULT_DENSE_REVISION)
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-revision", default=DEFAULT_RERANKER_REVISION)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=250)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    result = run_internal_retrieval_ablation(
        suite,
        sources_path=args.sources,
        papers_dir=args.papers_dir,
        modes=tuple(args.modes),
        top_k=args.top_k,
        dense_model=args.dense_model,
        dense_revision=args.dense_revision,
        dense_batch_size=args.dense_batch_size,
        reranker_model=args.reranker_model,
        reranker_revision=args.reranker_revision,
        reranker_batch_size=args.reranker_batch_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    write_ablation_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "suite_id": suite.suite_id,
                "dataset_version": suite.dataset_version,
                "modes": {
                    mode: {
                        **report.aggregate.model_dump(mode="json"),
                        "latency_seconds": result.latency_seconds[mode],
                    }
                    for mode, report in result.reports.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
