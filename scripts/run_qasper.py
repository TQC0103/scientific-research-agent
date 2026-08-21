"""Run the portable QASPER benchmark and write ignored runtime artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.qasper_runner import (
    AbstainingGenerator,
    TransformersGenerator,
    run_qasper,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--retrieval-mode", choices=["lexical", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--generator", choices=["none", "transformers"], default="none")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Test-set access requires --allow-test after the configuration is finalized.")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.generation_batch_size < 1:
        parser.error("--generation-batch-size must be positive")
    generator = (
        AbstainingGenerator()
        if args.generator == "none"
        else TransformersGenerator(
            args.generator_model, batch_size=args.generation_batch_size
        )
    )
    rows, aggregate = run_qasper(
        args.dataset,
        source_split=args.split,
        retrieval_mode=args.retrieval_mode,
        generator=generator,
        top_k=args.top_k,
        dense_model=args.dense_model,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    metrics = args.output_dir / "metrics.json"
    metrics.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"predictions": str(predictions), "metrics": str(metrics), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
