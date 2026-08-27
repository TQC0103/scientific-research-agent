"""Evaluate structured claim-to-evidence citation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.citations import (
    evaluate_citation_safety,
    load_citation_safety_suite,
    write_citation_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = evaluate_citation_safety(load_citation_safety_suite(args.suite))
    write_citation_report(report, args.output_dir)
    print(json.dumps(report.metrics, indent=2))


if __name__ == "__main__":
    main()
