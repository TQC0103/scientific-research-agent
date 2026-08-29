"""Export the Task 11 end-to-end report contract as JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.end_to_end import end_to_end_json_schema


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(end_to_end_json_schema(), indent=2), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
