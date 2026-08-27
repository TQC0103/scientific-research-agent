"""Export the Task 7 claim-verification contract as JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.models.claims import claim_verification_json_schema


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(claim_verification_json_schema(), indent=2), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
