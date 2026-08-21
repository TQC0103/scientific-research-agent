import json
from pathlib import Path

from pydantic import ValidationError

from app.evaluation.models import EvaluationSuite


class DatasetValidationError(ValueError):
    """Raised when an evaluation source artifact is malformed or inconsistent."""


def load_suite(path: str | Path) -> EvaluationSuite:
    source = Path(path)
    if source.suffix.casefold() != ".json":
        raise DatasetValidationError(f"Evaluation suites must be JSON files: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetValidationError(f"Could not read evaluation suite {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"Malformed JSON in {source} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    try:
        return EvaluationSuite.model_validate(payload)
    except ValidationError as exc:
        raise DatasetValidationError(f"Invalid evaluation suite {source}:\n{exc}") from exc
