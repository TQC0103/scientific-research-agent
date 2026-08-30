"""Prepare a narrow, ignored internal-retrieval Kaggle source bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "evaluation" / "kaggle" / "internal_retrieval_v0_5"
DESTINATION = ROOT / "data" / "evaluations" / "kaggle_jobs" / "internal_retrieval_v0_5"
DEFAULT_SUITE_NAME = "development_10"


def _app_files(suite_name: str = DEFAULT_SUITE_NAME) -> dict[Path, str]:
    return {
        ROOT / "app" / "__init__.py": "app/__init__.py",
        ROOT / "app" / "evaluation" / "__init__.py": "app/evaluation/__init__.py",
        ROOT / "app" / "evaluation" / "models.py": "app/evaluation/models.py",
        ROOT / "app" / "evaluation" / "loader.py": "app/evaluation/loader.py",
        ROOT / "app" / "evaluation" / "retrieval.py": "app/evaluation/retrieval.py",
        ROOT / "app" / "evaluation" / "metrics.py": "app/evaluation/metrics.py",
        ROOT / "app" / "evaluation" / "external.py": "app/evaluation/external.py",
        ROOT / "app" / "evaluation" / "qasper_runner.py": "app/evaluation/qasper_runner.py",
        ROOT / "app" / "evaluation" / "internal_retrieval_runner.py": (
            "app/evaluation/internal_retrieval_runner.py"
        ),
        ROOT / "app" / "ingestion" / "__init__.py": "app/ingestion/__init__.py",
        ROOT / "app" / "ingestion" / "chunking.py": "app/ingestion/chunking.py",
        ROOT / "app" / "ingestion" / "pdf_parser.py": "app/ingestion/pdf_parser.py",
        ROOT / "scripts" / "run_internal_retrieval_ablation.py": (
            "scripts/run_internal_retrieval_ablation.py"
        ),
        ROOT / "evaluation" / "suites" / "v0_5" / f"{suite_name}.json": (
            f"evaluation/suites/v0_5/{suite_name}.json"
        ),
        ROOT / "evaluation" / "suites" / "v0_5" / f"{suite_name}_sources.json": (
            f"evaluation/suites/v0_5/{suite_name}_sources.json"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedded_app(suite_name: str = DEFAULT_SUITE_NAME) -> str:
    payload = io.BytesIO()
    files = _app_files(suite_name)
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, target in sorted(files.items(), key=lambda item: item[1]):
            archive.write(source, target)
    return base64.b64encode(payload.getvalue()).decode("ascii")


def _embedded_requirements() -> str:
    return base64.b64encode((TEMPLATE / "requirements-kaggle.txt").read_bytes()).decode(
        "ascii"
    )


def main(
    *, suite_name: str = DEFAULT_SUITE_NAME, destination_path: Path | None = None
) -> None:
    if not suite_name.isidentifier() or not suite_name.startswith("development_"):
        raise ValueError(f"Unsafe suite name: {suite_name}")
    destination = (destination_path or DESTINATION).resolve()
    allowed_root = (ROOT / "data" / "evaluations" / "kaggle_jobs").resolve()
    if not destination.is_relative_to(allowed_root) or destination == allowed_root:
        raise ValueError(f"Unsafe Kaggle job destination: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source in TEMPLATE.iterdir():
        if not source.is_file() or source.name == "requirements-kaggle.txt":
            continue
        target = destination / source.name
        if source.name == "main.py":
            template = source.read_text(encoding="utf-8")
            target.write_text(
                template.replace("__APP_ARCHIVE_B64__", _embedded_app(suite_name))
                .replace("__KAGGLE_REQUIREMENTS_B64__", _embedded_requirements())
                .replace("__SUITE_FILENAME__", f"{suite_name}.json")
                .replace("__SOURCES_FILENAME__", f"{suite_name}_sources.json")
                .replace("__OUTPUT_DIRNAME__", destination.name),
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, target)
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "source": "scientific-research-agent internal retrieval v0.5",
        "suite_name": suite_name,
        "public_sources": json.loads(
            (
                ROOT
                / "evaluation"
                / "suites"
                / "v0_5"
                / f"{suite_name}_sources.json"
            ).read_text(encoding="utf-8")
        )["sources"],
        "files": {
            str(path.relative_to(destination)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", default=DEFAULT_SUITE_NAME)
    parser.add_argument("--destination", type=Path)
    arguments = parser.parse_args()
    main(suite_name=arguments.suite_name, destination_path=arguments.destination)
