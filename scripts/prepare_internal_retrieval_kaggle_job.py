"""Prepare a narrow, ignored internal-retrieval Kaggle source bundle."""

from __future__ import annotations

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
PAPERS = ROOT / "data" / "papers"

def _app_files() -> dict[Path, str]:
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
        ROOT / "evaluation" / "suites" / "v0_5" / "development_10.json": (
            "evaluation/suites/v0_5/development_10.json"
        ),
        ROOT / "evaluation" / "suites" / "v0_5" / "development_10_sources.json": (
            "evaluation/suites/v0_5/development_10_sources.json"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_pdfs() -> dict[Path, str]:
    manifest_path = (
        ROOT / "evaluation" / "suites" / "v0_5" / "development_10_sources.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {}
    for source in manifest["sources"]:
        versioned_id = source["versioned_id"]
        pdf = PAPERS / f"{versioned_id}.pdf"
        if not pdf.is_file():
            raise FileNotFoundError(f"Pinned source PDF is missing: {pdf}")
        actual = _sha256(pdf)
        if actual != source["pdf_sha256"]:
            raise ValueError(f"Pinned PDF checksum mismatch: {versioned_id}")
        result[pdf] = f"papers/{pdf.name}"
    return result


def _embedded_app() -> str:
    payload = io.BytesIO()
    files = {**_app_files(), **_validated_pdfs()}
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, target in sorted(files.items(), key=lambda item: item[1]):
            archive.write(source, target)
    return base64.b64encode(payload.getvalue()).decode("ascii")


def _embedded_requirements() -> str:
    return base64.b64encode((TEMPLATE / "requirements-kaggle.txt").read_bytes()).decode(
        "ascii"
    )


def main() -> None:
    destination = DESTINATION.resolve()
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
                template.replace("__APP_ARCHIVE_B64__", _embedded_app()).replace(
                    "__KAGGLE_REQUIREMENTS_B64__", _embedded_requirements()
                ),
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, target)
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "source": "scientific-research-agent internal retrieval v0.5",
        "public_sources": json.loads(
            (
                ROOT
                / "evaluation"
                / "suites"
                / "v0_5"
                / "development_10_sources.json"
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
    main()
