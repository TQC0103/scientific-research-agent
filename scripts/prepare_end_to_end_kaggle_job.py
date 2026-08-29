"""Prepare the narrow, ignored Kaggle end-to-end benchmark source."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "evaluation" / "kaggle" / "end_to_end_v0_5"
DESTINATION = ROOT / "data" / "evaluations" / "kaggle_jobs" / "end_to_end_v0_5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files() -> dict[Path, str]:
    relative = [
        "app/__init__.py", "app/config.py", "app/agent/__init__.py", "app/agent/graph.py",
        "app/agent/state.py", "app/db/__init__.py", "app/db/database.py",
        "app/ingestion/__init__.py", "app/ingestion/chunking.py",
        "app/ingestion/indexing.py", "app/ingestion/pdf_parser.py",
        "app/models/__init__.py", "app/models/claim_verifier.py", "app/models/claims.py",
        "app/models/llm.py", "app/models/verifier.py", "app/retrieval/__init__.py",
        "app/retrieval/vector_store.py", "app/tools/__init__.py", "app/tools/arxiv_search.py",
        "app/tools/paper_download.py", "app/evaluation/__init__.py",
        "app/evaluation/citations.py", "app/evaluation/end_to_end.py",
        "app/evaluation/external.py",
        "app/evaluation/loader.py", "app/evaluation/metrics.py", "app/evaluation/models.py",
        "app/evaluation/retrieval.py", "scripts/run_end_to_end_transformers.py",
        "evaluation/suites/v0_5/development_10.json",
        "evaluation/suites/v0_5/development_10_sources.json",
    ]
    return {ROOT / name: name for name in relative}


def _embedded_source() -> str:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, target in sorted(_source_files().items(), key=lambda item: item[1]):
            archive.write(source, target)
    return base64.b64encode(payload.getvalue()).decode("ascii")


def main() -> None:
    destination = DESTINATION.resolve()
    allowed_root = (ROOT / "data" / "evaluations" / "kaggle_jobs").resolve()
    if not destination.is_relative_to(allowed_root) or destination == allowed_root:
        raise ValueError(f"Unsafe Kaggle job destination: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    requirements = base64.b64encode(
        (TEMPLATE / "requirements-kaggle.txt").read_bytes()
    ).decode("ascii")
    for source in TEMPLATE.iterdir():
        if not source.is_file() or source.name == "requirements-kaggle.txt":
            continue
        target = destination / source.name
        if source.name == "main.py":
            target.write_text(
                source.read_text(encoding="utf-8")
                .replace("__APP_ARCHIVE_B64__", _embedded_source())
                .replace("__KAGGLE_REQUIREMENTS_B64__", requirements),
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, target)
    sources = json.loads(
        (ROOT / "evaluation/suites/v0_5/development_10_sources.json").read_text(
            encoding="utf-8"
        )
    )["sources"]
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "source": "scientific-research-agent v0.5 production end-to-end benchmark",
        "llm_model": "Qwen/Qwen3-4B",
        "llm_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "public_sources": sources,
        "files": {
            str(path.relative_to(destination)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
