"""Prepare the narrow, ignored SciFact Kaggle source bundle."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "evaluation" / "kaggle" / "scifact_v0_5"
DESTINATION = ROOT / "data" / "evaluations" / "kaggle_jobs" / "scifact_v0_5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files() -> dict[Path, str]:
    return {
        ROOT / "app" / "__init__.py": "app/__init__.py",
        ROOT / "app" / "evaluation" / "__init__.py": "app/evaluation/__init__.py",
        ROOT / "app" / "evaluation" / "external.py": "app/evaluation/external.py",
        ROOT / "app" / "evaluation" / "loader.py": "app/evaluation/loader.py",
        ROOT / "app" / "evaluation" / "metrics.py": "app/evaluation/metrics.py",
        ROOT / "app" / "evaluation" / "models.py": "app/evaluation/models.py",
        ROOT / "app" / "evaluation" / "scifact.py": "app/evaluation/scifact.py",
        ROOT / "scripts" / "run_scifact.py": "scripts/run_scifact.py",
    }


def _embedded_source() -> str:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, target in sorted(_source_files().items(), key=lambda item: item[1]):
            archive.write(source, target)
    return base64.b64encode(payload.getvalue()).decode("ascii")


def main() -> None:
    destination = DESTINATION.resolve()
    allowed = (ROOT / "data" / "evaluations" / "kaggle_jobs").resolve()
    if destination == allowed or not destination.is_relative_to(allowed):
        raise ValueError(f"Unsafe Kaggle destination: {destination}")
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
                .replace("__KAGGLE_REQUIREMENTS_B64__", requirements), encoding="utf-8"
            )
        else:
            shutil.copy2(source, target)
    files = sorted(path for path in destination.iterdir() if path.is_file())
    manifest = {
        "source": "scientific-research-agent v0.5 SciFact oracle-document runner",
        "dataset_url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        "dataset_sha256": "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
        "model": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "files": {path.name: _sha256(path) for path in files},
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
