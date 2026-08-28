"""Prepare the narrow, ignored Kaggle claim-verifier benchmark source."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "evaluation" / "kaggle" / "claim_verifier_v0_5"
DESTINATION = ROOT / "data" / "evaluations" / "kaggle_jobs" / "claim_verifier_v0_5"


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
        ROOT / "app" / "evaluation" / "claim_verifier.py": "app/evaluation/claim_verifier.py",
        ROOT / "app" / "evaluation" / "citations.py": "app/evaluation/citations.py",
        ROOT / "app" / "evaluation" / "loader.py": "app/evaluation/loader.py",
        ROOT / "app" / "evaluation" / "models.py": "app/evaluation/models.py",
        ROOT / "app" / "models" / "__init__.py": "app/models/__init__.py",
        ROOT / "app" / "models" / "claims.py": "app/models/claims.py",
        ROOT / "app" / "models" / "claim_verifier.py": "app/models/claim_verifier.py",
        ROOT / "scripts" / "run_claim_verifier_benchmark.py": "scripts/run_claim_verifier_benchmark.py",
        ROOT / "evaluation" / "suites" / "v0_5" / "claim_verifier_development.json": (
            "evaluation/suites/v0_5/claim_verifier_development.json"
        ),
    }


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
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "source": "scientific-research-agent v0.5 claim-verifier benchmark package",
        "model": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "files": {str(path.relative_to(destination)).replace("\\", "/"): _sha256(path)
                  for path in files},
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
