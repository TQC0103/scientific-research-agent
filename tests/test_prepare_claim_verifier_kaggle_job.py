import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts import prepare_claim_verifier_kaggle_job as package


def test_prepared_job_contains_only_required_runtime_sources(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    template = root / "evaluation" / "kaggle" / "claim_verifier_v0_5"
    destination = root / "data" / "evaluations" / "kaggle_jobs" / "claim_verifier_v0_5"
    targets = list(package._source_files().values())
    monkeypatch.setattr(package, "ROOT", root)
    monkeypatch.setattr(package, "TEMPLATE", template)
    monkeypatch.setattr(package, "DESTINATION", destination)
    template.mkdir(parents=True)
    (template / "main.py").write_text(
        'APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"\n'
        'REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"\n', encoding="utf-8"
    )
    (template / "kernel-metadata.json").write_text("{}\n", encoding="utf-8")
    (template / "requirements-kaggle.txt").write_text("wrapt==1.17.3\n", encoding="utf-8")
    for target in targets:
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {target}\n", encoding="utf-8")
    package.main()
    generated = (destination / "main.py").read_text(encoding="utf-8")
    encoded = generated.split('APP_ARCHIVE_B64 = "', 1)[1].split('"', 1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        names = set(archive.namelist())
    assert "app/evaluation/claim_verifier.py" in names
    assert "app/evaluation/citations.py" in names
    assert "app/evaluation/loader.py" in names
    assert "app/models/claim_verifier.py" in names
    assert "evaluation/suites/v0_5/claim_verifier_development.json" in names
    assert "app/evaluation/external.py" not in names
    manifest = json.loads((destination / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"


def test_entrypoint_preserves_safe_t4_runtime_and_generation() -> None:
    entrypoint = (package.TEMPLATE / "main.py").read_text(encoding="utf-8")
    runner = (package.ROOT / "scripts" / "run_claim_verifier_benchmark.py").read_text(
        encoding="utf-8"
    )
    requirements = (package.TEMPLATE / "requirements-kaggle.txt").read_text(encoding="utf-8")
    assert 'ENV_ROOT = Path("/kaggle/working/claim_verifier_v0_5_venv")' in entrypoint
    assert 'CODE_ROOT = Path("/tmp/claim_verifier_v0_5_source")' in entrypoint
    assert 'item["capability"] != [7, 5]' in entrypoint
    assert '"inference_device_ids": [0]' in entrypoint
    assert '"--smoke-cases", "1"' in entrypoint
    assert "torch==" not in requirements
    assert 'self.tokenizer.padding_side = "left"' in runner
    assert "enable_thinking=False" in runner
    assert "do_sample=False" in runner
    assert "generation_config.temperature = None" in runner


def test_real_embedded_archive_imports_in_isolation(tmp_path: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(package._embedded_source()))) as archive:
        archive.extractall(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(tmp_path)!r}); "
                "from app.evaluation.claim_verifier import load_claim_verifier_suite; "
                "print(load_claim_verifier_suite)"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
