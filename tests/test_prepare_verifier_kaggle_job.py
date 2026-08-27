import base64
import io
import json
import zipfile
from pathlib import Path

from scripts import prepare_verifier_kaggle_job as package


def test_prepared_verifier_job_is_narrow_and_embeds_both_suites(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    template = root / "evaluation" / "kaggle" / "verifier_v0_5"
    suite = root / "evaluation" / "suites" / "v0_5"
    destination = root / "data" / "evaluations" / "kaggle_jobs" / "verifier_v0_5"
    required_targets = list(package._source_files().values())
    monkeypatch.setattr(package, "ROOT", root)
    monkeypatch.setattr(package, "TEMPLATE", template)
    monkeypatch.setattr(package, "DESTINATION", destination)

    template.mkdir(parents=True)
    suite.mkdir(parents=True)
    (template / "main.py").write_text(
        'APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"\n'
        'REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"\n',
        encoding="utf-8",
    )
    (template / "kernel-metadata.json").write_text("{}\n", encoding="utf-8")
    (template / "requirements-kaggle.txt").write_text(
        "transformers==4.56.2\n", encoding="utf-8"
    )
    for target in required_targets:
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {target}\n", encoding="utf-8")

    package.main()

    assert sorted(path.name for path in destination.iterdir()) == [
        "kernel-metadata.json",
        "main.py",
        "source_manifest.json",
    ]
    generated = (destination / "main.py").read_text(encoding="utf-8")
    encoded = generated.split('APP_ARCHIVE_B64 = "', 1)[1].split('"', 1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        names = set(archive.namelist())
    assert "app/evaluation/verifier.py" in names
    assert "app/models/verifier.py" in names
    assert "scripts/run_verifier_benchmark.py" in names
    assert "evaluation/suites/v0_5/development_10.json" in names
    assert "evaluation/suites/v0_5/verifier_development.json" in names
    manifest = json.loads((destination / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"


def test_verifier_entrypoint_preserves_t4_runtime_and_safe_generation() -> None:
    entrypoint = (
        package.ROOT / "evaluation" / "kaggle" / "verifier_v0_5" / "main.py"
    ).read_text(encoding="utf-8")
    runner = (package.ROOT / "scripts" / "run_verifier_benchmark.py").read_text(
        encoding="utf-8"
    )
    requirements = (
        package.ROOT
        / "evaluation"
        / "kaggle"
        / "verifier_v0_5"
        / "requirements-kaggle.txt"
    ).read_text(encoding="utf-8")

    assert 'ENV_ROOT = Path("/kaggle/working/verifier_v0_5_venv")' in entrypoint
    assert 'CODE_ROOT = Path("/tmp/verifier_v0_5_source")' in entrypoint
    assert 'OUTPUT = Path("/kaggle/working/verifier_v0_5")' in entrypoint
    assert 'VIRTUALENV_VERSION = "20.36.1"' in entrypoint
    assert '"--system-site-packages"' in entrypoint
    assert 'item["capability"] != [7, 5]' in entrypoint
    assert '"inference_device_ids": [0]' in entrypoint
    assert '"--smoke-cases"' in entrypoint
    assert "torch==" not in requirements
    assert 'self.tokenizer.padding_side = "left"' in runner
    assert "enable_thinking=False" in runner
    assert "do_sample=False" in runner
    assert "self.model.generation_config.temperature = None" in runner
    assert "self.model.generation_config.top_p = None" in runner
    assert "self.model.generation_config.top_k = None" in runner
    assert "torch_dtype=" not in runner
