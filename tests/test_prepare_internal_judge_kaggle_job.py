import base64
import io
import json
import zipfile
from pathlib import Path

from scripts import prepare_internal_judge_kaggle_job as package


def test_prepared_judge_job_is_narrow_and_embeds_development_suite(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    template = root / "evaluation" / "kaggle" / "internal_judge_v0_5"
    suite = root / "evaluation" / "suites" / "v0_5"
    destination = root / "data" / "evaluations" / "kaggle_jobs" / "internal_judge_v0_5"
    for directory in (template, suite, root / "app" / "evaluation", root / "scripts"):
        directory.mkdir(parents=True, exist_ok=True)
    (template / "main.py").write_text(
        'APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"\n'
        'REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"\n',
        encoding="utf-8",
    )
    (template / "kernel-metadata.json").write_text("{}\n", encoding="utf-8")
    (template / "requirements-kaggle.txt").write_text("torch==2.10.0\n", encoding="utf-8")
    (root / "app" / "evaluation" / "judge.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "scripts" / "run_evaluation_judge.py").write_text(
        "print('judge')\n", encoding="utf-8"
    )
    (suite / "development_10.json").write_text('{"cases": []}\n', encoding="utf-8")

    monkeypatch.setattr(package, "ROOT", root)
    monkeypatch.setattr(package, "TEMPLATE", template)
    monkeypatch.setattr(package, "DESTINATION", destination)
    package.main()

    top_level_sources = [
        path for path in destination.iterdir() if path.suffix in {".py", ".ipynb"}
    ]
    assert [path.name for path in top_level_sources] == ["main.py"]
    generated = (destination / "main.py").read_text(encoding="utf-8")
    encoded = generated.split('APP_ARCHIVE_B64 = "', 1)[1].split('"', 1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        assert "app/run_evaluation_judge.py" in archive.namelist()
        assert "app/evaluation/judge.py" in archive.namelist()
        assert "evaluation/suites/v0_5/development_10.json" in archive.namelist()
    manifest = json.loads((destination / "source_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"kernel-metadata.json", "main.py"}


def test_judge_entrypoint_requires_isolated_t4_runtime_and_deterministic_batching() -> None:
    entrypoint = (
        package.ROOT / "evaluation" / "kaggle" / "internal_judge_v0_5" / "main.py"
    ).read_text(encoding="utf-8")
    runner = (package.ROOT / "scripts" / "run_evaluation_judge.py").read_text(
        encoding="utf-8"
    )

    assert 'ENV_ROOT = Path("/tmp/internal_judge_env")' in entrypoint
    assert 'CODE_ROOT = Path("/tmp/internal_judge_source")' in entrypoint
    assert 'OUTPUT = Path("/kaggle/working/internal_judge_v0_5")' in entrypoint
    assert 'VIRTUALENV_VERSION = "20.36.1"' in entrypoint
    assert 'capability != (7, 5)' in entrypoint
    assert '"--batch-size",\n            "2"' in entrypoint
    assert 'tokenizer.padding_side = "left"' in runner
    assert "do_sample=False" in runner
    assert "model.generation_config.temperature = None" in runner
    assert "model.generation_config.top_p = None" in runner
    assert "model.generation_config.top_k = None" in runner
    assert "torch_dtype=" not in runner
