import base64
import io
import json
import zipfile
from pathlib import Path

from scripts import prepare_qasper_kaggle_job as package


def test_prepared_job_has_one_top_level_entrypoint(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    template = root / "evaluation" / "kaggle" / "qasper_v0_5"
    destination = root / "data" / "evaluations" / "kaggle_jobs" / "qasper_v0_5"
    app = root / "app"
    scripts = root / "scripts"
    for directory in (template, app / "__pycache__", scripts):
        directory.mkdir(parents=True, exist_ok=True)
    (template / "main.py").write_text(
        'APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"\n'
        'REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"\n',
        encoding="utf-8",
    )
    (template / "kernel-metadata.json").write_text("{}\n", encoding="utf-8")
    (template / "requirements-kaggle.txt").write_text(
        "torch==2.10.0\n", encoding="utf-8"
    )
    (app / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (app / "__pycache__" / "module.pyc").write_bytes(b"cache")
    (scripts / "run_qasper.py").write_text("print('runner')\n", encoding="utf-8")

    monkeypatch.setattr(package, "ROOT", root)
    monkeypatch.setattr(package, "TEMPLATE", template)
    monkeypatch.setattr(package, "DESTINATION", destination)

    package.main()

    top_level_sources = [
        path for path in destination.iterdir() if path.suffix in {".py", ".ipynb"}
    ]
    assert [path.name for path in top_level_sources] == ["main.py"]
    generated = (destination / "main.py").read_text(encoding="utf-8")
    assert "__APP_ARCHIVE_B64__" not in generated
    assert "__KAGGLE_REQUIREMENTS_B64__" not in generated
    encoded = generated.split('APP_ARCHIVE_B64 = "', 1)[1].split('"', 1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        assert "app/run_qasper.py" in archive.namelist()
    requirements = generated.split('REQUIREMENTS_B64 = "', 1)[1].split('"', 1)[0]
    assert base64.b64decode(requirements).decode("utf-8").splitlines() == [
        "torch==2.10.0"
    ]
    manifest = json.loads((destination / "source_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"kernel-metadata.json", "main.py"}


def test_kaggle_entrypoint_uses_isolated_t4_runtime() -> None:
    entrypoint = (
        package.ROOT / "evaluation" / "kaggle" / "qasper_v0_5" / "main.py"
    ).read_text(encoding="utf-8")

    assert 'ENV_ROOT = Path("/kaggle/working/qasper_env")' in entrypoint
    assert 'VIRTUALENV_VERSION = "20.36.1"' in entrypoint
    assert '"virtualenv"' in entrypoint
    assert '"--no-download"' in entrypoint
    assert "--system-site-packages" not in entrypoint
    assert 'capability != (7, 5)' in entrypoint
    assert "cuda_preflight_sum" in entrypoint
    assert "resolved-requirements.txt" in entrypoint
    assert "_ensure_isolated_runtime()" in entrypoint
