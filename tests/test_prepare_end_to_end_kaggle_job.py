import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import prepare_end_to_end_kaggle_job as package


def test_prepared_end_to_end_bundle_is_narrow(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    template = root / "evaluation/kaggle/end_to_end_v0_5"
    destination = root / "data/evaluations/kaggle_jobs/end_to_end_v0_5"
    template.mkdir(parents=True)
    (template / "main.py").write_text(
        'APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"\n'
        'REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"\n', encoding="utf-8"
    )
    (template / "kernel-metadata.json").write_text("{}\n", encoding="utf-8")
    (template / "requirements-kaggle.txt").write_text("wrapt==1.17.3\n", encoding="utf-8")
    for target in package._source_files().values():
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        if target.endswith("development_10_sources.json"):
            path.write_text('{"sources": []}\n', encoding="utf-8")
        else:
            path.write_text(f"# {target}\n", encoding="utf-8")
    monkeypatch.setattr(package, "ROOT", root)
    monkeypatch.setattr(package, "TEMPLATE", template)
    monkeypatch.setattr(package, "DESTINATION", destination)
    package.main(
        [
            "--kernel-slug",
            "owner/scientific-research-agent-r8",
            "--title",
            "Scientific Research Agent R8",
        ]
    )
    generated = (destination / "main.py").read_text(encoding="utf-8")
    encoded = generated.split('APP_ARCHIVE_B64 = "', 1)[1].split('"', 1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        names = set(archive.namelist())
        timestamps = {item.date_time for item in archive.infolist()}
    assert "app/agent/graph.py" in names
    assert "app/evaluation/end_to_end.py" in names
    assert "scripts/run_end_to_end_transformers.py" in names
    assert "evaluation/run.py" not in names
    assert timestamps == {(1980, 1, 1, 0, 0, 0)}
    manifest = json.loads((destination / "source_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((destination / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "id": "owner/scientific-research-agent-r8",
        "title": "Scientific Research Agent R8",
    }
    assert manifest["llm_revision"].startswith("1cfa9a")
    assert set(manifest["files"]) == {"kernel-metadata.json", "main.py"}
    assert manifest["files"]["kernel-metadata.json"] == package._sha256(
        destination / "kernel-metadata.json"
    )


def test_entrypoint_smokes_before_full_and_preserves_system_torch() -> None:
    entrypoint = (package.TEMPLATE / "main.py").read_text(encoding="utf-8")
    runner = (package.ROOT / "scripts/run_end_to_end_transformers.py").read_text(
        encoding="utf-8"
    )
    requirements = (package.TEMPLATE / "requirements-kaggle.txt").read_text(encoding="utf-8")
    assert '"--system-site-packages"' in entrypoint
    assert '"--no-deps"' in entrypoint
    assert '"--smoke-cases", "1"' in entrypoint
    assert "smoke.aggregate.execution_failures" in runner
    assert 'self.tokenizer.padding_side = "left"' in runner
    assert "enable_thinking=False" in runner
    assert "do_sample=False" in runner
    assert 'device="cuda:1" if llm.torch.cuda.device_count() > 1 else "cpu"' in runner
    assert '"embedding_device": "cuda:1" if len(devices) > 1 else "cpu"' in entrypoint
    assert 'attn_implementation="sdpa"' in runner
    assert "self.torch.cuda.empty_cache()" in runner
    assert "torch==" not in requirements


@pytest.mark.parametrize(
    ("kernel_slug", "title", "message"),
    [
        (f"owner/{'s' * 51}", "Valid title", "slug exceeds"),
        ("owner/valid-slug", "T" * 51, "title exceeds"),
    ],
)
def test_packager_rejects_kaggle_identity_over_service_limit(
    kernel_slug: str, title: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        package._validate_kernel_identity(kernel_slug, title)


def test_packager_rejects_kernel_slug_without_owner() -> None:
    with pytest.raises(ValueError, match="owner/slug"):
        package._validate_kernel_identity("missing-owner", "Valid title")


def test_r25_source_selection_is_explicit_and_excludes_r10() -> None:
    names = set(package._source_files("development_25").values())

    assert "evaluation/suites/v0_5/development_25.json" in names
    assert "evaluation/suites/v0_5/development_25_sources.json" in names
    assert "evaluation/suites/v0_5/development_10.json" not in names


def test_real_embedded_archive_imports_metrics_dependency_in_isolation(tmp_path: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(package._embedded_source()))) as archive:
        archive.extractall(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(tmp_path)!r}); "
                "from app.evaluation.metrics import token_f1; "
                "print(token_f1)"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
