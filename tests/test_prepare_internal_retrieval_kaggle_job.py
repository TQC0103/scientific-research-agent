import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from scripts import prepare_internal_retrieval_kaggle_job as package


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepared_bundle_is_narrow_and_embeds_only_pinned_sources(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    template = root / "evaluation" / "kaggle" / "internal_retrieval_v0_5"
    destination = root / "data" / "evaluations" / "kaggle_jobs" / "internal_retrieval_v0_5"
    papers = root / "data" / "papers"
    template.mkdir(parents=True)
    papers.mkdir(parents=True)
    (template / "main.py").write_text(
        'APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"\n'
        'REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"\n',
        encoding="utf-8",
    )
    (template / "kernel-metadata.json").write_text("{}\n", encoding="utf-8")
    (template / "requirements-kaggle.txt").write_text("wrapt==1.17.3\n", encoding="utf-8")

    for source in package._app_files():
        relative = source.relative_to(package.ROOT)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n", encoding="utf-8")

    pdf_entries = []
    for versioned_id in ("1706.03762v7", "1810.04805v2"):
        pdf = papers / f"{versioned_id}.pdf"
        pdf.write_bytes(f"fixture-{versioned_id}".encode())
        pdf_entries.append(
            {
                "versioned_id": versioned_id,
                "pdf_url": f"https://arxiv.org/pdf/{versioned_id}",
                "pdf_sha256": _sha256(pdf),
                "page_count": 1,
            }
        )
    sources = root / "evaluation" / "suites" / "v0_5" / "development_10_sources.json"
    sources.write_text(json.dumps({"sources": pdf_entries}), encoding="utf-8")

    monkeypatch.setattr(package, "ROOT", root)
    monkeypatch.setattr(package, "TEMPLATE", template)
    monkeypatch.setattr(package, "DESTINATION", destination)
    monkeypatch.setattr(package, "PAPERS", papers)
    package.main()

    top_level_code = [path.name for path in destination.glob("*.py")]
    assert top_level_code == ["main.py"]
    generated = (destination / "main.py").read_text(encoding="utf-8")
    encoded = generated.split('APP_ARCHIVE_B64 = "', 1)[1].split('"', 1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        names = set(archive.namelist())
    assert "app/evaluation/internal_retrieval_runner.py" in names
    assert "scripts/run_internal_retrieval_ablation.py" in names
    assert "papers/1706.03762v7.pdf" not in names
    assert "papers/1810.04805v2.pdf" not in names
    assert all("data/evaluations" not in name for name in names)
    manifest = json.loads((destination / "source_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"kernel-metadata.json", "main.py"}
    assert len(manifest["public_sources"]) == 2


def test_kaggle_entrypoint_preserves_system_torch_and_cleans_environment() -> None:
    entrypoint = (
        package.ROOT
        / "evaluation"
        / "kaggle"
        / "internal_retrieval_v0_5"
        / "main.py"
    ).read_text(encoding="utf-8")
    requirements = (
        package.ROOT
        / "evaluation"
        / "kaggle"
        / "internal_retrieval_v0_5"
        / "requirements-kaggle.txt"
    ).read_text(encoding="utf-8")
    assert '"--system-site-packages"' in entrypoint
    assert 'VIRTUALENV_VERSION = "20.36.1"' in entrypoint
    assert '"--target"' in entrypoint
    assert '"--no-deps"' in entrypoint
    assert "torch.cuda.device_count()" in entrypoint
    assert "_download_papers()" in entrypoint
    assert "Downloaded PDF checksum mismatch" in entrypoint
    assert '"scripts.run_internal_retrieval_ablation"' in entrypoint
    assert "shutil.rmtree(ENV_ROOT" in entrypoint
    assert "shutil.rmtree(VIRTUALENV_BOOTSTRAP" in entrypoint
    assert "torch==" not in requirements
    assert "wrapt==" in requirements
