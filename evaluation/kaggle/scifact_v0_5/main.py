"""Generated Kaggle entry point for native-label SciFact evaluation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

OUTPUT = Path("/kaggle/working/scifact_v0_5")
ENV_ROOT = Path("/kaggle/working/scifact_v0_5_venv")
ENV_PYTHON = ENV_ROOT / "bin" / "python"
CODE_ROOT = Path("/tmp/scifact_v0_5_source")
DATA_ROOT = Path("/tmp/scifact_v0_5_data")
ARCHIVE = Path("/tmp/scifact-data.tar.gz")
REQUIREMENTS = Path("/tmp/scifact-v0-5-requirements.txt")
VIRTUALENV_BOOTSTRAP = Path("/tmp/scifact-v0-5-virtualenv")
VIRTUALENV_VERSION = "20.36.1"
DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"
REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"


def _extract_app() -> None:
    payload = base64.b64decode(APP_ARCHIVE_B64, validate=True)
    root = CODE_ROOT.resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            if not (CODE_ROOT / member.filename).resolve().is_relative_to(root):
                raise ValueError("Embedded application archive contains an unsafe path.")
        archive.extractall(CODE_ROOT)


def _download_data() -> None:
    with urllib.request.urlopen(DATA_URL, timeout=120) as response, ARCHIVE.open("wb") as output:
        shutil.copyfileobj(response, output)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"SciFact SHA-256 mismatch: {digest}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    root = DATA_ROOT.resolve()
    wanted = {"data/corpus.jsonl", "data/claims_dev.jsonl"}
    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        members = []
        for member in bundle.getmembers():
            normalized = member.name.removeprefix("./")
            if normalized not in wanted:
                continue
            if member.issym() or member.islnk():
                raise ValueError("SciFact archive contains an unsafe link.")
            target = (DATA_ROOT / normalized).resolve()
            if not target.is_relative_to(root):
                raise ValueError("SciFact archive contains an unsafe path.")
            members.append(member)
        if {member.name.removeprefix("./") for member in members} != wanted:
            raise ValueError("SciFact archive is missing dev data.")
        bundle.extractall(DATA_ROOT, members=members, filter="data")


def _create_runtime() -> None:
    REQUIREMENTS.write_bytes(base64.b64decode(REQUIREMENTS_B64, validate=True))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "--no-cache-dir", "--target", str(VIRTUALENV_BOOTSTRAP),
         f"virtualenv=={VIRTUALENV_VERSION}"], check=True
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(VIRTUALENV_BOOTSTRAP)
    subprocess.run(
        [sys.executable, "-m", "virtualenv", "--clear", "--no-download",
         "--system-site-packages", str(ENV_ROOT)], check=True, env=environment
    )
    subprocess.run(
        [str(ENV_PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
         "--no-cache-dir", "--no-deps", "--requirement", str(REQUIREMENTS)], check=True
    )


def _verify_runtime() -> None:
    probe = """
import json, accelerate, pydantic, torch, transformers, wrapt
if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
devices = [{"index": i, "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i))}
           for i in range(torch.cuda.device_count())]
if not devices or any(x["capability"] != [7, 5] for x in devices):
    raise RuntimeError(f"Expected Tesla T4 capability 7.5, got {devices}")
x = torch.ones((64, 64), device="cuda"); y = x @ x; torch.cuda.synchronize()
print(json.dumps({"python": __import__("sys").version, "torch": torch.__version__,
 "cuda_runtime": torch.version.cuda, "cuda_device_count": torch.cuda.device_count(),
 "cuda_devices": devices, "inference_device_ids": [0],
 "cuda_preflight_sum": float(y.sum()), "transformers": transformers.__version__,
 "accelerate": accelerate.__version__, "pydantic": pydantic.__version__,
 "wrapt": wrapt.__version__}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(ENV_PYTHON), "-c", probe], check=True, capture_output=True, text=True
    )
    runtime = json.loads(completed.stdout.strip().splitlines()[-1])
    resolved = subprocess.run(
        [str(ENV_PYTHON), "-m", "pip", "freeze", "--all"], check=True,
        capture_output=True, text=True
    ).stdout
    runtime["resolved_requirements_sha256"] = hashlib.sha256(resolved.encode()).hexdigest()
    runtime["dataset_url"] = DATA_URL
    runtime["dataset_sha256"] = DATA_SHA256
    (OUTPUT / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (OUTPUT / "resolved-requirements.txt").write_text(resolved, encoding="utf-8")
    print(json.dumps({"runtime_preflight": runtime}, indent=2))


def _run() -> None:
    subprocess.run(
        [str(ENV_PYTHON), "-m", "scripts.run_scifact",
         "--corpus", str(DATA_ROOT / "data/corpus.jsonl"),
         "--claims", str(DATA_ROOT / "data/claims_dev.jsonl"),
         "--source-split", "dev", "--output-dir", str(OUTPUT / "report"),
         "--model", "Qwen/Qwen3-4B",
         "--model-revision", "1cfa9a7208912126459214e8b04321603b3df60c",
         "--batch-size", "4", "--max-new-tokens", "256", "--smoke-cases", "3"],
        check=True, cwd=CODE_ROOT
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        _extract_app(); _download_data(); _create_runtime(); _verify_runtime(); _run()
    finally:
        shutil.rmtree(ENV_ROOT, ignore_errors=True)
        shutil.rmtree(CODE_ROOT, ignore_errors=True)
        shutil.rmtree(DATA_ROOT, ignore_errors=True)
        shutil.rmtree(VIRTUALENV_BOOTSTRAP, ignore_errors=True)
        ARCHIVE.unlink(missing_ok=True); REQUIREMENTS.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
