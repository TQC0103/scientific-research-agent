"""Kaggle entry point for QASPER development ablations; generated source only."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = Path("/kaggle/working/qasper_v0_5")
DATASET = Path("/kaggle/working/qasper-dev-v0.3.json")
CODE_ROOT = Path("/kaggle/working/qasper_source")
ENV_ROOT = Path("/kaggle/working/qasper_env")
ENV_PYTHON = ENV_ROOT / "bin" / "python"
VIRTUALENV_BOOTSTRAP = Path("/kaggle/working/virtualenv_bootstrap")
VIRTUALENV_VERSION = "20.36.1"
REQUIREMENTS = Path("/kaggle/working/requirements-kaggle.txt")
APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"
REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"
QASPER_ARCHIVE_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)
QASPER_ARCHIVE_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"


def _ensure_app() -> None:
    marker = CODE_ROOT / "app" / "run_qasper.py"
    if marker.is_file():
        return
    payload = base64.b64decode(APP_ARCHIVE_B64, validate=True)
    root = CODE_ROOT.resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            target = (CODE_ROOT / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Embedded application archive contains an unsafe path.")
        archive.extractall(CODE_ROOT)


def _ensure_dataset() -> None:
    if DATASET.is_file():
        return
    archive = Path("/kaggle/working/qasper-train-dev-v0.3.tgz")
    digest = hashlib.sha256()
    with urllib.request.urlopen(QASPER_ARCHIVE_URL, timeout=120) as response, archive.open(
        "wb"
    ) as output:
        while block := response.read(1024 * 1024):
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != QASPER_ARCHIVE_SHA256:
        raise ValueError("Downloaded QASPER archive failed SHA-256 verification.")
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember(DATASET.name)
        source = bundle.extractfile(member)
        if source is None or not member.isfile():
            raise ValueError("QASPER dev member is missing from the verified archive.")
        with source, DATASET.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)


def _ensure_isolated_runtime() -> None:
    REQUIREMENTS.write_bytes(base64.b64decode(REQUIREMENTS_B64, validate=True))
    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Pinned Kaggle requirements are missing: {REQUIREMENTS}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--target",
            str(VIRTUALENV_BOOTSTRAP),
            f"virtualenv=={VIRTUALENV_VERSION}",
        ],
        check=True,
    )
    bootstrap_env = os.environ.copy()
    bootstrap_env["PYTHONPATH"] = str(VIRTUALENV_BOOTSTRAP)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "virtualenv",
            "--clear",
            "--no-download",
            str(ENV_ROOT),
        ],
        check=True,
        env=bootstrap_env,
    )
    subprocess.run(
        [
            str(ENV_PYTHON),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--requirement",
            str(REQUIREMENTS),
        ],
        check=True,
    )


def _verify_runtime() -> None:
    probe = """
import json
import accelerate
import pydantic
import sentence_transformers
import torch
import transformers

if not torch.cuda.is_available():
    raise RuntimeError("The isolated runtime cannot see a CUDA device.")
capability = torch.cuda.get_device_capability()
if capability != (7, 5):
    raise RuntimeError(f"Expected Tesla T4 capability (7, 5), received {capability}.")
left = torch.ones((64, 64), device="cuda")
right = left @ left
torch.cuda.synchronize()
print(json.dumps({
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_capability": capability,
    "cuda_preflight_sum": float(right.sum().item()),
    "sentence_transformers": sentence_transformers.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "pydantic": pydantic.__version__,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(ENV_PYTHON), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(completed.stdout.strip().splitlines()[-1])
    resolved = subprocess.run(
        [str(ENV_PYTHON), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (OUTPUT / "resolved-requirements.txt").write_text(resolved, encoding="utf-8")
    runtime["resolved_requirements_sha256"] = hashlib.sha256(
        resolved.encode("utf-8")
    ).hexdigest()
    (OUTPUT / "runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"runtime_preflight": runtime}, ensure_ascii=False, indent=2))


def _run(name: str, retrieval: str, generator: str) -> dict:
    destination = OUTPUT / name
    command = [
        str(ENV_PYTHON),
        "-m",
        "app.run_qasper",
        "--dataset",
        str(DATASET),
        "--split",
        "dev",
        "--retrieval-mode",
        retrieval,
        "--generator",
        generator,
        "--top-k",
        "5",
        "--output-dir",
        str(destination),
    ]
    if generator == "transformers":
        command.extend(
            [
                "--generator-model",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--generation-batch-size",
                "8",
            ]
        )
    subprocess.run(command, check=True, cwd=CODE_ROOT)
    return json.loads((destination / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    _ensure_app()
    _ensure_dataset()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _ensure_isolated_runtime()
    _verify_runtime()
    aggregate = {
        "lexical_retrieval_smoke": _run("lexical", "lexical", "none"),
        "dense_retrieval_smoke": _run("dense", "dense", "none"),
        "hybrid_answer": _run("hybrid_answer", "hybrid", "transformers"),
    }
    (OUTPUT / "ablation_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
