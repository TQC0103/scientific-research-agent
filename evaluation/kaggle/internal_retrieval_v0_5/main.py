"""Generated Kaggle entry point for the internal retrieval ablation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

OUTPUT = Path("/kaggle/working/internal_retrieval_v0_5")
ENV_ROOT = Path("/kaggle/working/internal_retrieval_venv")
ENV_PYTHON = ENV_ROOT / "bin" / "python"
CODE_ROOT = Path("/tmp/internal_retrieval_source")
PAPERS = CODE_ROOT / "papers"
REQUIREMENTS = Path("/tmp/internal-retrieval-requirements.txt")
APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"
REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"


def _extract_app() -> None:
    payload = base64.b64decode(APP_ARCHIVE_B64, validate=True)
    root = CODE_ROOT.resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            target = (CODE_ROOT / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Embedded application archive contains an unsafe path.")
        archive.extractall(CODE_ROOT)


def _create_runtime() -> None:
    REQUIREMENTS.write_bytes(base64.b64decode(REQUIREMENTS_B64, validate=True))
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(ENV_ROOT)],
        check=True,
    )
    subprocess.run(
        [
            str(ENV_PYTHON),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            "--requirement",
            str(REQUIREMENTS),
        ],
        check=True,
    )


def _verify_runtime() -> None:
    probe = """
import json
import accelerate
import fitz
import pydantic
import sentence_transformers
import torch
import transformers
import wrapt

if not torch.cuda.is_available():
    raise RuntimeError("The isolated runtime cannot see a CUDA device.")
devices = []
for index in range(torch.cuda.device_count()):
    devices.append({
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "capability": list(torch.cuda.get_device_capability(index)),
    })
if not devices or any(item["capability"] != [7, 5] for item in devices):
    raise RuntimeError(f"Expected only Tesla T4 capability 7.5 devices, received {devices}.")
left = torch.ones((64, 64), device="cuda")
right = left @ left
torch.cuda.synchronize()
print(json.dumps({
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": devices,
    "cuda_preflight_sum": float(right.sum().item()),
    "sentence_transformers": sentence_transformers.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "pydantic": pydantic.__version__,
    "pymupdf": fitz.__version__,
    "wrapt": wrapt.__version__,
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
    runtime["resolved_requirements_sha256"] = hashlib.sha256(
        resolved.encode("utf-8")
    ).hexdigest()
    (OUTPUT / "runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "resolved-requirements.txt").write_text(resolved, encoding="utf-8")
    print(json.dumps({"runtime_preflight": runtime}, ensure_ascii=False, indent=2))


def _run_ablation() -> None:
    command = [
        str(ENV_PYTHON),
        str(CODE_ROOT / "scripts" / "run_internal_retrieval_ablation.py"),
        "--suite",
        str(CODE_ROOT / "evaluation" / "suites" / "v0_5" / "development_10.json"),
        "--sources",
        str(
            CODE_ROOT
            / "evaluation"
            / "suites"
            / "v0_5"
            / "development_10_sources.json"
        ),
        "--papers-dir",
        str(PAPERS),
        "--output-dir",
        str(OUTPUT / "ablation"),
        "--modes",
        "lexical",
        "dense",
        "hybrid",
        "--top-k",
        "5",
        "--dense-model",
        "Qwen/Qwen3-Embedding-0.6B",
        "--dense-revision",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "--dense-batch-size",
        "8",
    ]
    subprocess.run(command, check=True, cwd=CODE_ROOT)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        _extract_app()
        _create_runtime()
        _verify_runtime()
        _run_ablation()
    finally:
        shutil.rmtree(ENV_ROOT, ignore_errors=True)
        shutil.rmtree(CODE_ROOT, ignore_errors=True)
        REQUIREMENTS.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
