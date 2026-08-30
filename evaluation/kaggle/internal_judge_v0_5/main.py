"""Kaggle entry point for the advisory v0.5 internal-suite judge."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

OUTPUT = Path("/kaggle/working/__OUTPUT_DIRNAME__")
CODE_ROOT = Path("/tmp/internal_judge_source")
ENV_ROOT = Path("/tmp/internal_judge_env")
ENV_PYTHON = ENV_ROOT / "bin" / "python"
VIRTUALENV_BOOTSTRAP = Path("/tmp/virtualenv_bootstrap")
VIRTUALENV_VERSION = "20.36.1"
REQUIREMENTS = Path("/tmp/requirements-kaggle.txt")
APP_ARCHIVE_B64 = "__APP_ARCHIVE_B64__"
REQUIREMENTS_B64 = "__KAGGLE_REQUIREMENTS_B64__"


def _ensure_source() -> None:
    marker = CODE_ROOT / "app" / "run_evaluation_judge.py"
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


def _ensure_isolated_runtime() -> None:
    REQUIREMENTS.write_bytes(base64.b64decode(REQUIREMENTS_B64, validate=True))
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
            "--system-site-packages",
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
import pydantic
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
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "pydantic": pydantic.__version__,
    "wrapt": wrapt.__version__,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(ENV_PYTHON), "-c", probe], check=True, capture_output=True, text=True
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "resolved-requirements.txt").write_text(resolved, encoding="utf-8")
    (OUTPUT / "runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"runtime_preflight": runtime}, ensure_ascii=False, indent=2))


def main() -> None:
    _ensure_source()
    _ensure_isolated_runtime()
    _verify_runtime()
    subprocess.run(
        [
            str(ENV_PYTHON),
            "-m",
            "app.run_evaluation_judge",
            "--suite",
            str(CODE_ROOT / "evaluation" / "suites" / "v0_5" / "__SUITE_FILENAME__"),
            "--output",
            str(OUTPUT / "judge_report.json"),
            "--model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--batch-size",
            "2",
        ],
        check=True,
        cwd=CODE_ROOT,
    )


if __name__ == "__main__":
    main()
