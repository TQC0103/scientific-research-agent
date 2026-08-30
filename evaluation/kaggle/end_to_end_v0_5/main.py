"""Generated Kaggle entry point for the production end-to-end benchmark."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

OUTPUT = Path("/kaggle/working/__OUTPUT_DIRNAME__")
ENV_ROOT = Path("/kaggle/working/end_to_end_v0_5_venv")
ENV_PYTHON = ENV_ROOT / "bin" / "python"
CODE_ROOT = Path("/tmp/end_to_end_v0_5_source")
DATA_ROOT = Path("/tmp/end_to_end_v0_5_data")
REQUIREMENTS = Path("/tmp/end-to-end-v0-5-requirements.txt")
VIRTUALENV_BOOTSTRAP = Path("/tmp/end-to-end-v0-5-virtualenv")
VIRTUALENV_VERSION = "20.36.1"
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
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "--no-cache-dir", "--target", str(VIRTUALENV_BOOTSTRAP),
         f"virtualenv=={VIRTUALENV_VERSION}"],
        check=True,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(VIRTUALENV_BOOTSTRAP)
    subprocess.run(
        [sys.executable, "-m", "virtualenv", "--clear", "--no-download",
         "--system-site-packages", str(ENV_ROOT)],
        check=True,
        env=environment,
    )
    subprocess.run(
        [str(ENV_PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
         "--no-cache-dir", "--no-deps", "--requirement", str(REQUIREMENTS)],
        check=True,
    )


def _verify_runtime() -> None:
    probe = """
import json
import accelerate, arxiv, faiss, fitz, langgraph, pydantic, sentence_transformers
import torch, transformers, wrapt
if not torch.cuda.is_available():
    raise RuntimeError("The isolated runtime cannot see CUDA.")
devices = [{"index": i, "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i))}
           for i in range(torch.cuda.device_count())]
if not devices or any(item["capability"] != [7, 5] for item in devices):
    raise RuntimeError(f"Expected only Tesla T4 capability 7.5 devices, received {devices}.")
left = torch.ones((64, 64), device="cuda")
right = left @ left
torch.cuda.synchronize()
print(json.dumps({"python": __import__("sys").version, "torch": torch.__version__,
 "cuda_runtime": torch.version.cuda, "cuda_device_count": torch.cuda.device_count(),
 "cuda_devices": devices,
 "inference_device_ids": [0, 1] if len(devices) > 1 else [0],
 "embedding_device": "cuda:1" if len(devices) > 1 else "cpu",
 "cuda_preflight_sum": float(right.sum().item()),
 "transformers": transformers.__version__, "accelerate": accelerate.__version__,
 "sentence_transformers": sentence_transformers.__version__,
 "pydantic": pydantic.__version__, "pymupdf": fitz.__version__,
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
    runtime["resolved_requirements_sha256"] = hashlib.sha256(
        resolved.encode("utf-8")
    ).hexdigest()
    (OUTPUT / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (OUTPUT / "resolved-requirements.txt").write_text(resolved, encoding="utf-8")
    print(json.dumps({"runtime_preflight": runtime}, indent=2))


def _run_benchmark() -> None:
    subprocess.run(
        [str(ENV_PYTHON), "-m", "scripts.run_end_to_end_transformers",
         "--suite", str(CODE_ROOT / "evaluation/suites/v0_5/__SUITE_FILENAME__"),
         "--sources", str(CODE_ROOT / "evaluation/suites/v0_5/__SOURCES_FILENAME__"),
         "--output-dir", str(OUTPUT / "report"),
         "--data-dir", str(DATA_ROOT),
         "--retrieval-mode", "__RETRIEVAL_MODE__",
         "--config-name", "__CONFIG_NAME__",
         "--smoke-cases", "1"],
        check=True,
        cwd=CODE_ROOT,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        _extract_app()
        _create_runtime()
        _verify_runtime()
        _run_benchmark()
    finally:
        shutil.rmtree(ENV_ROOT, ignore_errors=True)
        shutil.rmtree(CODE_ROOT, ignore_errors=True)
        shutil.rmtree(DATA_ROOT, ignore_errors=True)
        shutil.rmtree(VIRTUALENV_BOOTSTRAP, ignore_errors=True)
        REQUIREMENTS.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
