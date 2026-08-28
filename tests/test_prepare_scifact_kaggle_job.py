import base64
import io
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts import prepare_scifact_kaggle_job as package


def test_real_scifact_archive_imports_in_isolation(tmp_path: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(package._embedded_source()))) as archive:
        archive.extractall(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from app.evaluation.scifact import run_scifact_benchmark; "
                "print(run_scifact_benchmark)"
            ),
            str(tmp_path),
        ],
        cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_scifact_entrypoint_pins_data_t4_and_safe_generation() -> None:
    entrypoint = (package.TEMPLATE / "main.py").read_text(encoding="utf-8")
    runner = (package.ROOT / "scripts" / "run_scifact.py").read_text(encoding="utf-8")
    requirements = (package.TEMPLATE / "requirements-kaggle.txt").read_text(encoding="utf-8")
    assert package._source_files()[package.ROOT / "app/evaluation/loader.py"]
    assert 'DATA_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"' in entrypoint
    assert 'item["capability"] != [7, 5]' not in entrypoint
    assert 'x["capability"] != [7, 5]' in entrypoint
    assert '"inference_device_ids": [0]' in entrypoint
    assert '"--smoke-cases", "3"' in entrypoint
    assert "torch==" not in requirements
    assert 'self.tokenizer.padding_side = "left"' in runner
    assert "enable_thinking=False" in runner
    assert "do_sample=False" in runner
    assert "out of memory" in runner
    assert "smoke.parse_failure_count == smoke.case_count" in runner
