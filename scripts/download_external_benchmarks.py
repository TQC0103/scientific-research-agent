"""Download and verify the pinned public evaluation datasets used by v0.5."""

import hashlib
import json
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "evaluations" / "external"


@dataclass(frozen=True)
class Artifact:
    dataset: str
    version: str
    url: str
    filename: str
    sha256: str
    extract_directory: str
    license: str


ARTIFACTS = (
    Artifact(
        dataset="QASPER",
        version="0.3.0",
        url="https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        filename="qasper-train-dev-v0.3.tgz",
        sha256="a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a",
        extract_directory="qasper",
        license="CC BY 4.0",
    ),
    Artifact(
        dataset="QASPER",
        version="0.3.0",
        url="https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
        filename="qasper-test-and-evaluator-v0.3.tgz",
        sha256="72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72",
        extract_directory="qasper",
        license="CC BY 4.0",
    ),
    Artifact(
        dataset="SciFact",
        version="official-release-pinned-by-sha256",
        url="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        filename="scifact-data.tar.gz",
        sha256="11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
        extract_directory="scifact",
        license="Apache-2.0 repository; verify dataset terms before redistribution",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(artifact: Artifact, destination: Path) -> Path:
    archive = destination / artifact.filename
    if not archive.exists():
        print(f"Downloading {artifact.dataset} {artifact.version}: {artifact.url}", flush=True)
        with (
            urllib.request.urlopen(artifact.url, timeout=120) as response,
            archive.open("wb") as output,
        ):
            while block := response.read(1024 * 1024):
                output.write(block)
    actual = _sha256(archive)
    if actual != artifact.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {archive}: expected {artifact.sha256}, got {actual}."
        )
    return archive


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = []
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member in {archive}: {member.name}")
            members.append(member)
        bundle.extractall(destination, members=members, filter="data")


def main() -> None:
    destination = DEFAULT_DESTINATION
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        archive = _download(artifact, destination)
        _safe_extract(archive, destination / artifact.extract_directory)
        print(f"Verified and extracted {archive.name}", flush=True)
    manifest = {
        "downloaded_at": datetime.now(UTC).isoformat(),
        "artifacts": [asdict(artifact) for artifact in ARTIFACTS],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
