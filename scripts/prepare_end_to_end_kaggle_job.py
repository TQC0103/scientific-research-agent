"""Prepare the narrow, ignored Kaggle end-to-end benchmark source."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "evaluation" / "kaggle" / "end_to_end_v0_5"
DESTINATION = ROOT / "data" / "evaluations" / "kaggle_jobs" / "end_to_end_v0_5"
KAGGLE_KERNEL_TEXT_LIMIT = 50
DEFAULT_SUITE_NAME = "development_10"
DEFAULT_RETRIEVAL_MODE = "rrf"
DEFAULT_CONFIG_NAME = "hybrid_verified_citation_scoped_v4_qwen3_4b_fp16"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(suite_name: str = DEFAULT_SUITE_NAME) -> dict[Path, str]:
    relative = [
        "app/__init__.py", "app/config.py", "app/agent/__init__.py", "app/agent/graph.py",
        "app/agent/state.py", "app/db/__init__.py", "app/db/database.py",
        "app/ingestion/__init__.py", "app/ingestion/chunking.py",
        "app/ingestion/indexing.py", "app/ingestion/pdf_parser.py",
        "app/models/__init__.py", "app/models/claim_verifier.py", "app/models/claims.py",
        "app/models/llm.py", "app/models/verifier.py", "app/retrieval/__init__.py",
        "app/retrieval/vector_store.py", "app/tools/__init__.py", "app/tools/arxiv_search.py",
        "app/tools/paper_download.py", "app/evaluation/__init__.py",
        "app/evaluation/citations.py", "app/evaluation/end_to_end.py",
        "app/evaluation/external.py",
        "app/evaluation/loader.py", "app/evaluation/metrics.py", "app/evaluation/models.py",
        "app/evaluation/retrieval.py", "scripts/run_end_to_end_transformers.py",
        f"evaluation/suites/v0_5/{suite_name}.json",
        f"evaluation/suites/v0_5/{suite_name}_sources.json",
    ]
    return {ROOT / name: name for name in relative}


def _embedded_source(suite_name: str = DEFAULT_SUITE_NAME) -> str:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, target in sorted(
            _source_files(suite_name).items(), key=lambda item: item[1]
        ):
            info = zipfile.ZipInfo(target, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return base64.b64encode(payload.getvalue()).decode("ascii")


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the isolated Task 11 Kaggle source bundle."
    )
    parser.add_argument(
        "--kernel-slug",
        help="Optional full owner/slug written before provenance hashes are computed.",
    )
    parser.add_argument(
        "--title",
        help="Optional Kaggle kernel title written before provenance hashes are computed.",
    )
    parser.add_argument("--suite-name", default=DEFAULT_SUITE_NAME)
    parser.add_argument(
        "--retrieval-mode",
        choices=("rrf", "windowed_rerank"),
        default=DEFAULT_RETRIEVAL_MODE,
    )
    parser.add_argument("--config-name")
    parser.add_argument("--destination", type=Path)
    return parser.parse_args(argv)


def _validate_kernel_identity(kernel_slug: str | None, title: str | None) -> None:
    if kernel_slug:
        owner, separator, slug = kernel_slug.partition("/")
        if not separator or not owner or not slug:
            raise ValueError("--kernel-slug must use owner/slug format.")
        if len(slug) > KAGGLE_KERNEL_TEXT_LIMIT:
            raise ValueError(
                "Kaggle kernel slug exceeds the 50-character service limit: "
                f"{len(slug)} characters."
            )
    if title and len(title) > KAGGLE_KERNEL_TEXT_LIMIT:
        raise ValueError(
            "Kaggle kernel title exceeds the 50-character service limit: "
            f"{len(title)} characters."
        )


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv or [])
    _validate_kernel_identity(args.kernel_slug, args.title)
    if not args.suite_name.isidentifier() or not args.suite_name.startswith(
        "development_"
    ):
        raise ValueError(f"Unsafe suite name: {args.suite_name}")
    config_name = args.config_name or DEFAULT_CONFIG_NAME
    destination = (args.destination or DESTINATION).resolve()
    allowed_root = (ROOT / "data" / "evaluations" / "kaggle_jobs").resolve()
    if not destination.is_relative_to(allowed_root) or destination == allowed_root:
        raise ValueError(f"Unsafe Kaggle job destination: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    requirements = base64.b64encode(
        (TEMPLATE / "requirements-kaggle.txt").read_bytes()
    ).decode("ascii")
    for source in TEMPLATE.iterdir():
        if not source.is_file() or source.name == "requirements-kaggle.txt":
            continue
        target = destination / source.name
        if source.name == "main.py":
            target.write_text(
                source.read_text(encoding="utf-8")
                .replace("__APP_ARCHIVE_B64__", _embedded_source(args.suite_name))
                .replace("__KAGGLE_REQUIREMENTS_B64__", requirements)
                .replace("__SUITE_FILENAME__", f"{args.suite_name}.json")
                .replace(
                    "__SOURCES_FILENAME__", f"{args.suite_name}_sources.json"
                )
                .replace("__OUTPUT_DIRNAME__", destination.name)
                .replace("__RETRIEVAL_MODE__", args.retrieval_mode)
                .replace(
                    "__CONFIG_NAME__",
                    config_name,
                ),
                encoding="utf-8",
            )
        elif source.name == "kernel-metadata.json" and (
            args.kernel_slug or args.title
        ):
            metadata = json.loads(source.read_text(encoding="utf-8"))
            if args.kernel_slug:
                metadata["id"] = args.kernel_slug
            if args.title:
                metadata["title"] = args.title
            target.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        else:
            shutil.copy2(source, target)
    sources = json.loads(
        (
            ROOT
            / "evaluation"
            / "suites"
            / "v0_5"
            / f"{args.suite_name}_sources.json"
        ).read_text(encoding="utf-8")
    )["sources"]
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "source": "scientific-research-agent v0.5 production end-to-end benchmark",
        "llm_model": "Qwen/Qwen3-4B",
        "llm_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "suite_name": args.suite_name,
        "retrieval_mode": args.retrieval_mode,
        "config_name": config_name,
        "reranker_model": (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
            if args.retrieval_mode == "windowed_rerank"
            else None
        ),
        "reranker_revision": (
            "233902d25c440f23af6f7d6e94d2946bac0bee0a"
            if args.retrieval_mode == "windowed_rerank"
            else None
        ),
        "public_sources": sources,
        "files": {
            str(path.relative_to(destination)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
