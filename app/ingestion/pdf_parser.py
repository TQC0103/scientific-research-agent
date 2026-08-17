import re
from pathlib import Path

import pymupdf as fitz

SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"approach|architecture|attention|experiments?|results?|discussion|limitations?|"
    r"conclusion|references)\s*$",
    re.IGNORECASE,
)


def parse_pdf(path: Path) -> list[dict]:
    pages: list[dict] = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()
            if text:
                pages.append({"page": page_number, "text": text})
    if not pages:
        raise ValueError(f"No extractable text found in {path.name}; it may be scanned.")
    return pages


def detect_section(line: str) -> str | None:
    normalized = " ".join(line.strip().split())
    match = SECTION_RE.match(normalized)
    if not match:
        return None
    heading = match.group(1).lower()
    canonical = {
        "methods": "Method",
        "methodology": "Method",
        "experiment": "Experiments",
        "result": "Results",
        "limitation": "Limitations",
    }
    return canonical.get(heading, heading.title())
