from app.ingestion.pdf_parser import detect_section


def chunk_pages(pages: list[dict], *, chunk_size: int = 1800, overlap: int = 250) -> list[dict]:
    chunks: list[dict] = []
    section = "Unknown"
    for page in pages:
        lines = page["text"].splitlines()
        blocks: list[tuple[str, str]] = []
        buffer: list[str] = []
        for line in lines:
            heading = detect_section(line)
            if heading:
                if buffer:
                    blocks.append((section, "\n".join(buffer)))
                    buffer = []
                section = heading
            else:
                buffer.append(line)
        if buffer:
            blocks.append((section, "\n".join(buffer)))

        for block_section, text in blocks:
            text = " ".join(text.split())
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text):
                    boundary = text.rfind(". ", start, end)
                    if boundary > start + chunk_size // 2:
                        end = boundary + 1
                body = text[start:end].strip()
                if len(body) >= 80:
                    chunks.append(
                        {
                            "page": page["page"],
                            "section": block_section,
                            "chunk_index": len(chunks),
                            "text": body,
                        }
                    )
                if end >= len(text):
                    break
                start = max(end - overlap, start + 1)
    return chunks
