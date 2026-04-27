from __future__ import annotations

import hashlib

from .config import Settings
from .models import ChunkRecord, ParsedDocument


def chunk_documents(documents: list[ParsedDocument], settings: Settings) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []

    for doc in documents:
        pieces = split_text(doc.text, settings.chunk_size, settings.chunk_overlap)
        for order, piece in enumerate(pieces, start=1):
            chunk_id = _chunk_id(doc.doc_id, doc.page_number, order, piece)
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    title=doc.title,
                    text=piece,
                    modality=doc.modality,
                    page_number=doc.page_number,
                    order=order,
                    metadata=doc.extra_meta.copy(),
                )
            )
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    results: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            results.append(piece)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return results


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _chunk_id(doc_id: str, page_number: int | None, order: int, text: str) -> str:
    digest = hashlib.md5(f"{doc_id}:{page_number}:{order}:{text[:100]}".encode("utf-8")).hexdigest()
    return f"chunk_{digest}"
