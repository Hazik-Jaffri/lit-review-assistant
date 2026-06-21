"""
chunking.py
------------
Splits paper full-text into overlapping chunks suitable for embedding.

A simple character-window splitter is used (rather than a tokenizer-aware
splitter) to avoid pulling in a heavy tokenizer dependency. The overlap
ensures that ideas spanning a chunk boundary are not lost from retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class Chunk:
    chunk_id: str
    paper_file_name: str
    text: str
    chunk_index: int


def chunk_text(
    text,
    paper_file_name,
    chunk_size=config.CHUNK_SIZE_CHARS,
    overlap=config.CHUNK_OVERLAP_CHARS,
):
    """Split `text` into overlapping chunks, breaking on paragraph/sentence
    boundaries where possible so chunks don't cut mid-sentence."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_len = len(text)
    index = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            window_floor = start + int(chunk_size * 0.8)
            boundary = text.rfind("\n\n", window_floor, end)
            if boundary == -1:
                boundary = text.rfind(". ", window_floor, end)
            if boundary != -1 and boundary > window_floor:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_file_name}::chunk{index}",
                    paper_file_name=paper_file_name,
                    text=piece,
                    chunk_index=index,
                )
            )
            index += 1

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_many_papers(papers):
    """Chunk a list of ExtractedPaper objects (from pdf_extractor)."""
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunk_text(paper.full_text, paper.file_name))
    return all_chunks
