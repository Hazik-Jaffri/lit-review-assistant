"""
pdf_extractor.py
-----------------
Extracts clean text from uploaded research-paper PDFs.

Uses PyMuPDF (fitz) because it is fast, dependency-light, and handles
two-column academic paper layouts noticeably better than pypdf for
plain text extraction (reading order is closer to correct).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedPaper:
    file_name: str
    full_text: str
    num_pages: int
    detected_title: str


def _clean_text(text: str) -> str:
    """Collapse excessive whitespace and strip page-artifact noise."""
    text = text.replace("\x0c", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_title(first_page_text: str, file_name: str) -> str:
    """
    Heuristic title detection: the first non-empty line on page 1 that is
    reasonably long and not an obvious header/footer artifact (e.g. an
    arXiv banner or a running page number).
    """
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
    for line in lines[:12]:
        if len(line) < 8 or len(line) > 220:
            continue
        if re.match(r"^(arxiv:|doi:|page \d+|issn|http)", line, re.IGNORECASE):
            continue
        if line.isdigit():
            continue
        return line
    return Path(file_name).stem.replace("_", " ").replace("-", " ").strip()


def extract_pdf(file_path, original_file_name=None) -> ExtractedPaper:
    """
    Extract text from a single PDF file on disk.

    Parameters
    ----------
    file_path: path to the PDF on disk (e.g. saved upload).
    original_file_name: the name to record/display (defaults to file_path's name).
    """
    file_path = Path(file_path)
    display_name = original_file_name or file_path.name

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf") from exc

    doc = fitz.open(str(file_path))
    try:
        num_pages = doc.page_count
        page_texts = []
        first_page_text = ""
        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            if i == 0:
                first_page_text = page_text
            page_texts.append(page_text)
    finally:
        doc.close()

    full_text = _clean_text("\n".join(page_texts))
    if not full_text or len(full_text) < 50:
        raise ValueError(
            f"'{display_name}' produced almost no extractable text. "
            "It may be a scanned/image-only PDF without OCR."
        )

    title = _guess_title(first_page_text, display_name)

    return ExtractedPaper(
        file_name=display_name,
        full_text=full_text,
        num_pages=num_pages,
        detected_title=title,
    )


class PartialExtractionError(Exception):
    """Raised when some PDFs fail extraction; carries the successful ones."""

    def __init__(self, message, successful_results):
        super().__init__(message)
        self.successful_results = successful_results


def extract_many(file_paths, original_names=None):
    """Extract a batch of PDFs, skipping (and reporting) any that fail."""
    results = []
    errors = []
    names = original_names or [Path(p).name for p in file_paths]

    for path, name in zip(file_paths, names):
        try:
            results.append(extract_pdf(path, original_file_name=name))
        except Exception as exc:  # noqa: BLE001 - continue with other files
            errors.append((name, str(exc)))

    if errors:
        error_summary = "; ".join(f"{name}: {msg}" for name, msg in errors)
        raise PartialExtractionError(error_summary, results)

    return results
