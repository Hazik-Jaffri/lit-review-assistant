"""
pipeline.py
------------
Orchestrates the full processing pipeline that runs when the user clicks
"Process Papers" in the Streamlit app:

    1. Extract text from each uploaded PDF
    2. Chunk + embed all papers, build the in-memory vector store
    3. Per-paper structured extraction (title, solution, limitations, confidence)
    4. Cross-paper gap analysis (clusters, contradictions, contribution hints)
    5. Assemble the final comparative table rows

Each step reports progress via a callback so the Streamlit UI can show a
live status instead of a frozen spinner during what can be a 1-3 minute
process for 15-20 papers on the free tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pdf_extractor import extract_pdf
from .chunking import chunk_many_papers
from .vector_store import VectorStore
from .extraction import extract_many
from .gap_analysis import run_gap_analysis


@dataclass
class PipelineResult:
    papers: list = field(default_factory=list)
    extracted_fields: list = field(default_factory=list)
    gap_result: object = None
    vector_store: object = None
    table_rows: list = field(default_factory=list)
    extraction_errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _build_table_rows(extracted_fields_list, gap_result):
    rows = []
    for i, f in enumerate(extracted_fields_list, start=1):
        if f.extraction_error:
            rows.append(
                {
                    "Reference": f"[{i}] {f.paper_file_name}",
                    "Title": f.title,
                    "Proposed Solution": "(extraction failed)",
                    "Limitations / Gaps": "(extraction failed)",
                    "Our Contribution": "",
                    "Confidence": 0,
                }
            )
            continue

        contribution_hint = ""
        if gap_result and not gap_result.error:
            contribution_hint = gap_result.per_paper_hints.get(f.paper_file_name, "")

        rows.append(
            {
                "Reference": f"[{i}] {f.paper_file_name}",
                "Title": f.title,
                "Proposed Solution": f.proposed_solution,
                "Limitations / Gaps": f.limitations,
                "Our Contribution": contribution_hint,
                "Confidence": f.confidence,
            }
        )
    return rows


def run_pipeline(gemini_client, uploaded_file_paths, original_names=None, progress_callback=None):
    """
    Parameters
    ----------
    gemini_client : GeminiClient
    uploaded_file_paths : list of paths to saved PDF files on disk
    original_names : matching list of display names (defaults to file basenames)
    progress_callback : callable(stage: str, detail: str, fraction: float) -> None
        Called repeatedly to update a Streamlit progress bar. `fraction` is 0-1.
    """

    def report(stage, detail, fraction):
        if progress_callback:
            progress_callback(stage, detail, fraction)

    result = PipelineResult()

    # ---- Stage 1: PDF extraction -----------------------------------------
    report("Extracting text", "Reading uploaded PDFs...", 0.05)
    names = original_names or [None] * len(uploaded_file_paths)
    papers = []
    extraction_errors = []
    n_files = max(len(uploaded_file_paths), 1)
    for idx, (path, name) in enumerate(zip(uploaded_file_paths, names)):
        try:
            papers.append(extract_pdf(path, original_file_name=name))
        except Exception as exc:  # noqa: BLE001
            extraction_errors.append((name or str(path), str(exc)))
        report("Extracting text", f"{idx + 1}/{len(uploaded_file_paths)} files", 0.05 + 0.10 * (idx + 1) / n_files)

    result.papers = papers
    result.extraction_errors = extraction_errors

    if not papers:
        result.warnings.append("No papers could be processed -- all PDF extractions failed.")
        return result

    # ---- Stage 2: Chunk + embed + build vector store ----------------------
    report("Indexing", "Splitting papers into chunks...", 0.20)
    chunks = chunk_many_papers(papers)

    report("Indexing", f"Embedding {len(chunks)} chunks via Gemini...", 0.30)
    chunk_texts = [c.text for c in chunks]
    embeddings = gemini_client.embed_batch(chunk_texts)

    report("Indexing", "Building vector store...", 0.45)
    store = VectorStore()
    store.add_chunks(chunks, embeddings)
    result.vector_store = store

    # ---- Stage 3: Per-paper structured extraction --------------------------
    report("Analyzing papers", "Extracting solution / limitations per paper...", 0.50)

    def _extract_progress(i, total, file_name):
        frac = 0.50 + 0.30 * (i / max(total, 1))
        report("Analyzing papers", f"{i}/{total}: {file_name}", frac)

    extracted_fields = extract_many(gemini_client, papers, progress_callback=_extract_progress)
    result.extracted_fields = extracted_fields

    failed = [f for f in extracted_fields if f.extraction_error]
    if failed:
        details = "; ".join(f"{f.paper_file_name} ({f.extraction_error})" for f in failed)
        result.warnings.append(f"{len(failed)} paper(s) failed structured extraction: {details}")

    # ---- Stage 4: Cross-paper gap analysis ---------------------------------
    report("Gap analysis", "Finding cross-paper patterns and gaps...", 0.85)
    gap_result = run_gap_analysis(gemini_client, extracted_fields)
    result.gap_result = gap_result
    if gap_result.error:
        result.warnings.append(f"Gap analysis could not be completed: {gap_result.error}")

    # ---- Stage 5: Assemble table -------------------------------------------
    report("Finalizing", "Assembling comparative table...", 0.95)
    result.table_rows = _build_table_rows(extracted_fields, gap_result)

    report("Done", "Pipeline complete.", 1.0)
    return result
