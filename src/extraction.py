"""
extraction.py
--------------
Per-paper structured extraction: asks Gemini to read a paper's full text
and produce the fields required by the course's Comparative Analysis Table
(Reference, Title, Proposed Solution, Limitations/Gaps) plus an internal
Confidence field used to flag possible hallucination.

The "Our Contribution" column is intentionally NOT generated here -- it
depends on the student's own proposed methodology, which the gap-analysis
module drafts as a suggestion, but the student must finalize it themselves
(this keeps the tool aligned with the academic-integrity policy: it assists
analysis, it does not write the student's claimed contribution for them).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .gemini_client import GeminiClient, GeminiAPIError

EXTRACTION_PROMPT_TEMPLATE = """You are assisting a graduate student in building a literature-review
comparative table. Read the research paper text below and extract the requested
fields as STRICT JSON. Be factual and conservative: only state what the paper
actually claims or demonstrates. Do not invent results, numbers, or claims that
are not supported by the text.

Return a JSON object with exactly these keys:
- "title": the paper's title (string)
- "proposed_solution": 2-3 sentences describing the approach/method/system the
  paper proposes (string)
- "limitations": 2-3 sentences describing what the paper explicitly states as
  limitations, OR gaps you can reasonably infer it does not address (string)
- "key_findings": 1-2 sentences summarizing the main result (string)
- "confidence": your own confidence (0-100, integer) that the above fields are
  an accurate, well-supported reading of the provided text. Use a LOWER score
  if the text is truncated, ambiguous, or you had to infer rather than read
  explicit statements.

Detected title hint (the text may begin mid-sentence; use this if more reliable
than the body text): {title_hint}

PAPER TEXT:
---
{paper_text}
---

Return ONLY the JSON object, no extra commentary.
"""


@dataclass
class ExtractedFields:
    paper_file_name: str
    title: str
    proposed_solution: str
    limitations: str
    key_findings: str
    confidence: int
    extraction_error: object = None


def extract_paper_fields(client, paper):
    """
    Run structured extraction for a single ExtractedPaper.
    On failure, returns an ExtractedFields with extraction_error set so the
    pipeline can surface a clear per-paper error in the UI rather than
    crashing the whole batch.
    """
    truncated_text = paper.full_text[: config.MAX_CHARS_PER_PAPER_FOR_EXTRACTION]
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        title_hint=paper.detected_title,
        paper_text=truncated_text,
    )

    try:
        data = client.generate_json(prompt, temperature=0.1, max_output_tokens=1024)
        if not isinstance(data, dict):
            raise GeminiAPIError(f"Expected a JSON object, got: {type(data)}")

        confidence_raw = data.get("confidence", 50)
        try:
            confidence = int(confidence_raw)
        except (TypeError, ValueError):
            confidence = 50
        confidence = max(0, min(100, confidence))

        return ExtractedFields(
            paper_file_name=paper.file_name,
            title=str(data.get("title") or paper.detected_title).strip(),
            proposed_solution=str(data.get("proposed_solution") or "").strip(),
            limitations=str(data.get("limitations") or "").strip(),
            key_findings=str(data.get("key_findings") or "").strip(),
            confidence=confidence,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to UI, not raised
        return ExtractedFields(
            paper_file_name=paper.file_name,
            title=paper.detected_title,
            proposed_solution="",
            limitations="",
            key_findings="",
            confidence=0,
            extraction_error=str(exc),
        )


def extract_many(client, papers, progress_callback=None):
    """
    Extract fields for each paper sequentially (Gemini free tier rate limits
    make concurrent calls risky). Calls progress_callback(i, total, file_name)
    after each paper so the Streamlit UI can show a live progress bar.
    """
    results = []
    total = len(papers)
    for i, paper in enumerate(papers, start=1):
        result = extract_paper_fields(client, paper)
        results.append(result)
        if progress_callback:
            progress_callback(i, total, paper.file_name)
    return results
