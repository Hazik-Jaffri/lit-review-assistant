"""
gap_analysis.py
-----------------
The novelty component of this project: instead of just summarizing papers
one by one, this module asks Gemini to reason ACROSS the whole batch of
extracted fields to produce:

    1. Thematic gap clusters  - groups of related, recurring gaps across
       multiple papers (e.g. "low-resource language support" appears as a
       gap in 6 of 18 papers) rather than 18 disconnected gap sentences.
    2. Cross-paper contradictions/disagreements - pairs of papers that
       report conflicting findings or take opposing positions on the same
       sub-problem.
    3. A suggested "Our Contribution" sentence per paper - a *draft*
       starting point the student must review/edit themselves, framed
       around the single most relevant gap cluster.

This is what is referenced as the "5% novelty" element when explaining the
project's research contribution: an automated, evidence-grounded gap
synthesis step that a plain summarizer does not perform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gemini_client import GeminiClient, GeminiAPIError

GAP_ANALYSIS_PROMPT_TEMPLATE = """You are helping a graduate student synthesize a literature review across
multiple papers. Below is a JSON list of papers, each with its proposed
solution and stated/inferred limitations.

Your task: find PATTERNS ACROSS papers, not just restate each paper alone.

Return STRICT JSON with exactly these keys:

1. "gap_clusters": a list of 3-7 objects, each with:
   - "theme": a short (3-6 word) name for this recurring gap
   - "description": 1-2 sentences describing the gap pattern
   - "papers": a list of paper file names (must exactly match the file names
     given below) that exhibit this gap
   - "suggested_contribution": 1 sentence suggesting, in general terms, what
     kind of contribution could address this gap (a STARTING POINT for the
     student to refine -- not a finished claim)

2. "contradictions": a list of 0-5 objects, each with:
   - "paper_a": file name
   - "paper_b": file name
   - "issue": 1-2 sentences describing what these two papers seem to
     disagree about or report conflicting results on. Only include this if
     there is a genuine apparent conflict -- an empty list is fine if none
     of the papers conflict with one another.

3. "per_paper_contribution_hint": a list of objects, one per input paper,
   each with:
   - "paper_file_name": file name (must match exactly)
   - "hint": 1 sentence connecting that paper's gap to the most relevant
     gap_cluster theme, phrased as a suggestion (e.g. "Could be extended to
     address [theme] by ..."), for the student to adapt into their own words.

Ground every claim only in the information given below. Do not invent paper
content that isn't implied by the provided proposed_solution/limitations text.

PAPERS:
{papers_json}

Return ONLY the JSON object, no extra commentary.
"""


@dataclass
class GapCluster:
    theme: str
    description: str
    papers: list = field(default_factory=list)
    suggested_contribution: str = ""


@dataclass
class Contradiction:
    paper_a: str
    paper_b: str
    issue: str


@dataclass
class GapAnalysisResult:
    gap_clusters: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    per_paper_hints: dict = field(default_factory=dict)
    error: object = None


def run_gap_analysis(client, extracted_fields_list):
    """
    extracted_fields_list: list[extraction.ExtractedFields]
    Only papers with a successful extraction (no extraction_error) are sent.
    """
    import json

    usable = [f for f in extracted_fields_list if not f.extraction_error]
    if len(usable) < 2:
        return GapAnalysisResult(
            error="Need at least 2 successfully-extracted papers to run cross-paper gap analysis."
        )

    papers_payload = [
        {
            "file_name": f.paper_file_name,
            "title": f.title,
            "proposed_solution": f.proposed_solution,
            "limitations": f.limitations,
        }
        for f in usable
    ]

    prompt = GAP_ANALYSIS_PROMPT_TEMPLATE.format(papers_json=json.dumps(papers_payload, indent=2))

    try:
        data = client.generate_json(prompt, temperature=0.3, max_output_tokens=3072)
        if not isinstance(data, dict):
            raise GeminiAPIError(f"Expected a JSON object, got: {type(data)}")

        clusters = [
            GapCluster(
                theme=str(c.get("theme", "")).strip(),
                description=str(c.get("description", "")).strip(),
                papers=list(c.get("papers", []) or []),
                suggested_contribution=str(c.get("suggested_contribution", "")).strip(),
            )
            for c in (data.get("gap_clusters") or [])
            if isinstance(c, dict)
        ]

        contradictions = [
            Contradiction(
                paper_a=str(c.get("paper_a", "")).strip(),
                paper_b=str(c.get("paper_b", "")).strip(),
                issue=str(c.get("issue", "")).strip(),
            )
            for c in (data.get("contradictions") or [])
            if isinstance(c, dict)
        ]

        per_paper_hints = {}
        for item in data.get("per_paper_contribution_hint") or []:
            if isinstance(item, dict) and item.get("paper_file_name"):
                per_paper_hints[str(item["paper_file_name"])] = str(item.get("hint", "")).strip()

        return GapAnalysisResult(
            gap_clusters=clusters,
            contradictions=contradictions,
            per_paper_hints=per_paper_hints,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to UI
        return GapAnalysisResult(error=str(exc))
