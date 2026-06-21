# Automated Literature Review & Gap Analysis Assistant

An NLP/Deep-Learning course project that automates the most time-consuming
part of writing a research report: reading a stack of papers and building
the Comparative Analysis Table (Reference, Title, Proposed Solution,
Limitations/Gaps, Our Contribution) -- while also surfacing patterns across
papers that a manual, paper-by-paper read-through tends to miss.

## What it does

1. **Upload PDFs** of the papers you want in your literature review.
2. **Per-paper extraction** -- Gemini reads each paper and extracts its
   title, proposed solution, and limitations/gaps, with a confidence score
   so you know which rows need a manual double-check.
3. **Cross-paper gap analysis (the novelty piece)** -- instead of just
   summarizing papers one at a time, the system reasons *across* all
   extracted fields to find:
   - **Thematic gap clusters**: recurring limitations that show up in
     multiple papers (e.g. "6 of 18 papers don't evaluate low-resource
     languages"), each with a suggested research direction.
   - **Cross-paper contradictions**: pairs of papers that report
     conflicting findings on the same sub-problem.
   - **Per-paper contribution hints**: a draft starting point for the
     "Our Contribution" column, tied to the most relevant gap cluster.
4. **Comparative table + export** -- the table renders in the app and
   exports to CSV or a ready-to-paste Word table matching the course's
   exact 5-column schema (plus a Confidence column).
5. **Ask Your Papers (RAG chat)** -- a retrieval-augmented chat that
   answers questions grounded only in your uploaded papers, with source
   citations.

## Why this satisfies both course requirement documents

| Requirement | Source | How it's satisfied |
|---|---|---|
| NLP component | NLP course statement | Summarization, semantic search/embeddings, RAG, information extraction |
| Working GUI | NLP course statement | Streamlit app with 4 functional tabs |
| Real dataset / real use case | NLP course statement | User-uploaded real research PDFs |
| Generative AI domain (NLP/DL) | Project guidelines | Built entirely on an LLM (Gemini) + embeddings pipeline |
| 5% novelty | Project guidelines | Automated thematic gap-clustering + cross-paper contradiction detection + confidence-scored extraction (see "Novelty" below) |
| Comparative Analysis Table (15-20 studies, 5 columns) | Project guidelines | The app's core output IS this table, in the exact required schema |
| No raw code in report | Project guidelines | This tool produces a table/diagram-friendly output, not code dumps |

## Novelty statement (for your proposal/report)

> We propose an automated literature-synthesis pipeline that goes beyond
> per-paper summarization by performing cross-paper thematic gap
> clustering and contradiction detection, grounding every extracted claim
> with an LLM-reported confidence score to flag potential hallucination.
> This produces a reviewer-ready comparative analysis table together with
> evidence-grounded research-gap clusters, which is not addressed by
> standard single-document summarization or RAG-only literature tools.

Make sure to state this novelty in your own words in the proposal --
per the course's AI-tools policy, the report text itself must be written
by you; this tool is for analysis/drafting assistance, not report writing.

## Architecture

```
PDF uploads
    |
    v
pdf_extractor.py  --> per-paper full text (PyMuPDF)
    |
    v
chunking.py       --> overlapping text chunks
    |
    v
gemini_client.py  --> Gemini embeddings (gemini-embedding-001)
    |
    v
vector_store.py   --> ChromaDB (in-memory, per-session)
    |
    +--> extraction.py    --> per-paper structured fields + confidence (Gemini)
    |         |
    |         v
    +--> gap_analysis.py  --> cross-paper clusters/contradictions (Gemini)
    |         |
    |         v
    +--> pipeline.py      --> assembles the comparative table
    |
    +--> rag_chat.py      --> "Ask Your Papers" retrieval + grounded answers
    |
    v
report_export.py --> CSV / DOCX export
    |
    v
app.py (Streamlit) --> 4-tab GUI tying it all together
```

## Project structure

```
lit_review_assistant/
├── app.py                  # Streamlit GUI (entry point)
├── requirements.txt
├── .env.example             # optional: GEMINI_API_KEY for local runs
├── .streamlit/config.toml   # upload size limit, theme
├── src/
│   ├── config.py             # all tunables in one place
│   ├── gemini_client.py      # Gemini API wrapper (retry/backoff, JSON mode)
│   ├── pdf_extractor.py      # PDF -> clean text (PyMuPDF)
│   ├── chunking.py           # text -> overlapping chunks
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── extraction.py         # per-paper structured field extraction
│   ├── gap_analysis.py       # cross-paper gap clustering (novelty component)
│   ├── rag_chat.py           # "Ask Your Papers" RAG chat
│   ├── pipeline.py           # orchestrates all of the above
│   └── report_export.py      # CSV / DOCX export
└── data/                     # uploads, vectorstore, processed (gitignored)
```

## Setup

### 1. Get a free Gemini API key
Visit https://aistudio.google.com/apikey and create a key (no payment
required for the free tier used by this app).

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
That's it -- CSV and Word (.docx) export both work out of the box; the
Word export uses `python-docx`, so no Node.js or other external tooling
is required.

### 3. Run the app
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (typically `http://localhost:8501`),
paste your Gemini API key into the sidebar, and you're ready to upload PDFs.

## Usage

1. Go to **Tab 1: Upload & Process**, upload your PDFs (text-based PDFs
   only -- scanned/image-only PDFs without OCR will be reported as failed).
2. Click **Process Papers** and watch the live progress bar (this can take
   1-3 minutes for 15-20 papers due to free-tier rate limits).
3. Go to **Tab 2: Comparative Table** to review the generated table.
   Rows highlighted in yellow have confidence below 60 -- double-check
   these against the original PDF before using them in your report.
4. Export to CSV or Word from Tab 2.
5. Go to **Tab 3: Gap Analysis** to see recurring gap themes and any
   detected contradictions -- great source material for your report's
   "Contribution and Motivation" and "Discussion" sections.
6. Go to **Tab 4: Ask Your Papers** to ask free-form questions across
   your corpus, with source citations.

## Known limitations

- Free-tier Gemini rate limits mean processing many papers takes a few
  minutes; the progress bar keeps you informed.
- Scanned/image-only PDFs are not supported (no OCR step is included).
- The vector store is in-memory and per-session; re-running the app loses
  the index (re-process your PDFs after a restart).
- LLM-extracted fields can still be wrong even at high confidence scores --
  always verify against the source PDF before submitting your report,
  consistent with the course's academic-integrity policy.

