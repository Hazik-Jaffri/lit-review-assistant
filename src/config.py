"""
config.py
---------
Central configuration for the Literature Review & Gap Analysis Assistant.
Keeping every tunable value here means no magic numbers/strings scattered
across the pipeline, GUI, and RAG modules.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

for _dir in (UPLOAD_DIR, VECTORSTORE_DIR, PROCESSED_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_TITLE = "Automated Literature Review & Gap Analysis Assistant"
APP_ICON = "books"

# ---------------------------------------------------------------------------
# Gemini model configuration
# ---------------------------------------------------------------------------
# text generation model (extraction, gap analysis, RAG answers)
# NOTE: gemini-2.0-flash was shut down by Google on June 1, 2026.
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
# embedding model (semantic search / vector store)
# NOTE: text-embedding-004 was shut down by Google on January 14, 2026.
# gemini-embedding-001 is the current GA replacement, but its free tier is
# more rate-limited (90 requests/min, ~27K tokens/min, 950 requests/day),
# which is why embed_batch() below uses a conservative batch size.
GEMINI_EMBED_MODEL = "gemini-embedding-001"

# Allow override via environment variable so the same code works for
# local runs (.env) and Streamlit Cloud (st.secrets), see gemini_client.py
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

# ---------------------------------------------------------------------------
# Chunking parameters (for embeddings / RAG)
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS = 1800          # ~ 400-450 tokens per chunk
CHUNK_OVERLAP_CHARS = 250
MAX_CHARS_PER_PAPER_FOR_EXTRACTION = 28000  # cap full-text sent to LLM in one call

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
TOP_K_CHUNKS = 5

# ---------------------------------------------------------------------------
# Comparative table columns (mirrors the course's required table schema)
# ---------------------------------------------------------------------------
TABLE_COLUMNS = [
    "Reference",
    "Title",
    "Proposed Solution",
    "Limitations / Gaps",
    "Our Contribution",
    "Confidence",
]

# ---------------------------------------------------------------------------
# Rate-limit friendly retry settings (Gemini free tier has low RPM limits)
# ---------------------------------------------------------------------------
API_MAX_RETRIES = 4
API_RETRY_BACKOFF_SECONDS = 8  # exponential: 8, 16, 32, 64...
