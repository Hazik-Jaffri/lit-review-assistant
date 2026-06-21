"""
gemini_client.py
-----------------
Thin wrapper around the Google Gemini API (`google-genai` SDK).

Why this module exists instead of calling the SDK directly everywhere:
    1. Centralizes API-key handling (env var, Streamlit secrets, or
       a key typed into the sidebar at runtime).
    2. Adds retry-with-backoff because the Gemini *free tier* has a low
       requests-per-minute limit and will return 429 errors under normal
       use (e.g. processing 15-20 papers back to back).
    3. Provides a single `generate_json()` helper that asks Gemini to
       return strict JSON and parses/repairs the response, so the rest
       of the codebase never deals with raw text-parsing of LLM output.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from . import config


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API fails after all retries are exhausted."""


@dataclass
class GeminiClient:
    """A small, stateful client. Construct once per Streamlit session."""

    api_key: str

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise GeminiAPIError("Gemini API key is empty. Please provide a valid key.")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise GeminiAPIError(
                "The 'google-genai' package is not installed. Run: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Low-level call with retry/backoff
    # ------------------------------------------------------------------
    def _call_with_retry(self, fn, *args, **kwargs):
        last_error: Exception | None = None
        for attempt in range(config.API_MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised as GeminiAPIError below
                last_error = exc
                message = str(exc).lower()
                transient = any(
                    token in message
                    for token in ("429", "resource_exhausted", "503", "deadline", "timeout", "overloaded")
                )
                if not transient or attempt == config.API_MAX_RETRIES - 1:
                    break
                wait = config.API_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(wait)
        raise GeminiAPIError(f"Gemini API call failed after retries: {last_error}") from last_error

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        thinking_budget: int = 0,
    ) -> str:
        """
        Plain text completion.

        thinking_budget=0 disables Gemini 2.5's "thinking" mode by default.
        IMPORTANT: gemini-2.5-flash has dynamic thinking ON by default, and
        thinking tokens are deducted from the SAME max_output_tokens budget
        as the visible answer. With the modest token budgets used in this
        app (1-3K), leaving thinking enabled can silently consume the whole
        budget and return an empty response. We don't need extended
        reasoning for these straightforward extraction/summarization
        prompts, so we disable it here; pass thinking_budget=-1 to re-enable
        dynamic thinking for a specific call if ever needed.
        """

        def _run():
            response = self._client.models.generate_content(
                model=config.GEMINI_TEXT_MODEL,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "thinking_config": {"thinking_budget": thinking_budget},
                },
            )
            text = getattr(response, "text", None)
            if not text:
                raise GeminiAPIError("Gemini returned an empty response.")
            return text

        return self._call_with_retry(_run)

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        thinking_budget: int = 0,
    ):
        """
        Ask Gemini for strict JSON output and parse it.
        Uses the SDK's `response_mime_type` to bias the model toward valid JSON,
        then falls back to regex extraction + json.loads if needed.

        thinking_budget=0 disables thinking by default -- see generate_text()
        docstring for why this matters with gemini-2.5-flash.
        """

        def _run():
            response = self._client.models.generate_content(
                model=config.GEMINI_TEXT_MODEL,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                    "thinking_config": {"thinking_budget": thinking_budget},
                },
            )
            text = getattr(response, "text", None)
            if not text:
                raise GeminiAPIError("Gemini returned an empty response.")
            return text

        raw_text = self._call_with_retry(_run)
        return _parse_json_relaxed(raw_text)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Used for query-time embedding in RAG search."""

        def _run():
            response = self._client.models.embed_content(
                model=config.GEMINI_EMBED_MODEL,
                contents=text,
            )
            return response.embeddings[0].values

        return self._call_with_retry(_run)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed many chunks. Gemini's embed_content accepts a list directly,
        but we batch conservatively (8 at a time) to stay well under the
        gemini-embedding-001 free tier limits, which are noticeably tighter
        than the old text-embedding-004 tier (~90 requests/min, ~27K
        tokens/min, 950 requests/day).
        """
        all_vectors: list[list[float]] = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            def _run(batch=batch):
                response = self._client.models.embed_content(
                    model=config.GEMINI_EMBED_MODEL,
                    contents=batch,
                )
                return [item.values for item in response.embeddings]

            all_vectors.extend(self._call_with_retry(_run))
        return all_vectors


def _parse_json_relaxed(raw_text: str):
    """
    Gemini with response_mime_type='application/json' almost always returns
    clean JSON, but occasionally wraps it in markdown fences or adds trailing
    commentary. This strips common wrappers before parsing.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise GeminiAPIError(f"Could not parse JSON from Gemini response: {exc}\nRaw: {text[:500]}") from exc
