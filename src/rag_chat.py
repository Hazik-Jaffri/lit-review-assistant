"""
rag_chat.py
------------
"Ask Your Papers" - a retrieval-augmented chat over the uploaded corpus.

Flow: embed the user's question -> retrieve top-k similar chunks from the
vector store -> stuff them into a grounded prompt -> ask Gemini to answer
using ONLY the retrieved context, with citations back to source file names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config

RAG_PROMPT_TEMPLATE = """You are a research assistant answering a question using ONLY the excerpts
below, which come from the user's uploaded papers. If the excerpts do not
contain enough information to answer, say so clearly instead of guessing.

When you state a claim, mention which paper it came from (by file name).

EXCERPTS:
{context_block}

QUESTION: {question}

Answer concisely (3-6 sentences unless the question needs more detail).
"""


@dataclass
class RagAnswer:
    answer: str
    sources: list = field(default_factory=list)
    error: object = None


def _format_context(retrieved_chunks):
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        blocks.append(f"[{i}] Source: {chunk['paper_file_name']}\n{chunk['text']}")
    return "\n\n".join(blocks)


def answer_question(client, vector_store, question, top_k=config.TOP_K_CHUNKS, paper_filter=None):
    """
    Embed the question, retrieve relevant chunks, and generate a grounded
    answer. Returns sources so the UI can show "based on: paper_x.pdf, paper_y.pdf".
    """
    try:
        query_embedding = client.embed_text(question)
        retrieved = vector_store.query(query_embedding, top_k=top_k, paper_filter=paper_filter)

        if not retrieved:
            return RagAnswer(
                answer="I couldn't find any relevant content in the uploaded papers for this question.",
                sources=[],
            )

        context_block = _format_context(retrieved)
        prompt = RAG_PROMPT_TEMPLATE.format(context_block=context_block, question=question)
        answer_text = client.generate_text(prompt, temperature=0.2, max_output_tokens=1024)

        sources = [
            {"file_name": r["paper_file_name"], "snippet": r["text"][:220]}
            for r in retrieved
        ]
        return RagAnswer(answer=answer_text, sources=sources)
    except Exception as exc:  # noqa: BLE001 - surfaced to UI
        return RagAnswer(answer="", sources=[], error=str(exc))
