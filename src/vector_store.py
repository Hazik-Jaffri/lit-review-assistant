"""
vector_store.py
----------------
Wraps ChromaDB for storing/retrieving chunk embeddings.

We compute embeddings ourselves via GeminiClient.embed_batch and pass them
into Chroma explicitly (instead of using Chroma's built-in embedding
function), so the same embeddings can be reused for both the gap-analysis
step and the "Ask Your Papers" RAG chat without re-calling the API.

Each Streamlit session gets a fresh, in-memory (ephemeral) Chroma client,
so multiple users on a shared deployment don't see each other's uploaded
papers, and re-processing always starts from a clean collection.
"""

from __future__ import annotations

from . import config


class VectorStore:
    def __init__(self, collection_name="papers"):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("chromadb is not installed. Run: pip install chromadb") from exc

        self._client = chromadb.EphemeralClient()
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(name=collection_name)

    def add_chunks(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"paper_file_name": c.paper_file_name, "chunk_index": c.chunk_index} for c in chunks],
        )

    def query(self, query_embedding, top_k=config.TOP_K_CHUNKS, paper_filter=None):
        """Return the top_k most similar chunks. Optionally restrict to one paper."""
        where = {"paper_file_name": paper_filter} if paper_filter else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {
                "text": doc,
                "paper_file_name": meta.get("paper_file_name"),
                "chunk_index": meta.get("chunk_index"),
                "distance": dist,
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def count(self):
        return self._collection.count()
