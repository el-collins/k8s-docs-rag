"""Hybrid retrieval: dense (Chroma) similarity fused with sparse (BM25) keyword search.

Fusing the two matters in practice: dense embeddings are good at semantic
similarity but routinely miss queries built around an exact token (a flag
name, a resource kind like `PodDisruptionBudget`) that BM25 catches
directly.
"""

from __future__ import annotations

import re

import ollama
from rank_bm25 import BM25Okapi

from k8s_docs_rag.models import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridIndex:
    """Wraps a Chroma collection with an in-memory BM25 index built from its documents.

    BM25 is rebuilt from `collection.get()` rather than persisted separately,
    so Chroma stays the single source of truth for indexed content.
    """

    def __init__(self, collection):
        self.collection = collection
        data = collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.documents: list[str] = data["documents"]
        self.metadatas: list[dict] = data["metadatas"]
        self._id_to_index = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        self._bm25 = BM25Okapi([_tokenize(doc) for doc in self.documents])

    def _chunk_at(self, index: int) -> RetrievedChunk:
        meta = self.metadatas[index]
        return RetrievedChunk(
            chunk_id=self.ids[index],
            url=meta["url"],
            title=meta["title"],
            heading=meta["heading"],
            text=self.documents[index],
        )

    def dense_search(self, query_embedding: list[float], k: int) -> list[str]:
        result = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return result["ids"][0]

    def sparse_search(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:k]]

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 5,
        dense_k: int = 20,
        sparse_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        dense_ids = self.dense_search(query_embedding, dense_k)
        sparse_ids = self.sparse_search(query, sparse_k)
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], rrf_k=rrf_k)

        dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids)}
        sparse_rank = {chunk_id: rank for rank, chunk_id in enumerate(sparse_ids)}

        results: list[RetrievedChunk] = []
        for chunk_id, score in fused[:k]:
            chunk = self._chunk_at(self._id_to_index[chunk_id])
            chunk.dense_rank = dense_rank.get(chunk_id)
            chunk.sparse_rank = sparse_rank.get(chunk_id)
            chunk.fused_score = score
            results.append(chunk)
        return results


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Combine multiple ranked id lists into one, via reciprocal rank fusion.

    score(id) = sum over rankings containing id of 1 / (rrf_k + rank)
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def embed_query(client: ollama.Client, model: str, query: str) -> list[float]:
    response = client.embed(model=model, input=query)
    return response["embeddings"][0]
