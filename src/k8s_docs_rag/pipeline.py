"""Ties retrieval and generation together into the single `ask` entry point used by the CLI."""

from __future__ import annotations

import chromadb.errors
import ollama

from k8s_docs_rag.generate import DEFAULT_MODEL, DEFAULT_TIMEOUT, GenerationError, answer_question
from k8s_docs_rag.ingest import DEFAULT_EMBED_MODEL, DEFAULT_PERSIST_DIR, get_collection
from k8s_docs_rag.models import AnswerResult
from k8s_docs_rag.retrieve import HybridIndex, embed_query


def ask(
    question: str,
    *,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chat_model: str = DEFAULT_MODEL,
    host: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    k: int = 5,
) -> AnswerResult:
    try:
        collection = get_collection(persist_dir)
    except chromadb.errors.NotFoundError as exc:
        raise GenerationError(
            f"No index found at '{persist_dir}'. Build one first with: k8s-rag ingest"
        ) from exc

    index = HybridIndex(collection)
    embed_client = ollama.Client(host=host, timeout=timeout)

    try:
        query_embedding = embed_query(embed_client, embed_model, question)
    except ConnectionError as exc:
        raise GenerationError(
            f"Could not reach Ollama at {host or 'the default host'} to embed the question. "
            "Make sure Ollama is installed and running: https://ollama.com"
        ) from exc
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise GenerationError(
                f"Embedding model '{embed_model}' is not available locally. "
                f"Pull it first with: ollama pull {embed_model}"
            ) from exc
        raise GenerationError(f"Ollama returned an error while embedding: {exc.error}") from exc

    chunks = index.hybrid_search(question, query_embedding, k=k)
    return answer_question(question, chunks, model=chat_model, host=host, timeout=timeout)
