"""Corpus ingestion: walk markdown files, chunk them, embed them, and index into Chroma."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import httpx
import ollama

from k8s_docs_rag.chunking import chunk_document, load_document
from k8s_docs_rag.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_PERSIST_DIR = "chroma"
COLLECTION_NAME = "k8s_docs"
EMBED_BATCH_SIZE = 32


class IngestError(Exception):
    """Raised when the corpus could not be embedded/indexed."""


def iter_markdown_files(content_root: Path, subdirs: list[str]):
    for subdir in subdirs:
        base = content_root / subdir
        if not base.exists():
            raise FileNotFoundError(f"Corpus subdir not found: {base}")
        yield from sorted(base.rglob("*.md"))


def load_and_chunk_corpus(content_root: Path, subdirs: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md_path in iter_markdown_files(content_root, subdirs):
        doc = load_document(md_path, content_root)
        if not doc.body:
            continue
        chunks.extend(chunk_document(doc))
    return chunks


def _batched(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def embed_texts(client: ollama.Client, model: str, texts: list[str], host: str | None = None) -> list[list[float]]:
    embeddings: list[list[float]] = []
    try:
        for batch in _batched(texts, EMBED_BATCH_SIZE):
            response = client.embed(model=model, input=batch)
            embeddings.extend(response["embeddings"])
    except ConnectionError as exc:
        raise IngestError(
            f"Could not reach Ollama at {host or 'the default host'}. "
            "Make sure Ollama is installed and running: https://ollama.com"
        ) from exc
    except httpx.TimeoutException as exc:
        raise IngestError("Timed out waiting for Ollama while embedding the corpus.") from exc
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise IngestError(
                f"Embedding model '{model}' is not available locally. Pull it first with: ollama pull {model}"
            ) from exc
        raise IngestError(f"Ollama returned an error while embedding: {exc.error}") from exc
    return embeddings


def get_collection(persist_dir: str = DEFAULT_PERSIST_DIR):
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    return chroma_client.get_collection(COLLECTION_NAME)


def build_index(
    content_root: Path,
    subdirs: list[str],
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embed_model: str = DEFAULT_EMBED_MODEL,
    host: str | None = None,
) -> dict:
    chunks = load_and_chunk_corpus(content_root, subdirs)
    if not chunks:
        raise ValueError(f"No chunks produced from {content_root} subdirs={subdirs}")

    logger.info("Embedding %d chunks with model=%s", len(chunks), embed_model)
    client = ollama.Client(host=host)
    embeddings = embed_texts(client, embed_model, [c.text for c in chunks], host=host)

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME, metadata={"embed_model": embed_model})

    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"url": c.url, "title": c.title, "heading": c.heading, "doc_path": c.doc_path}
            for c in chunks
        ],
    )

    return {
        "num_documents": len({c.doc_path for c in chunks}),
        "num_chunks": len(chunks),
        "persist_dir": persist_dir,
        "embed_model": embed_model,
    }
