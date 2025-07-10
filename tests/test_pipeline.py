import uuid
from unittest.mock import MagicMock, patch

import chromadb
import chromadb.errors
import pytest

from k8s_docs_rag.generate import GenerationError
from k8s_docs_rag.pipeline import ask


def _fake_collection():
    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"k8s_docs_{uuid.uuid4().hex}")
    collection.add(
        ids=["pods::intro::0", "deploy::intro::0"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=[
            "A Pod is the smallest deployable unit in Kubernetes.",
            "A Deployment manages a ReplicaSet of Pods.",
        ],
        metadatas=[
            {"url": "https://kubernetes.io/docs/concepts/pods/", "title": "Pods", "heading": "Intro", "doc_path": "docs/pods.md"},
            {"url": "https://kubernetes.io/docs/concepts/deployment/", "title": "Deployment", "heading": "Intro", "doc_path": "docs/deployment.md"},
        ],
    )
    return collection


@patch("ollama.Client")
@patch("k8s_docs_rag.pipeline.get_collection")
def test_ask_returns_grounded_answer(mock_get_collection, mock_client_cls):
    mock_get_collection.return_value = _fake_collection()

    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[1.0, 0.0]]}
    mock_client.chat.return_value = {"message": {"content": "A Pod is the smallest deployable unit. [doc 1]"}}
    mock_client_cls.return_value = mock_client

    result = ask("what is a pod?", k=2)

    assert "[doc 1]" in result.answer
    assert result.fully_grounded
    assert len(result.retrieved) == 2


@patch("k8s_docs_rag.pipeline.get_collection")
def test_ask_raises_clear_error_when_index_missing(mock_get_collection):
    mock_get_collection.side_effect = chromadb.errors.NotFoundError("missing")

    with pytest.raises(GenerationError, match="k8s-rag ingest"):
        ask("what is a pod?")


@patch("ollama.Client")
@patch("k8s_docs_rag.pipeline.get_collection")
def test_ask_maps_embedding_model_not_found(mock_get_collection, mock_client_cls):
    import ollama

    mock_get_collection.return_value = _fake_collection()
    mock_client = MagicMock()
    mock_client.embed.side_effect = ollama.ResponseError("not found", status_code=404)
    mock_client_cls.return_value = mock_client

    with pytest.raises(GenerationError, match="ollama pull"):
        ask("what is a pod?")
