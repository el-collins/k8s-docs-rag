import uuid
from unittest.mock import MagicMock, patch

import chromadb

from k8s_docs_rag.retrieve import HybridIndex, embed_query, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_favors_items_ranked_high_in_both_lists():
    dense = ["a", "b", "c"]
    sparse = ["c", "a", "b"]

    fused = reciprocal_rank_fusion([dense, sparse])
    fused_ids = [item_id for item_id, _ in fused]

    # "a" is #1 dense / #2 sparse, "c" is #3 dense / #1 sparse -> "a" should win narrowly
    assert fused_ids[0] == "a"
    assert set(fused_ids) == {"a", "b", "c"}


def test_reciprocal_rank_fusion_includes_items_present_in_only_one_list():
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert {item_id for item_id, _ in fused} == {"a", "b", "c"}


def _make_collection():
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


def test_hybrid_index_dense_search_returns_nearest_by_embedding():
    index = HybridIndex(_make_collection())
    ids = index.dense_search([1.0, 0.0], k=1)
    assert ids == ["pods::intro::0"]


def test_hybrid_index_sparse_search_matches_keyword():
    index = HybridIndex(_make_collection())
    ids = index.sparse_search("deployment replicaset", k=2)
    assert "deploy::intro::0" in ids


def test_hybrid_search_returns_retrieved_chunks_with_metadata():
    index = HybridIndex(_make_collection())
    results = index.hybrid_search("what is a pod", query_embedding=[1.0, 0.0], k=2)

    assert len(results) == 2
    top = results[0]
    assert top.url.startswith("https://kubernetes.io/")
    assert top.chunk_id in {"pods::intro::0", "deploy::intro::0"}


@patch("ollama.Client")
def test_embed_query_calls_client_with_model_and_input(mock_client_cls):
    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_client_cls.return_value = mock_client

    client = mock_client_cls()
    result = embed_query(client, "nomic-embed-text", "what is a pod?")

    assert result == [0.1, 0.2, 0.3]
    mock_client.embed.assert_called_once_with(model="nomic-embed-text", input="what is a pod?")
