from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest

from k8s_docs_rag.generate import (
    ABSTAIN_PHRASE,
    GenerationError,
    answer_question,
    build_generation_prompt,
    extract_cited_ids,
    validate_citations,
)
from k8s_docs_rag.models import RetrievedChunk

CHUNK_A = RetrievedChunk(chunk_id="a", url="https://kubernetes.io/docs/a/", title="A", heading="A", text="Pods are the smallest deployable unit.")
CHUNK_B = RetrievedChunk(chunk_id="b", url="https://kubernetes.io/docs/b/", title="B", heading="B", text="A Deployment manages Pods via a ReplicaSet.")


def test_build_generation_prompt_wraps_each_chunk_with_id_and_source():
    prompt = build_generation_prompt("what is a pod?", [CHUNK_A, CHUNK_B])
    assert '<doc id="1" source="https://kubernetes.io/docs/a/">' in prompt
    assert '<doc id="2" source="https://kubernetes.io/docs/b/">' in prompt
    assert "Question: what is a pod?" in prompt


def test_extract_cited_ids():
    assert extract_cited_ids("Pods are small [doc 1] and managed by Deployments [doc 2].") == {1, 2}
    assert extract_cited_ids("No citations here.") == set()


def test_validate_citations_flags_out_of_range_ids():
    cited, ungrounded = validate_citations("See [doc 1] and [doc 5].", num_chunks=2)
    assert cited == {1, 5}
    assert ungrounded == {5}


def test_answer_question_with_no_chunks_abstains_without_calling_ollama():
    result = answer_question("off-topic question", [])
    assert result.abstained
    assert result.answer == ABSTAIN_PHRASE


@patch("ollama.Client")
def test_answer_question_returns_grounded_result(mock_client_cls):
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "A Pod is the smallest deployable unit. [doc 1]"}}
    mock_client_cls.return_value = mock_client

    result = answer_question("what is a pod?", [CHUNK_A])

    assert not result.abstained
    assert result.fully_grounded
    assert result.cited_ids == {1}


@patch("ollama.Client")
def test_answer_question_flags_ungrounded_citation(mock_client_cls):
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "Answer with a made-up source [doc 9]."}}
    mock_client_cls.return_value = mock_client

    result = answer_question("what is a pod?", [CHUNK_A])

    assert not result.fully_grounded
    assert result.ungrounded_ids == {9}


@patch("ollama.Client")
def test_answer_question_maps_connection_error(mock_client_cls):
    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionError("boom")
    mock_client_cls.return_value = mock_client

    with pytest.raises(GenerationError, match="Ollama"):
        answer_question("what is a pod?", [CHUNK_A])


@patch("ollama.Client")
def test_answer_question_maps_timeout(mock_client_cls):
    mock_client = MagicMock()
    mock_client.chat.side_effect = httpx.ReadTimeout("boom")
    mock_client_cls.return_value = mock_client

    with pytest.raises(GenerationError, match="Timed out"):
        answer_question("what is a pod?", [CHUNK_A], timeout=1.0)


@patch("ollama.Client")
def test_answer_question_maps_model_not_found(mock_client_cls):
    mock_client = MagicMock()
    mock_client.chat.side_effect = ollama.ResponseError("not found", status_code=404)
    mock_client_cls.return_value = mock_client

    with pytest.raises(GenerationError, match="ollama pull"):
        answer_question("what is a pod?", [CHUNK_A], model="llama3.2")
