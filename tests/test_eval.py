import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_eval  # noqa: E402

from k8s_docs_rag.models import AnswerResult, RetrievedChunk  # noqa: E402

MATCHING_CHUNK = RetrievedChunk(chunk_id="a", url="https://kubernetes.io/docs/concepts/workloads/pods/", title="Pods", heading="Pods", text="...")
OTHER_CHUNK = RetrievedChunk(chunk_id="b", url="https://kubernetes.io/docs/concepts/services-networking/service/", title="Service", heading="Service", text="...")

ANSWER_CASE = {"id": "what-is-a-pod", "type": "answer", "question": "What is a Pod?", "source_contains": "/docs/concepts/workloads/pods/"}
ABSTAIN_CASE = {"id": "off-topic", "type": "abstain", "question": "How do I file my taxes?"}


def test_evaluate_case_passes_on_grounded_hit():
    fake_result = AnswerResult(question=ANSWER_CASE["question"], answer="A Pod is... [doc 1]", abstained=False, retrieved=[MATCHING_CHUNK])
    with patch("run_eval.ask", return_value=fake_result):
        result = run_eval.evaluate_case(ANSWER_CASE)

    assert result.passed
    assert "retrieval hit" in result.detail
    assert "grounded" in result.detail


def test_evaluate_case_fails_on_retrieval_miss():
    fake_result = AnswerResult(question=ANSWER_CASE["question"], answer="Something unrelated. [doc 1]", abstained=False, retrieved=[OTHER_CHUNK])
    with patch("run_eval.ask", return_value=fake_result):
        result = run_eval.evaluate_case(ANSWER_CASE)

    assert not result.passed
    assert "retrieval miss" in result.detail


def test_evaluate_case_fails_on_ungrounded_citation():
    fake_result = AnswerResult(
        question=ANSWER_CASE["question"],
        answer="A Pod is... [doc 9]",
        abstained=False,
        retrieved=[MATCHING_CHUNK],
        ungrounded_ids={9},
    )
    with patch("run_eval.ask", return_value=fake_result):
        result = run_eval.evaluate_case(ANSWER_CASE)

    assert not result.passed
    assert "ungrounded citations" in result.detail


def test_evaluate_case_abstain_passes_when_model_abstains():
    fake_result = AnswerResult(question=ABSTAIN_CASE["question"], answer="I don't have enough information in the provided documentation to answer that.", abstained=True)
    with patch("run_eval.ask", return_value=fake_result):
        result = run_eval.evaluate_case(ABSTAIN_CASE)

    assert result.passed


def test_evaluate_case_abstain_fails_when_model_answers_anyway():
    fake_result = AnswerResult(question=ABSTAIN_CASE["question"], answer="You should use TurboTax.", abstained=False)
    with patch("run_eval.ask", return_value=fake_result):
        result = run_eval.evaluate_case(ABSTAIN_CASE)

    assert not result.passed


def test_summarize_computes_pass_rates():
    results = [
        run_eval.CaseResult(id="a", type="answer", question="q", passed=True, detail="", abstained=False, fully_grounded=True, answer=""),
        run_eval.CaseResult(id="b", type="answer", question="q", passed=False, detail="", abstained=False, fully_grounded=False, answer=""),
        run_eval.CaseResult(id="c", type="abstain", question="q", passed=True, detail="", abstained=True, fully_grounded=True, answer=""),
    ]

    summary = run_eval.summarize(results)

    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["answer_pass_rate"] == 0.5
    assert summary["abstain_pass_rate"] == 1.0
