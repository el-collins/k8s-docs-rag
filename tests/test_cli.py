from pathlib import Path
from unittest.mock import patch

from k8s_docs_rag.cli import main
from k8s_docs_rag.generate import GenerationError
from k8s_docs_rag.ingest import IngestError
from k8s_docs_rag.models import AnswerResult


def test_ingest_command_reports_stats(capsys):
    fake_stats = {"num_chunks": 42, "num_documents": 7, "persist_dir": "chroma"}
    with patch("k8s_docs_rag.cli.build_index", return_value=fake_stats) as mock_build:
        exit_code = main(["ingest"])

    assert exit_code == 0
    assert "Indexed 42 chunks from 7 documents" in capsys.readouterr().out
    mock_build.assert_called_once()


def test_ingest_command_reports_clean_error_on_failure(capsys):
    with patch("k8s_docs_rag.cli.build_index", side_effect=IngestError("Ollama is not running")):
        exit_code = main(["ingest"])

    assert exit_code == 1
    assert "Error: Ollama is not running" in capsys.readouterr().err


def test_ask_command_prints_answer(capsys):
    fake_result = AnswerResult(question="q", answer="A Pod is the smallest unit. [doc 1]", abstained=False)
    with patch("k8s_docs_rag.cli.run_ask", return_value=fake_result):
        exit_code = main(["ask", "what is a pod?"])

    assert exit_code == 0
    assert "[doc 1]" in capsys.readouterr().out


def test_ask_command_warns_on_ungrounded_citation(capsys):
    fake_result = AnswerResult(question="q", answer="Answer [doc 9].", abstained=False, ungrounded_ids={9})
    with patch("k8s_docs_rag.cli.run_ask", return_value=fake_result):
        exit_code = main(["ask", "what is a pod?"])

    assert exit_code == 0
    assert "citations reference docs that were not retrieved" in capsys.readouterr().out


def test_ask_command_writes_to_output_file(tmp_path: Path):
    output_path = tmp_path / "answer.txt"
    fake_result = AnswerResult(question="q", answer="A Pod is the smallest unit.", abstained=False)
    with patch("k8s_docs_rag.cli.run_ask", return_value=fake_result):
        exit_code = main(["ask", "what is a pod?", "-o", str(output_path)])

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "A Pod is the smallest unit."


def test_ask_command_rejects_empty_question(capsys):
    exit_code = main(["ask", "   "])
    assert exit_code == 1
    assert "must not be empty" in capsys.readouterr().err


def test_ask_command_prints_clean_error_on_generation_failure(capsys):
    with patch("k8s_docs_rag.cli.run_ask", side_effect=GenerationError("No index found")):
        exit_code = main(["ask", "what is a pod?"])

    assert exit_code == 1
    assert "Error: No index found" in capsys.readouterr().err
