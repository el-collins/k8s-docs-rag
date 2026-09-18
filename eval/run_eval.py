"""Runs the versioned eval set (`eval_set.yaml`) against a built index and writes a report.

Requires Ollama running with both the embedding and chat models pulled, and
an index already built via `k8s-rag ingest`. Not run in hosted CI for that
reason - see the README's Evaluation section.

Usage:
    python eval/run_eval.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from k8s_docs_rag.generate import DEFAULT_MODEL, GenerationError
from k8s_docs_rag.ingest import DEFAULT_EMBED_MODEL, DEFAULT_PERSIST_DIR
from k8s_docs_rag.pipeline import ask

EVAL_SET_PATH = Path(__file__).parent / "eval_set.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class CaseResult:
    id: str
    type: str
    question: str
    passed: bool
    detail: str
    abstained: bool
    fully_grounded: bool
    answer: str


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["cases"]


def evaluate_case(case: dict, **ask_kwargs) -> CaseResult:
    result = ask(case["question"], **ask_kwargs)

    if case["type"] == "abstain":
        passed = result.abstained
        detail = "correctly abstained" if passed else "did not abstain when it should have"
    else:
        source_contains = case["source_contains"]
        hit = any(source_contains in chunk.url for chunk in result.retrieved)
        passed = hit and result.fully_grounded and not result.abstained
        details = [
            "retrieval hit" if hit else f"retrieval miss (expected a source containing '{source_contains}')",
            "grounded" if result.fully_grounded else f"ungrounded citations: {sorted(result.ungrounded_ids)}",
        ]
        if result.abstained:
            details.append("unexpectedly abstained")
        detail = "; ".join(details)

    return CaseResult(
        id=case["id"],
        type=case["type"],
        question=case["question"],
        passed=passed,
        detail=detail,
        abstained=result.abstained,
        fully_grounded=result.fully_grounded,
        answer=result.answer,
    )


def run(**ask_kwargs) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in load_eval_set():
        print(f"[{case['id']}] {case['question']}")
        try:
            case_result = evaluate_case(case, **ask_kwargs)
        except GenerationError as exc:
            case_result = CaseResult(
                id=case["id"],
                type=case["type"],
                question=case["question"],
                passed=False,
                detail=f"error: {exc}",
                abstained=False,
                fully_grounded=False,
                answer="",
            )
        print(f"  {'PASS' if case_result.passed else 'FAIL'} - {case_result.detail}")
        results.append(case_result)
    return results


def summarize(results: list[CaseResult]) -> dict:
    total = len(results)
    answer_cases = [r for r in results if r.type == "answer"]
    abstain_cases = [r for r in results if r.type == "abstain"]
    return {
        "total": total,
        "passed": sum(r.passed for r in results),
        "pass_rate": sum(r.passed for r in results) / total if total else 0.0,
        "answer_cases": len(answer_cases),
        "answer_pass_rate": (sum(r.passed for r in answer_cases) / len(answer_cases)) if answer_cases else None,
        "abstain_cases": len(abstain_cases),
        "abstain_pass_rate": (sum(r.passed for r in abstain_cases) / len(abstain_cases)) if abstain_cases else None,
    }


def write_report(results: list[CaseResult], summary: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = RESULTS_DIR / f"report_{timestamp}.json"
    report_path.write_text(
        json.dumps({"summary": summary, "cases": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the k8s-docs-rag evaluation set against a built index.")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=None)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    results = run(
        persist_dir=args.persist_dir,
        embed_model=args.embed_model,
        chat_model=args.model,
        host=args.host,
        k=args.top_k,
    )
    summary = summarize(results)
    report_path = write_report(results, summary)

    print()
    print(f"Passed {summary['passed']}/{summary['total']} ({summary['pass_rate']:.0%})")
    if summary["answer_pass_rate"] is not None:
        print(f"  answer cases:   {summary['answer_pass_rate']:.0%} ({summary['answer_cases']} cases)")
    if summary["abstain_pass_rate"] is not None:
        print(f"  abstain cases:  {summary['abstain_pass_rate']:.0%} ({summary['abstain_cases']} cases)")
    print(f"Report written to {report_path}")

    return 0 if summary["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
