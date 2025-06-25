"""Command-line interface for k8s-docs-rag: `ingest` builds the index, `ask` queries it."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from k8s_docs_rag.generate import DEFAULT_MODEL, DEFAULT_TIMEOUT, GenerationError
from k8s_docs_rag.ingest import DEFAULT_EMBED_MODEL, DEFAULT_PERSIST_DIR, IngestError, build_index
from k8s_docs_rag.pipeline import ask as run_ask

logger = logging.getLogger("k8s_docs_rag")

DEFAULT_CONTENT_ROOT = Path("website/content/en")
DEFAULT_SUBDIRS = ["docs/concepts", "docs/tasks"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="k8s-rag",
        description="Answer Kubernetes questions from a local docs corpus, with grounded citations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build (or rebuild) the local index from the fetched docs corpus.")
    ingest_parser.add_argument("--content-root", default=DEFAULT_CONTENT_ROOT, type=Path, help=f"Path to the cloned website's content/en directory (default: {DEFAULT_CONTENT_ROOT})")
    ingest_parser.add_argument("--subdir", dest="subdirs", action="append", help="Corpus subdirectory to index, relative to --content-root (repeatable). Default: docs/concepts, docs/tasks")
    ingest_parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Where to persist the Chroma index (default: %(default)s)")
    ingest_parser.add_argument("--embed-model", default=os.environ.get("K8S_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    ingest_parser.add_argument("--host", default=None, help="Ollama host URL (default: $OLLAMA_HOST or http://localhost:11434)")
    ingest_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the built index.")
    ask_parser.add_argument("question", nargs="?", help="Question to ask. If omitted, you'll be prompted interactively.")
    ask_parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    ask_parser.add_argument("--embed-model", default=os.environ.get("K8S_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    ask_parser.add_argument("--model", default=os.environ.get("K8S_RAG_MODEL", DEFAULT_MODEL))
    ask_parser.add_argument("--host", default=None)
    ask_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ask_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve (default: %(default)s)")
    ask_parser.add_argument("-o", "--output", type=Path, default=None)
    ask_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args(argv)


def _run_ingest(args: argparse.Namespace) -> int:
    subdirs = args.subdirs or DEFAULT_SUBDIRS
    try:
        stats = build_index(
            args.content_root,
            subdirs,
            persist_dir=args.persist_dir,
            embed_model=args.embed_model,
            host=args.host,
        )
    except (FileNotFoundError, ValueError, IngestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Indexed {stats['num_chunks']} chunks from {stats['num_documents']} documents into '{stats['persist_dir']}'.")
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    try:
        question = args.question if args.question is not None else input("Ask a question: ")
        question = question.strip()
        if not question:
            print("Error: question must not be empty.", file=sys.stderr)
            return 1

        result = run_ask(
            question,
            persist_dir=args.persist_dir,
            embed_model=args.embed_model,
            chat_model=args.model,
            host=args.host,
            timeout=args.timeout,
            k=args.top_k,
        )
    except GenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130

    lines = [result.answer]
    if result.ungrounded_ids:
        lines.append(f"\n[warning: citations reference docs that were not retrieved: {sorted(result.ungrounded_ids)}]")
    output_text = "\n".join(lines)

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Answer written to {args.output}")
    else:
        print(output_text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(message)s")

    if args.command == "ingest":
        return _run_ingest(args)
    if args.command == "ask":
        return _run_ask(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
