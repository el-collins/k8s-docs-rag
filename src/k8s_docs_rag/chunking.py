"""Markdown loading, cleanup, and header-aware chunking.

Chunk sizing is word-count based rather than tokenizer-based (a lightweight
proxy: ~0.75 tokens per English word), to avoid pulling in a tokenizer
dependency for what is a coarse retrieval-chunk boundary, not a model
context-limit calculation.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from k8s_docs_rag.models import Chunk, Document

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
SHORTCODE_RE = re.compile(r"\{\{[%<]/?[^}]*?[%>]\}\}")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADER_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)

TARGET_CHUNK_WORDS = 550
OVERLAP_WORDS = 80


def parse_front_matter(text: str) -> tuple[dict, str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}
    body = text[match.end() :]
    return metadata, body


def clean_body(body: str) -> str:
    body = COMMENT_RE.sub("", body)
    body = SHORTCODE_RE.sub("", body)
    lines = [line.rstrip() for line in body.splitlines()]
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def derive_url(relative_path: str) -> str:
    """Map a path relative to `content/en/` to its canonical kubernetes.io URL."""
    trimmed = relative_path[:-3] if relative_path.endswith(".md") else relative_path
    if trimmed.endswith("/_index"):
        trimmed = trimmed[: -len("/_index")]
    elif trimmed == "_index":
        trimmed = ""
    return f"https://kubernetes.io/{trimmed}/".replace("//", "/").replace("https:/", "https://")


def load_document(md_path: Path, content_root: Path) -> Document:
    raw = md_path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw)
    relative_path = md_path.relative_to(content_root).as_posix()
    title = metadata.get("title") or md_path.stem.replace("-", " ").title()
    return Document(
        path=relative_path,
        url=derive_url(relative_path),
        title=str(title),
        body=clean_body(body),
    )


def _split_by_headers(body: str) -> list[tuple[str, str]]:
    """Split into (heading, section_text) pairs. Text before the first header uses heading=""."""
    matches = list(HEADER_RE.finditer(body))
    if not matches:
        return [("", body.strip())] if body.strip() else []

    sections: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        if section_text:
            sections.append((heading, section_text))
    return sections


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _split_long_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    parts: list[str] = []
    start = 0
    step = max(max_words - overlap_words, 1)
    while start < len(words):
        end = min(start + max_words, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return parts


def chunk_document(
    doc: Document,
    max_words: int = TARGET_CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_index, (heading, section_text) in enumerate(_split_by_headers(doc.body)):
        parts = _split_long_text(section_text, max_words, overlap_words)
        for part_index, part_text in enumerate(parts):
            # section_index disambiguates sections whose headings slugify to
            # the same text (e.g. two "Note" or "Example" subsections).
            chunk_id = f"{doc.path}::{section_index}-{_slugify(heading or doc.title)}::{part_index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_path=doc.path,
                    url=doc.url,
                    title=doc.title,
                    heading=heading,
                    text=part_text,
                )
            )
    return chunks
