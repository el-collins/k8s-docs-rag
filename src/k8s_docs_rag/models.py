"""Shared dataclasses passed between the chunking, retrieval, and generation stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A single source markdown file after front-matter/shortcode cleanup."""

    path: str  # path relative to the corpus root, e.g. "docs/concepts/overview/components.md"
    url: str  # canonical kubernetes.io URL derived from `path`
    title: str
    body: str


@dataclass
class Chunk:
    """A retrievable unit of text: one heading section, or a slice of a long one."""

    id: str  # stable id: f"{doc.path}::{heading_slug}::{part_index}"
    doc_path: str
    url: str
    title: str
    heading: str
    text: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    url: str
    title: str
    heading: str
    text: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float = 0.0


@dataclass
class AnswerResult:
    question: str
    answer: str
    abstained: bool
    cited_ids: set[int] = field(default_factory=set)
    ungrounded_ids: set[int] = field(default_factory=set)
    retrieved: list[RetrievedChunk] = field(default_factory=list)

    @property
    def fully_grounded(self) -> bool:
        return not self.ungrounded_ids
