from pathlib import Path

from k8s_docs_rag.chunking import (
    chunk_document,
    clean_body,
    derive_url,
    load_document,
    parse_front_matter,
)

SAMPLE_MD = """---
title: Pods
weight: 10
---

<!-- overview -->

This page explains Pods.

{{< figure src="/images/pod.svg" alt="Pod" >}}

<!-- body -->

## What is a Pod?

A Pod is the smallest deployable unit of computing in Kubernetes.

## Pod lifecycle

A Pod's status field is a PodStatus object.
"""


def test_parse_front_matter_extracts_metadata():
    metadata, body = parse_front_matter(SAMPLE_MD)
    assert metadata["title"] == "Pods"
    assert metadata["weight"] == 10
    assert "This page explains Pods." in body


def test_parse_front_matter_handles_missing_front_matter():
    metadata, body = parse_front_matter("# Just a heading\n\ntext")
    assert metadata == {}
    assert body == "# Just a heading\n\ntext"


def test_clean_body_strips_comments_and_shortcodes():
    _, body = parse_front_matter(SAMPLE_MD)
    cleaned = clean_body(body)
    assert "<!--" not in cleaned
    assert "{{<" not in cleaned
    assert "This page explains Pods." in cleaned
    assert "## What is a Pod?" in cleaned


def test_derive_url_regular_page():
    assert derive_url("docs/concepts/overview/components.md") == "https://kubernetes.io/docs/concepts/overview/components/"


def test_derive_url_index_page():
    assert derive_url("docs/concepts/overview/_index.md") == "https://kubernetes.io/docs/concepts/overview/"


def test_load_document_and_chunk(tmp_path: Path):
    content_root = tmp_path / "content" / "en"
    doc_dir = content_root / "docs" / "concepts"
    doc_dir.mkdir(parents=True)
    md_path = doc_dir / "pods.md"
    md_path.write_text(SAMPLE_MD, encoding="utf-8")

    doc = load_document(md_path, content_root)
    assert doc.title == "Pods"
    assert doc.url == "https://kubernetes.io/docs/concepts/pods/"

    chunks = chunk_document(doc)
    headings = [c.heading for c in chunks]
    assert "What is a Pod?" in headings
    assert "Pod lifecycle" in headings
    for chunk in chunks:
        assert chunk.url == doc.url
        assert chunk.doc_path == "docs/concepts/pods.md"


def test_chunk_document_splits_long_sections():
    from k8s_docs_rag.models import Document

    long_text = " ".join(f"word{i}" for i in range(1000))
    doc = Document(path="docs/x.md", url="https://kubernetes.io/docs/x/", title="X", body=f"## Section\n\n{long_text}")

    chunks = chunk_document(doc, max_words=200, overlap_words=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text.split()) <= 200
    # consecutive chunks overlap
    first_tail = chunks[0].text.split()[-10:]
    second_words = chunks[1].text.split()
    assert any(word in second_words[:60] for word in first_tail)
