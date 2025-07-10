# k8s-docs-rag

A retrieval-augmented question-answering system over the official Kubernetes
documentation, running entirely on a local LLM via [Ollama](https://ollama.com) —
no cloud API key, no per-token cost.

It's built to demonstrate the parts of a RAG system that actually matter in
production, not just "stuff docs into a vector store and hope":

- **Hybrid retrieval** — dense (embedding) similarity fused with BM25 keyword
  search via reciprocal rank fusion, so exact terms (`PodDisruptionBudget`,
  `--dry-run`) aren't lost to semantic-similarity noise.
- **Grounded citations, verified in code** — every answer must cite `[doc N]`
  for its claims; every citation is checked against what was *actually*
  retrieved, not trusted at face value. Ungrounded citations are surfaced as
  a warning, not silently accepted.
- **Explicit abstention** — if the retrieved docs don't support an answer,
  the model is required to say so rather than guessing.
- **Indirect prompt-injection defense** — retrieved passages are wrapped in
  `<doc>` delimiters with an explicit instruction that their content is data,
  not commands to follow, even if a passage contains something that reads
  like an instruction.
- **A bounded, reproducible corpus** — indexes a fixed, documented subset of
  `kubernetes/website` (the `docs/concepts` and `docs/tasks` sections), not
  "the whole internet," so scope and provenance are always clear.
- **A versioned evaluation set** — see [`eval/`](eval/) for retrieval
  hit-rate, citation-validity, and abstention-correctness checks.

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- Models pulled locally:
  ```
  ollama pull nomic-embed-text
  ollama pull llama3.2:1b
  ```
- Python 3.9+
- Git (for fetching the corpus)

## Install

```
python -m venv venv
venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

## 1. Fetch the corpus

A shallow, sparse clone of `kubernetes/website`, restricted to the
`docs/concepts` and `docs/tasks` sections — this is the entire, fixed
corpus boundary:

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/kubernetes/website.git
cd website
git sparse-checkout set content/en/docs/concepts content/en/docs/tasks
cd ..
```

The `website/` directory is gitignored in this repo (it's an external
corpus, re-fetched on demand, not vendored).

## 2. Build the index

```
k8s-rag ingest
```

This walks the corpus, chunks it (header-aware, ~550-word target chunk size
with overlap), embeds every chunk with `nomic-embed-text`, and persists a
[Chroma](https://www.trychroma.com/) collection to `./chroma/` (also
gitignored — it's a rebuildable artifact, not source).

## 3. Ask questions

```
k8s-rag ask "how do I set a memory limit on a container?"
k8s-rag ask "what is a PodDisruptionBudget?" -o answer.txt
k8s-rag ask                                    # no question given -> prompts interactively
```

Equivalently, without the installed console script: `python -m k8s_docs_rag ask "..."`.

### Flags

| Flag | Applies to | Default | Description |
| --- | --- | --- | --- |
| `--content-root` | `ingest` | `website/content/en` | Root of the cloned corpus |
| `--subdir` (repeatable) | `ingest` | `docs/concepts`, `docs/tasks` | Corpus subdirectories to index |
| `--persist-dir` | both | `chroma` | Where the Chroma index lives |
| `--embed-model` | both | `nomic-embed-text` (or `$K8S_RAG_EMBED_MODEL`) | Ollama embedding model |
| `--model` | `ask` | `llama3.2:1b` (or `$K8S_RAG_MODEL`) | Ollama chat model |
| `--host` | both | `$OLLAMA_HOST` / `http://localhost:11434` | Ollama server URL |
| `--timeout` | `ask` | `120` | Seconds to wait before giving up |
| `-k, --top-k` | `ask` | `5` | Number of chunks to retrieve |
| `-o, --output` | `ask` | — | Write the answer to a file instead of stdout |
| `-v, --verbose` | both | off | Enable debug logging |

### Errors

Ollama not running, a model not pulled yet, or a missing index all produce a
one-line `Error: ...` message on stderr and exit code `1` — not a Python
traceback.

## Testing

```
pytest
```

The suite mocks `ollama.Client` and uses Chroma's in-memory ephemeral mode,
so it runs fully offline — no Ollama installation required to validate the
chunking, fusion, citation-validation, and CLI wiring logic.

## Evaluation

```
python eval/run_eval.py
```

Runs a versioned set of ~20 questions (`eval/eval_set.yaml`) — a mix of
answerable questions (checking retrieval hit-rate and citation validity) and
deliberately out-of-scope ones (checking correct abstention) — against a
*built* index, and writes a timestamped report to `eval/results/`. This
needs Ollama running with both models pulled; it isn't run in hosted CI for
that reason. The CLI/chunking/fusion logic that *can* run without a live
model is covered by `pytest` instead.

## Project layout

```
src/k8s_docs_rag/
├── chunking.py   # markdown loading, cleanup, header-aware chunking
├── ingest.py     # corpus walking, embedding, Chroma indexing
├── retrieve.py   # hybrid (dense + BM25) retrieval with reciprocal rank fusion
├── generate.py   # prompt construction, the Ollama call, citation validation, abstention
├── pipeline.py   # wires retrieve + generate into the `ask()` used by the CLI
└── cli.py        # `k8s-rag ingest` / `k8s-rag ask`
eval/             # versioned eval set + eval runner
tests/            # pytest suite (mocked/offline)
```
