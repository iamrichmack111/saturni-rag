# 🪐 Saturni RAG

**A lightweight, local-first Retrieval-Augmented Generation system built for transparent retrieval, deterministic citations, and practical knowledge search.**

![Version](https://img.shields.io/badge/version-1.2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)
![Quality Gate](https://img.shields.io/badge/quality%20gate-passing-brightgreen)
![Engineering Index](https://img.shields.io/badge/engineering%20index-8.3%2F10-brightgreen)
![Richmack Weissman](https://img.shields.io/badge/Richmack%20Weissman-9.1%2F10-brightgreen)

Saturni RAG turns a directory of documents into a searchable local knowledge system.

Instead of sending your entire knowledge base to a remote service, Saturni indexes documents locally, creates embeddings through Ollama, stores vectors with FAISS, retrieves the most relevant evidence, and gives a local language model the context needed to answer a question.

Version **1.2.0** focuses on retrieval quality, citation integrity, runtime hardening, testing, packaging, and engineering automation.

---

## ✨ Features

- Local document ingestion
- Automatic text chunking
- Configurable chunk overlap
- Ollama embeddings
- FAISS vector search
- Similarity-based retrieval
- Minimum similarity thresholds
- Candidate over-fetching
- Maximum Marginal Relevance (MMR)
- Duplicate-result reduction
- Deterministic source citations
- Citation sanitization
- Retrieval abstention for weak matches
- Document update handling
- Local generation through Ollama
- Interactive CLI
- Runtime diagnostics
- Configurable retrieval parameters
- Python package installation
- Automated testing
- Ruff linting
- Python 3.10 / 3.11 / 3.12 CI
- Automated package builds
- GitHub release automation
- Engineering metrics
- Richmack quality gate

---

## 🧠 How Saturni RAG Works

The basic pipeline is:

    Documents
        │
        ▼
    Text Discovery
        │
        ▼
    Chunking
        │
        ▼
    Ollama Embeddings
        │
        ▼
    FAISS Vector Index
        │
        ▼
    Similarity Search
        │
        ▼
    Candidate Over-Fetching
        │
        ▼
    MMR Diversification
        │
        ▼
    Similarity Threshold
        │
        ▼
    Retrieved Evidence
        │
        ▼
    Local LLM
        │
        ▼
    Answer + Citations

The core principle is simple:

> Retrieval happens before generation.

The model is not expected to magically know what is inside your documents.

Saturni first finds relevant evidence and then provides that evidence to the generation model.

---

## 🔎 Retrieval Pipeline

### 1. Embedding Search

The user's question is converted into an embedding.

That vector is compared against vectors stored in the FAISS index.

    Question
       ↓
    Embedding
       ↓
    Vector similarity
       ↓
    Candidate chunks

### 2. Candidate Over-Fetching

Instead of immediately taking only the final number of requested results, Saturni can retrieve a larger candidate pool first.

For example:

    top_k   = 5
    fetch_k = 20

Saturni can retrieve 20 candidates and then select the best 5.

This gives the diversification stage more evidence to work with.

### 3. Maximum Marginal Relevance

MMR balances relevance and diversity.

Without diversification, vector search can return several nearly identical chunks from the same section of a document.

Conceptually:

    MMR = λ × relevance - (1 - λ) × redundancy

A higher lambda favors relevance.

A lower lambda favors diversity.

### 4. Minimum Similarity Threshold

Not every vector match is useful.

Saturni can reject retrieved chunks below a configured similarity threshold.

Conceptually:

    similarity >= minimum_similarity

If the evidence is too weak, Saturni can abstain instead of generating an answer from poor context.

---

## 📚 Citation Integrity

Saturni attaches retrieved evidence to generated answers.

Citation handling is deterministic.

The system validates citation references against the evidence actually returned by retrieval.

This prevents generated citation markers from silently pointing to nonexistent evidence.

Example:

    Saturn is the sixth planet from the Sun. [1]

    Sources:
    [1] astronomy.txt

The generation model produces the answer.

Saturni controls the evidence mapping.

---

## 🛡️ Retrieval Abstention

A RAG system should sometimes say:

    I don't have enough evidence to answer that.

That is preferable to generating an answer from unrelated chunks.

Saturni's similarity threshold allows retrieval to fail safely when the knowledge base does not contain sufficiently relevant evidence.

---

## 🦙 Ollama

Saturni uses Ollama for local model access.

Typical architecture:

    Saturni
       │
       ├── Embedding Model
       │
       └── Generation Model
              │
              ▼
            Ollama

Check locally installed models with:

    ollama list

---

## 🚀 Installation

Clone the repository:

    git clone git@github.com:iamrichmack111/saturni-rag.git
    cd saturni-rag

Create a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install Saturni:

    python -m pip install --upgrade pip
    python -m pip install -e .

For development:

    python -m pip install -e '.[dev]'

---

## ⚡ Quick Start

Verify the installation:

    saturni --version

Expected version:

    1.2.0

View available commands:

    saturni --help

---

## 🩺 Diagnostics

Use Saturni's available CLI diagnostics to verify the local runtime and Ollama environment.

Check the available commands with:

    saturni --help

---

## 📥 Index Documents

Saturni's indexing pipeline performs:

    discover files
          ↓
    read text
          ↓
    split into chunks
          ↓
    create embeddings
          ↓
    build FAISS index
          ↓
    store retrieval metadata

Use:

    saturni --help

to see the exact indexing syntax for the installed version.

---

## 💬 Ask Questions

After building an index, Saturni retrieves relevant chunks and provides them to the configured generation model.

Conceptually:

    question
       ↓
    query embedding
       ↓
    FAISS candidates
       ↓
    MMR
       ↓
    threshold filtering
       ↓
    evidence
       ↓
    generation
       ↓
    answer + sources

Use:

    saturni --help

for the exact CLI options available in the installed version.

---

## ⚙️ Retrieval Controls

Saturni includes retrieval concepts such as:

    top_k
    fetch_k
    min_similarity
    mmr_lambda
    chunk_size
    overlap

### top_k

Number of final chunks returned to the generation stage.

### fetch_k

Number of initial candidates retrieved before diversification.

Usually:

    fetch_k >= top_k

### min_similarity

Minimum acceptable similarity between the query and retrieved evidence.

Higher values make retrieval stricter.

### mmr_lambda

Controls the relevance/diversity tradeoff during MMR selection.

### chunk_size

Controls the approximate amount of text placed into each indexed chunk.

### overlap

Controls how much neighboring chunks share.

Overlap helps preserve context around chunk boundaries.

---

## 🧪 Testing

Run the complete test suite:

    pytest

Saturni v1.2.0 currently passes:

    21 tests

The tests cover areas including retrieval behavior, thresholds, citation integrity, and runtime hardening.

---

## 🧹 Linting

Saturni uses Ruff.

Run:

    ruff check .

Automatically fix supported lint issues:

    ruff check . --fix

Current v1.2.0 state:

    All checks passed!

---

## 🚦 Quality Gate

Saturni includes a local engineering quality gate.

Run:

    ./scripts/quality-gate

The gate verifies:

    Syntax
    Tests
    Metrics JSON
    Engineering Metrics

A successful run ends with:

    PASS: Saturni quality gate

---

## 📊 Engineering Metrics

Run:

    ./scripts/richmack-metrics

Saturni v1.2.0 metrics:

    Saturni Engineering Metrics
    ==============================================================
    Source files:        4
    Source lines:        1305
    Test files:          5
    Test lines:          593
    Test/source ratio:   45.44%
    Git commits:         23
    Contributors:        3
    Release tags:        2

    Python complexity
    --------------------------------------------------------------
    Functions:           39
    Average complexity:  4.74
    Maximum complexity:  17
    Long functions:      8
    Syntax errors:       0

    Engineering Scores
    --------------------------------------------------------------
    Throughput              9.8/10
    Automation              9.0/10
    Testing                 8.7/10
    Complexity              6.6/10
    Technical Debt          7.2/10
    Maintainability         8.8/10

    ENGINEERING INDEX       8.3/10
    RICHMACK WEISSMAN       9.1/10

---

## 📐 Richmack Weissman

The **Richmack Weissman** is a custom engineering-efficiency metric used to evaluate how effectively a repository converts development activity into tested, automated, maintainable software.

It is not a standard industry benchmark.

It is a project-specific engineering measurement used across Richmack projects.

Saturni RAG v1.2.0:

    RICHMACK WEISSMAN
    9.1 / 10

---

## 🏗️ Continuous Integration

Saturni runs automated GitHub Actions workflows across:

    Python 3.10
    Python 3.11
    Python 3.12

The CI pipeline performs:

    Install
       ↓
    Version Check
       ↓
    Ruff
       ↓
    Tests
       ↓
    Package Build
       ↓
    Package Validation
       ↓
    Wheel Installation Test

This verifies both the source tree and the package users actually install.

---

## 📦 Packaging

Saturni uses a standard Python package structure.

The primary package lives under:

    src/saturni_rag/

Distribution artifacts can include:

    Python wheel
    Source archive

The release workflow builds and validates the distributions.

---

## 🚀 Release Automation

Version tags matching:

    vX.Y.Z

trigger the release workflow.

For example:

    v1.2.0

The release pipeline verifies:

    Git tag
       ↓
    Package version
       ↓
    CHANGELOG entry
       ↓
    Ruff
       ↓
    Tests
       ↓
    Build
       ↓
    Distribution validation
       ↓
    Wheel installation
       ↓
    SHA-256 checksums
       ↓
    GitHub Release

A release is not published unless the release checks succeed.

---

## 📁 Project Structure

Simplified repository structure:

    saturni-rag/
    │
    ├── .github/
    │   └── workflows/
    │
    ├── src/
    │   └── saturni_rag/
    │       ├── __init__.py
    │       ├── cli.py
    │       └── core.py
    │
    ├── scripts/
    │   ├── quality-gate
    │   └── richmack-metrics
    │
    ├── tests/
    ├── test_citation_integrity.py
    ├── test_retrieval_threshold.py
    ├── CHANGELOG.md
    ├── pyproject.toml
    └── README.md

---

## 🎯 Design Philosophy

### Local First

Your knowledge base and models can remain on hardware you control.

### Retrieval Before Generation

Evidence is selected before the model answers.

### Evidence Over Confidence

A confident model response is not a substitute for relevant retrieved evidence.

### Abstention Over Guessing

Weak retrieval should produce no answer rather than an unsupported answer.

### Deterministic Citations

Citation integrity should be controlled by software rather than trusted entirely to generation.

### Test the Retrieval Layer

RAG quality depends on more than whether an LLM can generate fluent text.

Retrieval, ranking, filtering, citations, document updates, and runtime failure modes all need testing.

### Automation Over Memory

If a release requirement matters, it should be checked automatically.

---

## 🪐 Saturni RAG v1.2.0

Version 1.2.0 represents a shift from a basic local RAG application toward a more hardened retrieval system.

The release adds stronger retrieval controls, diversified evidence selection, similarity filtering, deterministic citation handling, runtime hardening, broader automated testing, engineering metrics, multi-version CI, package validation, and automated releases.

Current verified local release state:

    Ruff                PASS
    Tests               21/21 PASS
    Quality Gate        PASS
    Engineering Index   8.3/10
    Richmack Weissman   9.1/10

---

## 👤 Author

**Jeremy Franklin**

GitHub: `iamrichmack111`

---

## 📜 License

See the repository license for usage and distribution terms.
