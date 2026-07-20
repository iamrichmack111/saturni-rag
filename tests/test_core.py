from pathlib import Path

import numpy as np

from saturni_rag.core import (
    RetrievedChunk,
    VectorStore,
    build_prompt,
    chunk_words,
    discover_text_files,
    normalize_ollama_url,
)


class FakeClient:
    def ensure_model(self, model, progress=None):
        if progress:
            progress(f"ready: {model}")

    def embed_many(self, texts, model):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("justice") + 1),
                    float(lowered.count("virtue") + 1),
                    float(len(lowered.split()) + 1),
                ]
            )
        array = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / norms


def test_chunk_words_uses_overlap():
    chunks = chunk_words("one two three four five six", size=4, overlap=2)
    assert chunks == ["one two three four", "three four five six", "five six"]


def test_chunk_words_rejects_invalid_overlap():
    try:
        chunk_words("text", size=5, overlap=5)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_discover_text_files_supports_directories(tmp_path: Path):
    documents = tmp_path / "documents"
    documents.mkdir()
    first = documents / "a.txt"
    second = documents / "b.txt"
    ignored = documents / "c.md"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    ignored.write_text("c", encoding="utf-8")

    assert discover_text_files([str(documents)], cwd=tmp_path) == [first, second]


def test_build_prompt_requests_numbered_citations():
    prompt = build_prompt(
        "What is virtue?",
        [RetrievedChunk("plato.txt", "Virtue is discussed here.", 2, 0.9)],
    )
    assert "[1] Source: plato.txt" in prompt
    assert "Cite supporting passages" in prompt


def test_normalize_ollama_url_adds_scheme():
    assert normalize_ollama_url("localhost:11434/") == "http://localhost:11434"


def test_vector_store_build_and_search(tmp_path: Path):
    document = tmp_path / "plato.txt"
    document.write_text(
        "Justice is harmony in the city and soul. Virtue requires wisdom and courage.",
        encoding="utf-8",
    )
    store = VectorStore(tmp_path / "index")
    client = FakeClient()

    count = store.build(
        [document],
        client,
        chunk_size=7,
        overlap=2,
        batch_size=2,
        force=True,
    )
    results = store.search("justice", client, top_k=2)

    assert count >= 2
    assert store.exists()
    assert results
    assert results[0].source == "plato.txt"
