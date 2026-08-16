import json

import faiss
import numpy as np
import pytest

from saturni_rag.core import (
    OllamaClient,
    SaturniError,
    VectorStore,
)


class FakeResponse:
    def __init__(self, data=None, lines=None):
        self._data = data
        self._lines = lines or []

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data

    def iter_lines(self):
        return iter(self._lines)

    def raise_for_status(self):
        return None


def test_embed_rejects_missing_vectors(monkeypatch):
    client = OllamaClient()

    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: FakeResponse(
            data={"embeddings": []}
        ),
    )

    with pytest.raises(
        SaturniError,
        match="no embeddings",
    ):
        client.embed_many(
            ["wisdom"],
            "nomic-embed-text",
        )


def test_embed_rejects_invalid_matrix(monkeypatch):
    client = OllamaClient()

    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: FakeResponse(
            data={"embeddings": [[1.0, 2.0], [3.0, 4.0]]}
        ),
    )

    with pytest.raises(
        SaturniError,
        match="invalid embedding matrix",
    ):
        client.embed_many(
            ["wisdom"],
            "nomic-embed-text",
        )


def test_embed_rejects_invalid_json(monkeypatch):
    client = OllamaClient()

    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: FakeResponse(
            data=ValueError("bad json")
        ),
    )

    with pytest.raises(
        SaturniError,
        match="valid JSON",
    ):
        client.embed_many(
            ["wisdom"],
            "nomic-embed-text",
        )


def test_load_rejects_count_mismatch(tmp_path):
    store = VectorStore(tmp_path)

    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype="float32",
    )

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    store.data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(store.index_path),
    )

    payload = {
        "schema_version": 1,
        "embedding_model": "nomic-embed-text",
        "dimension": 2,
        "chunk_size": 500,
        "overlap": 75,
        "chunks": [
            {
                "source": "/tmp/a.txt",
                "source_name": "a.txt",
                "sha256": "a",
                "chunk_number": 1,
                "text": "one",
            },
            {
                "source": "/tmp/b.txt",
                "source_name": "b.txt",
                "sha256": "b",
                "chunk_number": 1,
                "text": "two",
            },
        ],
    }

    store.metadata_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        SaturniError,
        match="inconsistent",
    ):
        store._load()


def test_load_rejects_corrupt_metadata(tmp_path):
    store = VectorStore(tmp_path)

    vectors = np.array(
        [[1.0, 0.0]],
        dtype="float32",
    )

    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    store.data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(store.index_path),
    )

    store.metadata_path.write_text(
        "not-json",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, json.JSONDecodeError)):
        store._load()


def test_generate_rejects_malformed_stream(monkeypatch):
    client = OllamaClient()

    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: FakeResponse(
            lines=[b"not-json"]
        ),
    )

    with pytest.raises(
        SaturniError,
        match="malformed",
    ):
        client.generate(
            "prompt",
            "gemma3:4b",
        )


def test_generate_rejects_empty_response(monkeypatch):
    client = OllamaClient()

    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: FakeResponse(
            lines=[
                json.dumps(
                    {"done": True}
                ).encode()
            ]
        ),
    )

    with pytest.raises(
        SaturniError,
        match="empty response",
    ):
        client.generate(
            "prompt",
            "gemma3:4b",
        )
