import json

import faiss
import numpy as np

from saturni_rag.core import VectorStore


class FakeClient:
    def ensure_model(self, _model, progress=None):
        pass

    def embed_many(self, _texts, _model):
        return np.array([[1.0, 0.0]], dtype="float32")


def make_store(tmp_path):
    store = VectorStore(tmp_path)

    vectors = np.array(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.70, 0.71],
        ],
        dtype="float32",
    )
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    payload = {
        "schema_version": 1,
        "embedding_model": "nomic-embed-text",
        "dimension": 2,
        "chunk_size": 500,
        "overlap": 75,
        "chunks": [
            {
                "source": "/tmp/plato.txt",
                "source_name": "plato.txt",
                "sha256": "1",
                "chunk_number": 1,
                "text": "Plato one",
            },
            {
                "source": "/tmp/plato.txt",
                "source_name": "plato.txt",
                "sha256": "1",
                "chunk_number": 2,
                "text": "Plato two",
            },
            {
                "source": "/tmp/plato.txt",
                "source_name": "plato.txt",
                "sha256": "1",
                "chunk_number": 3,
                "text": "Plato three",
            },
            {
                "source": "/tmp/aristotle.txt",
                "source_name": "aristotle.txt",
                "sha256": "2",
                "chunk_number": 1,
                "text": "Aristotle evidence",
            },
        ],
    }

    store.data_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store.index_path))
    store.metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    return store


def test_dense_keeps_nearest_neighbors(tmp_path):
    store = make_store(tmp_path)

    results = store.search(
        "wisdom",
        FakeClient(),
        strategy="dense",
        top_k=3,
        min_similarity=0.0,
    )

    assert [x.source for x in results] == [
        "plato.txt",
        "plato.txt",
        "plato.txt",
    ]


def test_mmr_increases_context_diversity(tmp_path):
    store = make_store(tmp_path)

    results = store.search(
        "wisdom",
        FakeClient(),
        strategy="mmr",
        top_k=3,
        fetch_k=4,
        lambda_mult=0.50,
        min_similarity=0.0,
    )

    filenames = [x.source for x in results]

    assert "plato.txt" in filenames
    assert "aristotle.txt" in filenames


def test_invalid_strategy_is_rejected(tmp_path):
    store = make_store(tmp_path)

    try:
        store.search(
            "wisdom",
            FakeClient(),
            strategy="magic",
        )
    except Exception as exc:
        assert "dense" in str(exc)
        assert "mmr" in str(exc)
    else:
        raise AssertionError("Invalid retrieval strategy was accepted")
