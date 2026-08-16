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
            [1.0, 0.0],
            [0.8, 0.6],
            [0.2, 0.98],
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
                "source": "/tmp/a.txt",
                "source_name": "a.txt",
                "sha256": "a",
                "chunk_number": 1,
                "text": "Strong match",
            },
            {
                "source": "/tmp/b.txt",
                "source_name": "b.txt",
                "sha256": "b",
                "chunk_number": 1,
                "text": "Medium match",
            },
            {
                "source": "/tmp/c.txt",
                "source_name": "c.txt",
                "sha256": "c",
                "chunk_number": 1,
                "text": "Weak match",
            },
        ],
    }

    store.data_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store.index_path))
    store.metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    return store


def test_threshold_filters_weak_results(tmp_path):
    store = make_store(tmp_path)

    results = store.search(
        "wisdom",
        FakeClient(),
        min_similarity=0.75,
        strategy="dense",
        top_k=3,
    )

    assert len(results) == 2
    assert all(chunk.score >= 0.75 for chunk in results)


def test_high_threshold_can_abstain(tmp_path):
    store = make_store(tmp_path)

    results = store.search(
        "wisdom",
        FakeClient(),
        min_similarity=1.01,
        strategy="dense",
        top_k=3,
    )

    assert results == []
