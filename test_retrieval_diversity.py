import ast
from pathlib import Path
import numpy as np

source = Path("saturni.py").read_text()
tree = ast.parse(source)

wanted = {
    "_cosine_similarity",
    "_mmr_select",
    "retrieve",
}

nodes = [
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name in wanted
]

code = compile(
    ast.Module(body=nodes, type_ignores=[]),
    "saturni.py",
    "exec",
)


class FakeIndex:
    def search(self, qv, k):
        distances = np.array(
            [[0.60, 0.61, 0.62, 0.70]],
            dtype="float32",
        )
        indices = np.array(
            [[0, 1, 2, 3]],
            dtype="int64",
        )
        return distances[:, :k], indices[:, :k]

    def reconstruct(self, idx):
        vectors = {
            0: np.array([1.00, 0.00], dtype="float32"),
            1: np.array([0.99, 0.01], dtype="float32"),
            2: np.array([0.98, 0.02], dtype="float32"),
            3: np.array([0.20, 0.98], dtype="float32"),
        }
        return vectors[idx]


docs = [
    ("plato.txt", "Plato chunk one"),
    ("plato.txt", "Plato chunk two"),
    ("plato.txt", "Plato chunk three"),
    ("aristotle.txt", "Aristotle evidence"),
]


def load_index():
    return FakeIndex(), docs


def embed(_q):
    return np.array([[1.0, 0.0]], dtype="float32")


env = {
    "np": np,
    "load_index": load_index,
    "embed": embed,
}

exec(code, env)

retrieve = env["retrieve"]


def test_dense_keeps_nearest_neighbors():
    results = retrieve(
        "wisdom",
        topk=3,
        max_distance=0.90,
        strategy="dense",
        fetch_k=4,
    )

    assert [x[0] for x in results] == [
        "plato.txt",
        "plato.txt",
        "plato.txt",
    ]


def test_mmr_increases_context_diversity():
    results = retrieve(
        "wisdom",
        topk=3,
        max_distance=0.90,
        strategy="mmr",
        fetch_k=4,
        lambda_mult=0.50,
    )

    filenames = [x[0] for x in results]

    assert "plato.txt" in filenames
    assert "aristotle.txt" in filenames


def test_threshold_still_applies_to_mmr():
    results = retrieve(
        "wisdom",
        topk=3,
        max_distance=0.65,
        strategy="mmr",
        fetch_k=4,
        lambda_mult=0.50,
    )

    assert all(distance <= 0.65 for _, _, distance in results)


def test_invalid_strategy_is_rejected():
    try:
        retrieve(
            "wisdom",
            strategy="magic",
        )
    except ValueError as exc:
        assert "dense" in str(exc)
        assert "mmr" in str(exc)
    else:
        raise AssertionError("Invalid retrieval strategy was accepted")
