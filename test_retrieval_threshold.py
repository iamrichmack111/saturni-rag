import ast
from pathlib import Path
import numpy as np

source = Path("saturni.py").read_text()
tree = ast.parse(source)

wanted = {
    "_cosine_similarity",
    "_mmr_select",
    "retrieve",
    "rag_answer",
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
    def search(self, qv, topk):
        distances = np.array([[0.70, 0.82, 0.98]], dtype="float32")
        indices = np.array([[0, 1, 2]], dtype="int64")
        return distances, indices


docs = [
    ("wisdom.txt", "Wisdom is practical judgment."),
    ("virtue.txt", "Virtue concerns good action."),
    ("quantum.txt", "Unrelated material."),
]


def load_index():
    return FakeIndex(), docs


def embed(_q):
    return np.array([[1.0, 2.0]], dtype="float32")


generated = []


def ollama_generate(prompt, model):
    generated.append(prompt)
    return "generated"


env = {
    "np": np,
    "load_index": load_index,
    "embed": embed,
    "ollama_generate": ollama_generate,
}

exec(code, env)

retrieve = env["retrieve"]
rag_answer = env["rag_answer"]


def test_threshold_filters_weak_results():
    results = retrieve("wisdom", topk=3, max_distance=0.90)

    assert len(results) == 2
    assert results[0][0] == "wisdom.txt"
    assert results[1][0] == "virtue.txt"
    assert all(distance <= 0.90 for _, _, distance in results)


def test_no_evidence_abstains_without_generation():
    answer = rag_answer(
        "quantum computing",
        model="gemma3:4b",
        max_distance=0.50,
        retrieval="dense",
    )

    assert answer == "Not found in text."
    assert generated == []
