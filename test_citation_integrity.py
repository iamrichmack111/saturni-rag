import ast
import re
from pathlib import Path


source = Path("saturni.py").read_text()
tree = ast.parse(source)

wanted = {
    "sanitize_citations",
    "format_sources",
    "rag_answer",
}

nodes = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in wanted
]

code = compile(
    ast.Module(body=nodes, type_ignores=[]),
    "saturni.py",
    "exec",
)


ctx = [
    ("wisdom.txt", "Wisdom concerns practical judgment.", 0.68),
    ("virtue.txt", "Virtue concerns right action.", 0.74),
]


def retrieve(_q, topk=3, max_distance=0.90):
    return ctx


generated_answers = []


def ollama_generate(_prompt, model):
    return generated_answers[-1]


env = {
    "re": re,
    "retrieve": retrieve,
    "ollama_generate": ollama_generate,
}

exec(code, env)

sanitize = env["sanitize_citations"]
format_sources = env["format_sources"]
rag_answer = env["rag_answer"]


def test_invalid_citation_is_removed():
    answer = "Wisdom matters [CITE 1]. Invented claim [CITE 9]."

    cleaned = sanitize(answer, {1, 2})

    assert "[CITE 1]" in cleaned
    assert "[CITE 9]" not in cleaned


def test_source_footnotes_are_not_removed():
    answer = "The source discusses wisdom [13] and judgment [CITE 1]."

    cleaned = sanitize(answer, {1})

    assert "[13]" in cleaned
    assert "[CITE 1]" in cleaned


def test_source_map_is_deterministic():
    sources = format_sources(ctx)

    assert "[CITE 1] wisdom.txt | distance=0.6800" in sources
    assert "[CITE 2] virtue.txt | distance=0.7400" in sources


def test_rag_answer_removes_hallucinated_citation():
    generated_answers.append(
        "Wisdom is practical judgment [CITE 1]. "
        "This fake citation should disappear [CITE 4]."
    )

    answer = rag_answer(
        "What is wisdom?",
        model="gemma3:4b",
        max_distance=0.90,
    )

    assert "[CITE 1]" in answer
    assert "[CITE 4]" not in answer

    assert "Sources:" in answer
    assert "wisdom.txt" in answer
    assert "virtue.txt" in answer
