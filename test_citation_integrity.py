from saturni_rag.core import (
    RetrievedChunk,
    format_sources,
    sanitize_citations,
)


def test_invalid_citation_is_removed():
    answer = "Wisdom matters [CITE 1]. Invented claim [CITE 9]."

    cleaned = sanitize_citations(answer, 2)

    assert "[CITE 1]" in cleaned
    assert "[CITE 9]" not in cleaned


def test_valid_citations_are_preserved():
    answer = "Wisdom [CITE 1]. Virtue [CITE 2]."

    cleaned = sanitize_citations(answer, 2)

    assert "[CITE 1]" in cleaned
    assert "[CITE 2]" in cleaned


def test_sources_are_deterministic():
    chunks = [
        RetrievedChunk("wisdom.txt", "Wisdom text", 3, 0.82),
        RetrievedChunk("virtue.txt", "Virtue text", 7, 0.76),
    ]

    sources = format_sources(chunks)

    assert "[CITE 1] wisdom.txt" in sources
    assert "chunk=3" in sources
    assert "[CITE 2] virtue.txt" in sources
    assert "chunk=7" in sources
