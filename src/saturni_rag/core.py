"""Core retrieval, indexing, and Ollama integration for Saturni RAG."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import requests

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_GENERATION_MODEL = "gemma2:2b"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 75
DEFAULT_TOP_K = 3
DEFAULT_FETCH_K = 12
DEFAULT_MIN_SIMILARITY = 0.45
DEFAULT_MMR_LAMBDA = 0.70


class SaturniError(RuntimeError):
    """Base exception for user-facing Saturni failures."""


class OllamaUnavailableError(SaturniError):
    """Raised when the Ollama server cannot be reached."""


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    chunk_number: int
    score: float


def default_data_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "saturni-rag" / "data"


def normalize_ollama_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return "http://127.0.0.1:11434"
    if "://" not in value:
        value = f"http://{value}"
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_words(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be at least zero and smaller than chunk size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def discover_text_files(paths: Sequence[str] | None = None, cwd: Path | None = None) -> list[Path]:
    cwd = (cwd or Path.cwd()).resolve()
    candidates: list[Path] = []

    if paths:
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = cwd / path
            if path.is_file() and path.suffix.lower() == ".txt":
                candidates.append(path.resolve())
            elif path.is_dir():
                candidates.extend(item.resolve() for item in path.rglob("*.txt") if item.is_file())
            else:
                raise SaturniError(f"Document path not found or not a .txt file: {raw}")
    else:
        documents_dir = cwd / "documents"
        if documents_dir.is_dir():
            candidates.extend(item.resolve() for item in documents_dir.rglob("*.txt"))
        candidates.extend(item.resolve() for item in cwd.glob("clean_pg*.txt"))
        candidates.extend(item.resolve() for item in cwd.glob("pg*.txt"))

    return sorted(set(candidates), key=lambda item: str(item).lower())


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        configured_url = base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.base_url = normalize_ollama_url(configured_url)
        self.timeout = timeout
        self.session = requests.Session()
        self._model_cache: set[str] | None = None

    def _request(self, method: str, endpoint: str, **kwargs: object) -> requests.Response:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{endpoint}",
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                f"Unable to reach Ollama at {self.base_url}. Start it with 'ollama serve'."
            ) from exc

    def models(self, refresh: bool = False) -> set[str]:
        if self._model_cache is not None and not refresh:
            return self._model_cache

        payload = self._request("GET", "/api/tags").json()
        names: set[str] = set()
        for model in payload.get("models", []):
            name = model.get("model") or model.get("name")
            if name:
                names.add(str(name))
        self._model_cache = names
        return names

    def ensure_model(
        self,
        model: str,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        installed = self.models()
        if model in installed or any(name.split(":", 1)[0] == model for name in installed):
            return

        if progress:
            progress(f"Pulling Ollama model: {model}")

        response = self._request(
            "POST",
            "/api/pull",
            json={"model": model, "stream": True},
            stream=True,
        )
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if error := data.get("error"):
                raise SaturniError(f"Ollama failed to pull {model}: {error}")
        self.models(refresh=True)

    def embed_many(
        self,
        texts: Sequence[str],
        model: str = DEFAULT_EMBED_MODEL,
    ) -> np.ndarray:
        if not texts:
            raise SaturniError("No text was supplied for embedding.")

        response = self._request(
            "POST",
            "/api/embed",
            json={"model": model, "input": list(texts)},
        )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SaturniError(
                "Ollama embedding response was not valid JSON."
            ) from exc

        embeddings = payload.get("embeddings")

        if not embeddings:
            raise SaturniError(
                "Ollama returned no embeddings."
            )

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        if vectors.ndim != 2:
            raise SaturniError(
                "Ollama returned an invalid embedding matrix."
            )

        if vectors.shape[0] != len(texts):
            raise SaturniError(
                "Ollama returned an invalid embedding matrix."
            )

        if vectors.shape[1] == 0:
            raise SaturniError(
                "Ollama returned an empty embedding vector."
            )

        if not np.isfinite(vectors).all():
            raise SaturniError(
                "Ollama returned invalid embedding values."
            )

        faiss.normalize_L2(vectors)
        return vectors

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_GENERATION_MODEL,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        response = self._request(
            "POST",
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
            },
            stream=True,
        )

        output: list[str] = []

        for line in response.iter_lines():
            if not line:
                continue

            try:
                data = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SaturniError(
                    "Ollama generation stream returned malformed data."
                ) from exc

            if error := data.get("error"):
                raise SaturniError(
                    f"Ollama generation failed: {error}"
                )

            token = str(
                data.get("response", "")
            )

            if token:
                output.append(token)

                if on_token:
                    on_token(token)

        answer = "".join(output).strip()

        if not answer:
            raise SaturniError(
                "Ollama generation returned an empty response."
            )

        return answer



class VectorStore:
    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir or default_data_dir()).expanduser().resolve()
        self.index_path = self.data_dir / "books.faiss"
        self.metadata_path = self.data_dir / "metadata.json"

    def exists(self) -> bool:
        return self.index_path.is_file() and self.metadata_path.is_file()

    def _load(self) -> tuple[faiss.Index, dict[str, object]]:
        if not self.exists():
            raise SaturniError("No index found. Run 'saturni index <documents>' first.")
        index = faiss.read_index(str(self.index_path))
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise SaturniError("Unsupported index metadata. Rebuild the index.")
        chunks = payload.get("chunks")
        if not isinstance(chunks, list) or index.ntotal != len(chunks):
            raise SaturniError(
                "Index metadata is inconsistent. Rebuild with 'saturni index --force'."
            )
        return index, payload

    def _save(self, index: faiss.Index, payload: dict[str, object]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.data_dir) as temp_dir:
            temp = Path(temp_dir)
            temp_index = temp / "books.faiss"
            temp_metadata = temp / "metadata.json"
            faiss.write_index(index, str(temp_index))
            temp_metadata.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_index.replace(self.index_path)
            temp_metadata.replace(self.metadata_path)

    def build(
        self,
        files: Sequence[Path],
        client: OllamaClient,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        batch_size: int = 16,
        force: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> int:
        if self.exists() and not force:
            raise SaturniError(
                "An index already exists. Use --force to replace it or 'saturni add'."
            )
        if not files:
            raise SaturniError("No text documents were found.")
        if batch_size <= 0:
            raise SaturniError("Batch size must be greater than zero.")

        client.ensure_model(embedding_model, progress=progress)
        metadata: list[dict[str, object]] = []
        texts: list[str] = []

        for file_path in files:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            file_hash = sha256_file(file_path)
            chunks = chunk_words(content, chunk_size, overlap)
            if progress:
                progress(f"Prepared {len(chunks)} chunks from {file_path.name}")
            for number, chunk in enumerate(chunks, start=1):
                metadata.append(
                    {
                        "source": str(file_path),
                        "source_name": file_path.name,
                        "sha256": file_hash,
                        "chunk_number": number,
                        "text": chunk,
                    }
                )
                texts.append(chunk)

        if not texts:
            raise SaturniError("The selected documents contained no indexable text.")

        vector_batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            if progress:
                progress(f"Embedding chunks {start + 1}-{stop} of {len(texts)}")
            vector_batches.append(client.embed_many(texts[start:stop], embedding_model))

        vectors = np.vstack(vector_batches).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        payload: dict[str, object] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": embedding_model,
            "dimension": int(vectors.shape[1]),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": metadata,
        }
        self._save(index, payload)
        return len(metadata)

    def add(
        self,
        files: Sequence[Path],
        client: OllamaClient,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        batch_size: int = 16,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[int, list[str]]:
        if not self.exists():
            return (
                self.build(
                    files,
                    client,
                    embedding_model,
                    chunk_size,
                    overlap,
                    batch_size,
                    force=True,
                    progress=progress,
                ),
                [],
            )

        index, payload = self._load()

        stored = {
            "embedding_model": str(payload.get("embedding_model", "")),
            "chunk_size": int(payload.get("chunk_size", -1)),
            "overlap": int(payload.get("overlap", -1)),
        }

        requested = {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }

        if stored != requested:
            raise SaturniError(
                "Index configuration mismatch. "
                f"Stored={stored}, requested={requested}. "
                "Use the stored settings or rebuild the index."
            )

        metadata = payload["chunks"]
        assert isinstance(metadata, list)

        skipped: list[str] = []
        changed: dict[str, tuple[Path, str]] = {}

        for file_path in files:
            resolved = file_path.resolve()
            source_id = str(resolved)
            file_hash = sha256_file(resolved)

            existing = [
                item
                for item in metadata
                if str(item.get("source", "")) == source_id
            ]

            if existing and all(
                str(item.get("sha256", "")) == file_hash
                for item in existing
            ):
                skipped.append(resolved.name)
                continue

            changed[source_id] = (resolved, file_hash)

        if not changed:
            return 0, skipped

        retained_metadata: list[dict[str, object]] = []
        retained_vectors: list[np.ndarray] = []

        for position, item in enumerate(metadata):
            source_id = str(item.get("source", ""))

            if source_id in changed:
                continue

            retained_metadata.append(item)
            retained_vectors.append(
                np.asarray(
                    index.reconstruct(position),
                    dtype="float32",
                ).reshape(1, -1)
            )

        new_metadata: list[dict[str, object]] = []
        new_texts: list[str] = []

        client.ensure_model(embedding_model, progress=progress)

        for source_id, (file_path, file_hash) in changed.items():
            chunks = chunk_words(
                file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ),
                chunk_size,
                overlap,
            )

            for number, chunk in enumerate(chunks, start=1):
                new_metadata.append(
                    {
                        "source": source_id,
                        "source_name": file_path.name,
                        "sha256": file_hash,
                        "chunk_number": number,
                        "text": chunk,
                    }
                )
                new_texts.append(chunk)

        new_batches: list[np.ndarray] = []

        for start_pos in range(0, len(new_texts), batch_size):
            stop = min(start_pos + batch_size, len(new_texts))

            if progress:
                progress(
                    f"Embedding chunks {start_pos + 1}-{stop} "
                    f"of {len(new_texts)}"
                )

            new_batches.append(
                client.embed_many(
                    new_texts[start_pos:stop],
                    embedding_model,
                )
            )

        parts: list[np.ndarray] = []

        if retained_vectors:
            parts.append(
                np.vstack(retained_vectors).astype("float32")
            )

        if new_batches:
            parts.append(
                np.vstack(new_batches).astype("float32")
            )

        if not parts:
            raise SaturniError(
                "No chunks remain after updating the index."
            )

        vectors = np.vstack(parts).astype("float32")
        faiss.normalize_L2(vectors)

        rebuilt = faiss.IndexFlatIP(vectors.shape[1])
        rebuilt.add(vectors)

        payload["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        payload["chunks"] = retained_metadata + new_metadata

        self._save(rebuilt, payload)

        return len(new_metadata), skipped

    def search(
        self,
        question: str,
        client: OllamaClient,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        strategy: str = "mmr",
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise SaturniError("top-k must be greater than zero.")

        if strategy not in {"dense", "mmr"}:
            raise SaturniError(
                "retrieval strategy must be 'dense' or 'mmr'."
            )

        if not 0.0 <= lambda_mult <= 1.0:
            raise SaturniError(
                "MMR lambda must be between 0 and 1."
            )

        index, payload = self._load()

        stored_model = str(payload.get("embedding_model", ""))

        if stored_model != embedding_model:
            raise SaturniError(
                f"Index uses embedding model '{stored_model}', "
                f"not '{embedding_model}'. "
                "Use the stored model or rebuild the index."
            )

        metadata = payload["chunks"]
        assert isinstance(metadata, list)

        client.ensure_model(embedding_model)

        query_vector = client.embed_many(
            [question],
            embedding_model,
        )

        candidate_count = (
            max(top_k, fetch_k)
            if strategy == "mmr"
            else top_k
        )

        count = min(candidate_count, index.ntotal)

        scores, indices = index.search(
            query_vector,
            count,
        )

        candidates: list[dict[str, object]] = []

        for score, index_position in zip(
            scores[0],
            indices[0],
            strict=True,
        ):
            if index_position < 0:
                continue

            similarity = float(score)

            if similarity < min_similarity:
                continue

            item = metadata[int(index_position)]

            candidates.append(
                {
                    "position": int(index_position),
                    "source": str(
                        item.get("source_name")
                        or item.get("source")
                    ),
                    "text": str(item["text"]),
                    "chunk_number": int(
                        item.get("chunk_number", 0)
                    ),
                    "score": similarity,
                }
            )

        if strategy == "dense":
            chosen = candidates[:top_k]

        else:
            selected: list[dict[str, object]] = []
            remaining = list(candidates)

            while remaining and len(selected) < top_k:
                if not selected:
                    best = max(
                        remaining,
                        key=lambda item: float(item["score"]),
                    )
                else:
                    def mmr_score(item: dict[str, object]) -> float:
                        vector = np.asarray(
                            index.reconstruct(
                                int(item["position"])
                            ),
                            dtype="float32",
                        )

                        redundancy = max(
                            float(
                                np.dot(
                                    vector,
                                    np.asarray(
                                        index.reconstruct(
                                            int(other["position"])
                                        ),
                                        dtype="float32",
                                    ),
                                )
                            )
                            for other in selected
                        )

                        same_source = any(
                            item["source"] == other["source"]
                            for other in selected
                        )

                        source_penalty = (
                            0.10 if same_source else 0.0
                        )

                        return (
                            lambda_mult * float(item["score"])
                            - (1.0 - lambda_mult) * redundancy
                            - source_penalty
                        )

                    best = max(remaining, key=mmr_score)

                selected.append(best)
                remaining.remove(best)

            chosen = selected

        return [
            RetrievedChunk(
                source=str(item["source"]),
                text=str(item["text"]),
                chunk_number=int(item["chunk_number"]),
                score=float(item["score"]),
            )
            for item in chosen
        ]



def build_prompt(
    question: str,
    chunks: Iterable[RetrievedChunk],
) -> str:
    context_sections = []

    for number, chunk in enumerate(chunks, start=1):
        context_sections.append(
            f"[CITE {number}]\n"
            f"Source: {chunk.source}\n"
            f"Chunk: {chunk.chunk_number}\n"
            f"Similarity: {chunk.score:.4f}\n"
            f"{chunk.text}"
        )

    context = "\n\n".join(context_sections)

    return (
        "You are Saturni, a careful retrieval-grounded research assistant.\n"
        "Answer only from the supplied context.\n"
        "Use only Saturni citation identifiers that appear in the context, "
        "such as [CITE 1] or [CITE 2].\n"
        "Do not invent citation identifiers.\n"
        "If the context does not contain enough evidence, say exactly: "
        "Not found in text.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def sanitize_citations(answer: str, valid_count: int) -> str:
    import re

    valid = set(range(1, valid_count + 1))

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return match.group(0) if number in valid else ""

    return re.sub(
        r"\[CITE\s+(\d+)\]",
        replace,
        answer,
    ).strip()


def format_sources(
    chunks: Sequence[RetrievedChunk],
) -> str:
    lines = ["Sources:"]

    for number, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[CITE {number}] {chunk.source} "
            f"| chunk={chunk.chunk_number} "
            f"| similarity={chunk.score:.4f}"
        )

    return "\n".join(lines)


def answer_question(
    question: str,
    store: VectorStore,
    client: OllamaClient,
    model: str = DEFAULT_GENERATION_MODEL,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    retrieval: str = "mmr",
    fetch_k: int = DEFAULT_FETCH_K,
    lambda_mult: float = DEFAULT_MMR_LAMBDA,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    chunks = store.search(
        question,
        client,
        embedding_model,
        top_k,
        min_similarity=min_similarity,
        strategy=retrieval,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    if not chunks:
        return "Not found in text.", []

    client.ensure_model(model)

    answer = client.generate(
        build_prompt(question, chunks),
        model,
        on_token=on_token,
    )

    answer = sanitize_citations(
        answer,
        len(chunks),
    )

    return (
        f"{answer}\n\n{format_sources(chunks)}",
        chunks,
    )


def ollama_binary_available() -> bool:
    return shutil.which("ollama") is not None
