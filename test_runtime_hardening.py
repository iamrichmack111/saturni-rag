import ast
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import requests


source = Path("saturni.py").read_text()
tree = ast.parse(source)

wanted = {
    "embed",
    "load_index",
    "ollama_generate",
}

nodes = [
    n for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name in wanted
]

code = compile(
    ast.Module(body=nodes, type_ignores=[]),
    "saturni.py",
    "exec",
)


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, _n):
        pass


class FakeResponse:
    def __init__(
        self,
        data=None,
        lines=None,
        error=None,
    ):
        self.data = data
        self.lines = lines or []
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    def iter_lines(self):
        return iter(self.lines)


class FakeIndex:
    def __init__(self, ntotal):
        self.ntotal = ntotal


def make_env():
    return {
        "os": os,
        "np": np,
        "requests": requests,
        "json": json,
        "pickle": pickle,
        "tqdm": DummyTqdm,
        "EMBED_MODEL": "nomic-embed-text",
        "INDEX_FILE": "books.faiss",
        "META_FILE": "meta.pkl",
        "ensure_ollama_model": lambda _model: None,
    }


def test_embed_rejects_missing_vectors(monkeypatch):
    env = make_env()

    class FakeRequests:
        RequestException = requests.RequestException

        @staticmethod
        def post(*args, **kwargs):
            return FakeResponse(data={"embeddings": []})

    env["requests"] = FakeRequests

    exec(code, env)

    with pytest.raises(RuntimeError, match="missing vectors"):
        env["embed"]("wisdom")


def test_embed_reports_http_failure():
    env = make_env()

    class FakeRequests:
        RequestException = requests.RequestException

        @staticmethod
        def post(*args, **kwargs):
            return FakeResponse(
                error=requests.ConnectionError("offline")
            )

    env["requests"] = FakeRequests

    exec(code, env)

    with pytest.raises(RuntimeError, match="request failed"):
        env["embed"]("wisdom")


def test_load_index_rejects_count_mismatch(tmp_path):
    env = make_env()

    meta = tmp_path / "meta.pkl"
    meta.write_bytes(
        pickle.dumps([
            ("a.txt", "one"),
            ("b.txt", "two"),
        ])
    )

    env["INDEX_FILE"] = str(tmp_path / "books.faiss")
    env["META_FILE"] = str(meta)

    Path(env["INDEX_FILE"]).write_bytes(b"fake")

    class FakeFaiss:
        @staticmethod
        def read_index(_path):
            return FakeIndex(ntotal=3)

    env["faiss"] = FakeFaiss

    exec(code, env)

    with pytest.raises(RuntimeError, match="mismatch"):
        env["load_index"]()


def test_load_index_rejects_corrupt_metadata(tmp_path):
    env = make_env()

    index_path = tmp_path / "books.faiss"
    meta_path = tmp_path / "meta.pkl"

    index_path.write_bytes(b"fake")
    meta_path.write_bytes(b"not-a-pickle")

    env["INDEX_FILE"] = str(index_path)
    env["META_FILE"] = str(meta_path)

    class FakeFaiss:
        @staticmethod
        def read_index(_path):
            return FakeIndex(ntotal=1)

    env["faiss"] = FakeFaiss

    exec(code, env)

    with pytest.raises(RuntimeError, match="metadata"):
        env["load_index"]()


def test_generate_rejects_malformed_stream():
    env = make_env()

    class FakeRequests:
        RequestException = requests.RequestException

        @staticmethod
        def post(*args, **kwargs):
            return FakeResponse(lines=[b"not-json"])

    env["requests"] = FakeRequests

    exec(code, env)

    with pytest.raises(RuntimeError, match="malformed"):
        env["ollama_generate"](
            "prompt",
            model="gemma3:4b",
        )


def test_generate_rejects_empty_response():
    env = make_env()

    class FakeRequests:
        RequestException = requests.RequestException

        @staticmethod
        def post(*args, **kwargs):
            return FakeResponse(
                lines=[
                    json.dumps({"done": True}).encode()
                ]
            )

    env["requests"] = FakeRequests

    exec(code, env)

    with pytest.raises(RuntimeError, match="empty response"):
        env["ollama_generate"](
            "prompt",
            model="gemma3:4b",
        )
