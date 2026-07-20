"""Command-line interface for Saturni RAG."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import sys
from pathlib import Path

from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm

from saturni_rag import __version__
from saturni_rag.core import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    OllamaClient,
    OllamaUnavailableError,
    SaturniError,
    VectorStore,
    answer_question,
    default_data_dir,
    discover_text_files,
    ollama_binary_available,
)


def banner() -> None:
    print(colored(Figlet(font="slant").renderText("Saturni"), "cyan"))


_embedding_bar = None
_embedding_total: int | None = None


def close_embedding_bar() -> None:
    """Close and reset the current embedding progress bar."""
    global _embedding_bar, _embedding_total

    if _embedding_bar is not None:
        _embedding_bar.close()

    _embedding_bar = None
    _embedding_total = None


def progress(message: str) -> None:
    """Render status messages and live embedding progress."""
    global _embedding_bar, _embedding_total

    match = re.fullmatch(
        r"Embedding chunks (\d+)-(\d+) of (\d+)",
        message,
    )

    if match:
        _start, stop, total = (
            int(value) for value in match.groups()
        )

        if _embedding_bar is None or _embedding_total != total:
            close_embedding_bar()

            _embedding_total = total
            _embedding_bar = tqdm(
                total=total,
                desc="Embedding documents",
                unit="chunk",
                dynamic_ncols=True,
                leave=True,
            )

        increment = stop - _embedding_bar.n

        if increment > 0:
            _embedding_bar.update(increment)

        if stop >= total:
            close_embedding_bar()

        return

    close_embedding_bar()
    tqdm.write(colored(f"• {message}", "yellow"))


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Index directory (default: %(default)s)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama server URL",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBED_MODEL,
        help="Ollama embedding model (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")


def add_chunk_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=16)


def add_answer_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument(
        "--choose-model",
        action="store_true",
        help="Choose an installed Ollama model from an interactive menu",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Append the question and answer to a file",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saturni",
        description="Local-first RAG over text documents using Ollama and FAISS.",
    )
    parser.add_argument("--version", action="version", version=f"Saturni RAG {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build a new vector index")
    add_runtime_options(index_parser)
    add_chunk_options(index_parser)
    index_parser.add_argument("paths", nargs="*", help="Text files or directories")
    index_parser.add_argument("--force", action="store_true", help="Replace an existing index")

    add_parser = subparsers.add_parser("add", help="Add documents to the existing index")
    add_runtime_options(add_parser)
    add_chunk_options(add_parser)
    add_parser.add_argument("paths", nargs="+", help="Text files or directories")

    ask_parser = subparsers.add_parser("ask", help="Ask one question")
    add_runtime_options(ask_parser)
    add_answer_options(ask_parser)
    ask_parser.add_argument("question", help="Question to answer")

    repl_parser = subparsers.add_parser("repl", help="Open an interactive question loop")
    add_runtime_options(repl_parser)
    add_answer_options(repl_parser)

    pull_parser = subparsers.add_parser("pull", help="Download an Ollama model")
    add_runtime_options(pull_parser)
    pull_parser.add_argument("model")

    doctor_parser = subparsers.add_parser("doctor", help="Check the installation and runtime")
    add_runtime_options(doctor_parser)

    return parser


def translate_legacy_args(argv: list[str]) -> list[str]:
    """Preserve the original command flags while advertising the new subcommands."""
    if "--index" in argv:
        translated = [item for item in argv if item != "--index"]
        return ["index", *translated]
    if "--add" in argv:
        position = argv.index("--add")
        return ["add", *argv[position + 1 :], *argv[:position]]
    if "--query" in argv:
        position = argv.index("--query")
        if position + 1 >= len(argv):
            return argv
        question = argv[position + 1]
        remaining = argv[:position] + argv[position + 2 :]
        remaining = ["--model" if item == "--ai" else item for item in remaining]
        return ["ask", question, *remaining]
    if "--repl" in argv:
        translated = [item for item in argv if item != "--repl"]
        translated = ["--model" if item == "--ai" else item for item in translated]
        return ["repl", *translated]
    return argv


def choose_installed_model(client: OllamaClient, current: str) -> str:
    """Prompt the user to choose a locally installed generation model."""
    models = sorted(
        model
        for model in client.models(refresh=True)
        if "embed" not in model.lower()
    )

    if not models:
        raise SaturniError("No Ollama generation models are installed.")

    default_model = current if current in models else models[0]

    print(colored("\nInstalled Ollama models", "cyan"))

    for number, model in enumerate(models, start=1):
        marker = "  (current)" if model == default_model else ""
        print(f"  {number:>2}. {model}{marker}")

    while True:
        choice = input(
            colored(
                f"Select model [Enter = {default_model}]: ",
                "blue",
            )
        ).strip()

        if not choice:
            return default_model

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index]

        if choice in models:
            return choice

        print(colored(
            "Enter a listed number or exact model name.",
            "red",
        ))


def append_transcript(path: Path | None, question: str, answer: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"Question: {question}\n\n{answer}\n\n---\n\n")
    print(colored(f"Saved transcript to {path}", "cyan"))


def print_sources(chunks: list[object]) -> None:
    print(colored("\nSources", "yellow"))
    for number, chunk in enumerate(chunks, start=1):
        print(
            f"[{number}] {chunk.source} · chunk {chunk.chunk_number} "
            f"· similarity {chunk.score:.3f}"
        )


def ask_once(
    args: argparse.Namespace,
    store: VectorStore,
    client: OllamaClient,
) -> None:
    generation_bar = tqdm(
        desc=f"Generating with {args.model}",
        unit="chunk",
        dynamic_ncols=True,
        leave=True,
    )

    try:
        answer, chunks = answer_question(
            args.question,
            store,
            client,
            model=args.model,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            on_token=lambda _token: generation_bar.update(1),
        )
    finally:
        generation_bar.close()

    print(colored(answer, "green"))

    if args.show_sources:
        print_sources(chunks)

    append_transcript(
        args.output,
        args.question,
        answer,
    )


def run_doctor(args: argparse.Namespace) -> int:
    store = VectorStore(args.data_dir)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python", sys.version_info >= (3, 10), platform.python_version()))
    for package in ("faiss", "numpy", "requests", "pyfiglet", "termcolor", "tqdm"):
        checks.append(
            (
                f"Package: {package}",
                importlib.util.find_spec(package) is not None,
                "installed",
            )
        )
    checks.append(("Ollama command", ollama_binary_available(), "found in PATH"))

    client = OllamaClient(args.ollama_url, args.timeout)
    try:
        model_count = len(client.models())
        checks.append(("Ollama server", True, f"reachable; {model_count} model(s) installed"))
    except OllamaUnavailableError as exc:
        checks.append(("Ollama server", False, str(exc)))

    try:
        store.data_dir.mkdir(parents=True, exist_ok=True)
        probe = store.data_dir / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(("Data directory", True, str(store.data_dir)))
    except OSError as exc:
        checks.append(("Data directory", False, str(exc)))
    checks.append(("Vector index", store.exists(), str(store.index_path)))

    print(colored("Saturni diagnostics\n", "cyan"))
    for name, ok, detail in checks:
        symbol = "PASS" if ok else "FAIL"
        color = "green" if ok else "red"
        print(colored(f"{symbol:4}", color), f" {name}: {detail}")
    return 0 if all(ok for name, ok, _ in checks if name != "Vector index") else 1


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(translate_legacy_args(raw_args))

    if args.command == "doctor":
        return run_doctor(args)

    store = VectorStore(args.data_dir)
    client = OllamaClient(args.ollama_url, args.timeout)

    try:
        if args.command in {"ask", "repl"} and args.choose_model:
            args.model = choose_installed_model(client, args.model)
            print(colored(f"Using model: {args.model}\n", "magenta"))

        if args.command == "index":
            banner()
            files = discover_text_files(args.paths)
            count = store.build(
                files,
                client,
                args.embedding_model,
                args.chunk_size,
                args.overlap,
                args.batch_size,
                force=args.force,
                progress=progress,
            )
            print(colored(f"Indexed {count} chunks from {len(files)} document(s).", "green"))

        elif args.command == "add":
            banner()
            files = discover_text_files(args.paths)
            count, skipped = store.add(
                files,
                client,
                args.embedding_model,
                args.chunk_size,
                args.overlap,
                args.batch_size,
                progress=progress,
            )
            print(colored(f"Added {count} new chunks.", "green"))
            if skipped:
                print(colored(f"Skipped unchanged documents: {', '.join(skipped)}", "yellow"))

        elif args.command == "ask":
            ask_once(args, store, client)

        elif args.command == "repl":
            banner()
            print(colored(f"Active model: {args.model}", "magenta"))
            print(colored("Type '/model' to switch models.", "magenta"))
            print(colored("Type 'exit' or 'quit' to close Saturni.\n", "magenta"))

            while True:
                try:
                    question = input(colored("Ask> ", "blue")).strip()
                except EOFError:
                    print()
                    break

                if question.lower() in {"exit", "quit"}:
                    break

                if question.lower() in {"/model", "model"}:
                    args.model = choose_installed_model(client, args.model)
                    print(colored(
                        f"Using model: {args.model}\n",
                        "magenta",
                    ))
                    continue

                if not question:
                    continue

                args.question = question
                ask_once(args, store, client)
                print()

        elif args.command == "pull":
            client.ensure_model(args.model, progress=progress)
            print(colored(f"Model ready: {args.model}", "green"))

        return 0
    except (SaturniError, ValueError) as exc:
        print(colored(f"Error: {exc}", "red"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(colored("\nCancelled.", "yellow"), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
