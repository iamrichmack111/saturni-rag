# Changelog

All notable changes to Saturni RAG are documented here.

## [1.1.0] - 2026-07-20

### Added

- Interactive selection from locally installed Ollama models.
- Runtime model switching from the REPL with `/model`.
- Live document-embedding progress bar.
- Answer-generation activity and speed display.
- README screenshots demonstrating indexing, model selection, citations,
  similarity scores, transcript logging, and the interactive REPL.

### Fixed

- Python 3.10 compatibility by replacing `datetime.UTC` with
  `timezone.utc`.
- Package diagnostics now recognize the progress-bar dependency.

## [1.0.0] - 2026-07-20

### Added

- Isolated Linux/macOS installer and data-preserving uninstaller.
- Standard `pyproject.toml` package with `saturni` and `saturni-rag` commands.
- Subcommands for indexing, incremental additions, querying, REPL use, model pulls, and diagnostics.
- Batched Ollama embeddings, overlapping chunks, normalized vectors, and cosine similarity search.
- JSON metadata, SHA-256 duplicate detection, atomic index writes, timeouts, and actionable errors.
- Source-number citations in RAG prompts and optional retrieval score display.
- Unit tests, Ruff checks, package builds, and GitHub Actions CI.

### Changed

- Generated indexes now live under the user's XDG data directory instead of the repository.
- The original command flags remain available through compatibility translation.
