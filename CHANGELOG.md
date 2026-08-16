# Changelog

## [1.2.0] - 2026-08-16

### Added
- Document-aware MMR retrieval.
- Similarity thresholding and out-of-domain abstention.
- Deterministic source-traceable citations.
- Runtime hardening for Ollama embedding and generation failures.
- Package-aware Richmack engineering metrics.
- Automated Saturni quality gate and expanded regression coverage.

### Changed
- Integrated Saturni into the `src/saturni_rag` package architecture.
- Improved document update handling to prevent stale chunks.
- Added index configuration validation.
- Updated CLI retrieval controls and package tests.

### Quality
- 21 regression tests passing.
- Richmack Weissman: 9.0/10.
- Engineering Index: 8.3/10.


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
