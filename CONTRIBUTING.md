# Contributing

Contributions, bug reports, and documentation improvements are welcome.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Before opening a pull request, run:

```bash
ruff check .
pytest
python -m build
```

Keep changes focused, add tests for new behavior, and do not commit generated indexes, transcripts, virtual environments, model files, or private document collections.
