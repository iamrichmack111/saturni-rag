#!/usr/bin/env python3
"""Compatibility launcher for running Saturni directly from the repository."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from saturni_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
