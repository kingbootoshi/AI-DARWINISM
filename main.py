#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin entrypoint for the modular GEPA-in-DSPy application.

This file now defers all logic to `src/dspy_app` while preserving the
`python main.py` experience.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure `src/` is importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SRC_DIR.as_posix())


def main() -> None:
    # Defer heavy imports until after path setup
    from dspy_app.cli import parse_args
    from dspy_app.runner import run

    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()