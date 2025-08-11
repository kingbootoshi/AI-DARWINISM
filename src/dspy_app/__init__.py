"""
High-level DSPy application package.

This package contains a modular implementation of the GEPA optimization loop,
task definitions, and a thin runner that keeps `main.py` as the entrypoint.

Public surfaces:
- dspy_app.cli: CLI parser helpers
- dspy_app.runner: run(args) orchestrator
- dspy_app.gepa: GEPA core, signatures, systems, reflection and PDF extraction
- dspy_app.tasks: Task-specific modules (hotpot, horror)
"""

from __future__ import annotations

__all__ = [
    "cli",
    "runner",
]


