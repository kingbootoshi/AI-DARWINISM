"""
Command-line interface for the DSPy GEPA app.

This module isolates argument parsing so the public entrypoint (`main.py`)
can remain minimal and testable. It returns an `argparse.Namespace` that is
consumed by `dspy_app.runner.run`.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with flags mirrored from the legacy main.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default=None, help="Path to arXiv PDF (2507.19457).")
    parser.add_argument(
        "--use-paper-prompts",
        action="store_true",
        help="Load exact GEPA prompts for HotpotQA/GPT-4.1 Mini from the PDF.",
    )
    parser.add_argument("--optimize", action="store_true", help="Run the GEPA optimization loop.")
    parser.add_argument("--budget", type=int, default=60, help="GEPA rollout budget (mutation attempts).")
    parser.add_argument("--minibatch", type=int, default=6, help="Minibatch size for mutation tests.")
    parser.add_argument("--pareto", type=int, default=8, help="Pareto set size.")
    parser.add_argument(
        "--merge-prob",
        type=float,
        default=0.0,
        help="Probability of trying a Merge step per iteration.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--task", choices=["hotpot", "horror"], default="hotpot", help="Choose task to run.")
    parser.add_argument("--web", action="store_true", help="Launch a local web UI on http://127.0.0.1:3000")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args from argv or sys.argv.

    Parameters
    ----------
    argv: list[str] | None
        Optional argument vector; when None uses sys.argv.
    """
    parser = build_parser()
    return parser.parse_args(argv)



