"""
Minimal example showing how to access LM history and usage.
"""

from __future__ import annotations

import json
from loguru import logger
import dspy

from config.settings import get_settings
from config.logging import init_logging
from dspy_app.lm import configure_dspy


def main() -> None:
    settings = get_settings()
    init_logging(
        app_name="Darwinism",
        level=settings.log_level,
        log_dir=settings.log_dir,
        colorize_console=settings.colorize_console,
        file_rotation=settings.file_rotation,
        file_retention=settings.file_retention,
    )
    lm = configure_dspy(settings)

    # Run a tiny prediction
    predict = dspy.Predict("question -> answer")
    out = predict(question="2+2?")
    logger.info("Out: {}", out)

    # Inspect recent DSPy interactions
    dspy.inspect_history(n=3)

    # Peek at the raw log of the last provider call
    try:
        last = lm.history[-1]
        logger.debug("LM entry keys: {}", list(last.keys()))
        logger.debug("Usage: {}", last.get("usage"))
        (settings.log_dir / "lm_last_compact.json").write_text(
            json.dumps({k: last.get(k) for k in ("prompt", "messages", "outputs", "usage", "cost")}, indent=2)
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not access lm.history: {}", exc)


if __name__ == "__main__":  # pragma: no cover
    main()


