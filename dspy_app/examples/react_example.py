"""
ReAct example with rich logging, history inspection, and optional MLflow traces.
"""

from __future__ import annotations

import json
from pathlib import Path
from loguru import logger
import dspy

from config.settings import get_settings
from config.logging import init_logging
from config.tracing import init_tracing
from dspy_app.lm import configure_dspy
from dspy_app.callbacks import AgentLoggingCallback
from dspy_app.tools import evaluate_math, search_wikipedia


def main(question: str | None = None) -> None:
    """Run a ReAct agent and persist inputs/outputs and chain steps to logs.

    Parameters
    ----------
    question: Optional[str]
        Custom question; falls back to a numeric+retrieval query when omitted.
    """
    settings = get_settings()
    init_logging(
        app_name="Darwinism",
        level=settings.log_level,
        log_dir=settings.log_dir,
        colorize_console=settings.colorize_console,
        file_rotation=settings.file_rotation,
        file_retention=settings.file_retention,
    )

    tracing = init_tracing(settings)
    lm = configure_dspy(settings)
    dspy.settings.configure(callbacks=[AgentLoggingCallback()])

    # Include reasoning in outputs to make streaming and post-mortem easier
    signature = "question -> answer, reasoning"
    react = dspy.ReAct(signature, tools=[evaluate_math, search_wikipedia], max_iters=4)

    q = (
        question
        or "What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?"
    )
    logger.info("Running ReAct with question: {}", q)
    pred = react(question=q)

    logger.success("Answer: {}", pred.answer)
    if getattr(pred, "reasoning", None):
        logger.info("Reasoning: {}", pred.reasoning)

    # Show the last interactions (prompts/messages/responses)
    dspy.inspect_history(n=5)

    # Persist the last LM call in structured JSON for later study
    try:
        last = lm.history[-1]
        out_path = settings.log_dir / "lm_last.json"
        with out_path.open("w") as f:
            json.dump(last, f, indent=2)
        logger.info("Wrote last LM call to {}", out_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not persist LM history: {}", exc)


if __name__ == "__main__":  # pragma: no cover
    main()


