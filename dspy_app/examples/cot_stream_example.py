"""
Streaming reasoning example using ChainOfThought + streamify.
"""

from __future__ import annotations

import asyncio
from loguru import logger
import dspy

from config.settings import get_settings
from config.logging import init_logging
from config.tracing import init_tracing
from dspy_app.lm import configure_dspy


class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


async def _run_streamed(question: str):
    # Ask for reasoning implicitly via ChainOfThought
    cot = dspy.ChainOfThought(QA)
    listener = dspy.streaming.StreamListener("reasoning")
    stream_cot = dspy.streamify(cot, stream_listeners=[listener])

    final = None
    async for chunk in stream_cot(question=question):
        if isinstance(chunk, dspy.Prediction):
            final = chunk
        else:
            logger.info("<blue>stream</blue>: {}", chunk)
    return final


def main(question: str | None = None) -> None:
    settings = get_settings()
    init_logging(
        app_name="Darwinism",
        level=settings.log_level,
        log_dir=settings.log_dir,
        colorize_console=settings.colorize_console,
        file_rotation=settings.file_rotation,
        file_retention=settings.file_retention,
    )
    init_tracing(settings)
    configure_dspy(settings)

    q = question or "Why did the chicken cross the road?"
    logger.info("Streaming reasoning for: {}", q)
    pred = asyncio.run(_run_streamed(q))
    logger.success("Answer: {}", getattr(pred, "answer", None))
    if getattr(pred, "reasoning", None):
        logger.debug("Full reasoning: {}", pred.reasoning)


if __name__ == "__main__":  # pragma: no cover
    main()


