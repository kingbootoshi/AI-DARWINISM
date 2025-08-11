"""
Centralized Loguru logging configuration with colorful console + rotating files.

This module sets up:
- Colorized console logs with rich, high-signal context
- Rotating human-readable log files
- Rotating JSONL log files for structured analysis
- Interception of stdlib logging so third-party libraries (e.g., aiohttp, openai)
  are unified into Loguru sinks
"""

from __future__ import annotations

import logging
import inspect
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


class InterceptHandler(logging.Handler):
    """Bridge stdlib logging into Loguru, preserving levels and call sites."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = inspect.currentframe()
        depth = 0
        # Walk back until we exit the logging module to report the original caller
        while frame:
            filename = frame.f_code.co_filename
            if depth > 0 and filename != logging.__file__:
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def init_logging(
    app_name: str,
    *,
    level: str,
    log_dir: Path,
    colorize_console: bool,
    file_rotation: str,
    file_retention: str,
) -> None:
    """Initialize Loguru sinks for console and files.

    Parameters
    ----------
    app_name: str
        Arbitrary label included in console logs to identify this app.
    level: str
        Minimum log level, e.g., "DEBUG" | "INFO".
    log_dir: Path
        Directory where log files will be written.
    colorize_console: bool
        Whether to use ANSI colors in console output.
    file_rotation: str
        Rotation policy (e.g., "50 MB", "1 day").
    file_retention: str
        Retention policy (e.g., "14 days").
    """

    # Remove Loguru's default stderr handler to avoid duplicates
    logger.remove()

    # Console sink: high-signal, colorful format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
        f"<cyan>[{app_name}]</cyan> "
        "<level>{level: <8}</level> "
        "<magenta>{name}</magenta>:<magenta>{function}</magenta>:<magenta>{line}</magenta> "
        "- <level>{message}</level>\n"
    )
    logger.add(
        sys.stderr,
        level=level,
        format=console_format,
        colorize=colorize_console,
        backtrace=True,
        diagnose=False,
    )

    # Human-readable rotating log file
    logger.add(
        (log_dir / "app_{time}.log").as_posix(),
        level=level,
        rotation=file_rotation,
        retention=file_retention,
        delay=True,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    # Structured JSONL rotating log file
    logger.add(
        (log_dir / "app_{time}.jsonl").as_posix(),
        level=level,
        rotation=file_rotation,
        retention=file_retention,
        delay=True,
        enqueue=True,
        serialize=True,
        backtrace=True,
        diagnose=False,
    )

    # Intercept stdlib logs into Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for noisy_logger in ("dspy", "httpx", "openai", "aiohttp", "urllib3", "litellm"):
        logging.getLogger(noisy_logger).setLevel(logging.DEBUG)

    logger.bind(app=app_name).success("Logging initialized")


