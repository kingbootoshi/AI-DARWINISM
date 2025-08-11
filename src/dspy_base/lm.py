"""
DSPy LM configuration and diagnostics toggles.

This module creates a configured `dspy.LM` instance based on app settings and
activates optional Diagnostics (DSPy logging and LiteLLM debug logs).
"""

from __future__ import annotations

from typing import Optional
from loguru import logger
import dspy

from config.settings import Settings


def configure_dspy(settings: Settings) -> dspy.LM:
    """Configure DSPy with a language model and diagnostics.

    Note: `cache=False` is recommended while learning, so every run is visible in
    logs and traces. Switch to True when you want to avoid repeated calls.
    """
    # Build the LM: DSPy forwards provider args to LiteLLM under the hood
    lm_kwargs = {
        "model": settings.model,
        "cache": settings.cache_lm,
    }
    # Always pass provider base/key through to LM so downstream clones inherit them
    if settings.api_base:
        lm_kwargs["api_base"] = settings.api_base
    if settings.api_key:
        lm_kwargs["api_key"] = settings.api_key

    lm = dspy.LM(**lm_kwargs)
    dspy.settings.configure(lm=lm, track_usage=settings.track_usage)
    logger.info("DSPy configured with model: {} (cache={})", settings.model, settings.cache_lm)

    # Toggle DSPy event logging
    if settings.enable_dspy_logging:
        try:
            dspy.enable_logging()
            logger.debug("DSPy internal event logging enabled")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to enable DSPy logging: {}", exc)

    # Toggle LiteLLM debug logs for provider calls (noisy but insightful)
    if settings.enable_litellm_logging:
        try:
            dspy.enable_litellm_logging()
            logger.debug("LiteLLM debug logging enabled")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to enable LiteLLM logging: {}", exc)

    return lm


