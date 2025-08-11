"""
Configuration settings for DSPy experiments and logging.

This module centralizes environment-driven configuration to avoid
hard-coding secrets and to keep behavior consistent across examples.

Environment variables (recommended via a local .env file):
- OPENROUTER_API_KEY / OPENAI_API_KEY: Provider API key
- OPENROUTER_API_BASE / OPENAI_API_BASE: Custom API base if using a proxy
- DSPY_MODEL: Model slug (default: openai/gpt-4o-mini)
- DSPY_CACHE: 'true' | 'false' to enable/disable DSPy LM cache (default: false)
- DSPY_ENABLE_LOGGING: Enable DSPy event logging (default: true)
- DSPY_ENABLE_LITELLM_LOGGING: Enable LiteLLM debug logs (default: false)
- MLFLOW_TRACKING_URI: e.g., http://127.0.0.1:5000 (optional)
- MLFLOW_EXPERIMENT: Experiment name (default: DSPy)
- MLFLOW_AUTOLOG: 'true' | 'false' to enable DSPy tracing (default: true if URI set)
- LOG_LEVEL: log level for Loguru (default: DEBUG)
- LOG_DIR: directory for log files (default: logs)

All values have sane defaults so examples run with minimal setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


def _to_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean-like environment value.

    Accepts: '1', 'true', 'yes', 'on' as True; '0', 'false', 'no', 'off' as False.
    Falls back to the provided default when None/empty/unrecognized.
    """
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class Settings:
    """App-wide configuration container.

    This is intentionally explicit to make it easy to learn which knobs
    affect DSPy, logging, and tracing.
    """

    # Model & provider
    model: str
    api_key: str | None
    api_base: str | None
    cache_lm: bool

    # DSPy & LiteLLM diagnostics
    enable_dspy_logging: bool
    enable_litellm_logging: bool
    track_usage: bool

    # MLflow tracing
    mlflow_tracking_uri: str | None
    mlflow_experiment: str
    mlflow_autolog: bool

    # Logging (Loguru)
    log_level: str
    log_dir: Path
    colorize_console: bool
    file_rotation: str
    file_retention: str


def get_settings() -> Settings:
    """Load settings from environment and defaults."""
    load_dotenv(override=False)

    # Provider credentials: prefer OpenRouter if provided, else OpenAI
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENAI_API_BASE")

    model = os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    cache_lm = _to_bool(os.getenv("DSPY_CACHE"), default=False)
    enable_dspy_logging = _to_bool(os.getenv("DSPY_ENABLE_LOGGING"), default=True)
    enable_litellm_logging = _to_bool(
        os.getenv("DSPY_ENABLE_LITELLM_LOGGING"), default=False
    )
    track_usage = True  # Track per-prediction token/cost usage while learning

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT", "DSPy")
    # Default autolog to True if a tracking URI is present
    mlflow_autolog = _to_bool(os.getenv("MLFLOW_AUTOLOG"), default=mlflow_tracking_uri is not None)

    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    log_dir = Path(os.getenv("LOG_DIR", "logs")).resolve()
    colorize_console = True
    file_rotation = os.getenv("LOG_FILE_ROTATION", "50 MB")
    file_retention = os.getenv("LOG_FILE_RETENTION", "14 days")

    # Ensure log directory exists early so sinks can open lazily
    log_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        model=model,
        api_key=api_key,
        api_base=api_base,
        cache_lm=cache_lm,
        enable_dspy_logging=enable_dspy_logging,
        enable_litellm_logging=enable_litellm_logging,
        track_usage=track_usage,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_autolog=mlflow_autolog,
        log_level=log_level,
        log_dir=log_dir,
        colorize_console=colorize_console,
        file_rotation=file_rotation,
        file_retention=file_retention,
    )


