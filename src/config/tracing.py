"""
MLflow tracing bootstrap for DSPy programs.

This enables full, clickable traces in the MLflow UI when a tracking URI is
configured. Traces include nested spans for LM calls, tools, and modules.
"""

from __future__ import annotations

from loguru import logger

from .settings import Settings


def init_tracing(settings: Settings) -> bool:
    """Enable MLflow autologging for DSPy and optionally LiteLLM.

    Returns True if tracing is enabled, False otherwise.
    """
    if not settings.mlflow_tracking_uri:
        logger.info("MLflow tracing disabled (no MLFLOW_TRACKING_URI set)")
        return False

    try:
        import mlflow  # Lazy import in case user hasn't installed it yet
    except Exception as exc:  # pragma: no cover - import-time diagnostics only
        logger.warning("MLflow not available; skipping tracing: {}", exc)
        return False

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    if settings.mlflow_autolog:
        # Enable DSPy traces (includes module calls and nested tool/LM spans)
        try:
            mlflow.dspy.autolog(
                log_compiles=True,
                log_evals=True,
                log_traces_from_compile=True,
                log_traces_from_eval=True,
            )
            logger.success("MLflow DSPy autologging enabled")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to enable MLflow DSPy autologging: {}", exc)

        # Also capture provider-level calls if routed via LiteLLM/OpenAI
        for enabler_name, enabler in (
            ("litellm", getattr(mlflow, "litellm", None)),
            ("openai", getattr(mlflow, "openai", None)),
        ):
            if enabler is None:
                continue
            try:
                enabler.autolog()
                logger.info("MLflow {} autologging enabled", enabler_name)
            except Exception as exc:  # pragma: no cover
                logger.debug("MLflow {} autologging not available: {}", enabler_name, exc)

        return True

    logger.info("MLflow configured but autologging disabled by env")
    return False


