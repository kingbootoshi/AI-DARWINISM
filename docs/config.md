## Configuration

Environment variables (see `.env.example` and `config/settings.py`):

- `OPENROUTER_API_KEY` / `OPENAI_API_KEY`: provider key
- `OPENROUTER_API_BASE` / `OPENAI_API_BASE`: custom API base
- `DSPY_MODEL`: model slug (default `openai/gpt-4o-mini`)
- `DSPY_CACHE`: enable DSPy LM cache (`true`/`false`)
- `DSPY_ENABLE_LOGGING`: DSPy internal logging (`true`)
- `DSPY_ENABLE_LITELLM_LOGGING`: provider debug logs (`false`)
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `MLFLOW_AUTOLOG`
- `LOG_LEVEL`, `LOG_DIR`, `LOG_FILE_ROTATION`, `LOG_FILE_RETENTION`

Logs are written to `logs/` with human-readable and JSONL files.