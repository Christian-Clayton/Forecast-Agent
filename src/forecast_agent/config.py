import os
from collections.abc import Mapping
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    database_path: Path = Path("data/forecast-agent.sqlite3")
    skills_dir: Path = Path("skills")
    openai_compat_base_url: str = "https://openrouter.ai/api/v1"
    openai_compat_api_key: SecretStr | None = None
    openai_compat_model: str | None = None
    ensemble_models: tuple[str, ...] = ()
    max_steps: int = Field(default=24, ge=1)
    max_tokens: int = Field(default=12_000, ge=1)
    max_cost_usd: float = Field(default=1.0, ge=0)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_initial_backoff_seconds: float = Field(default=0.25, ge=0)
    stage_max_retries: int = Field(default=1, ge=0, le=3)
    llm_circuit_failure_threshold: int = Field(default=5, ge=1)
    llm_circuit_reset_seconds: float = Field(default=30.0, gt=0)

    @field_validator("openai_compat_base_url")
    @classmethod
    def validate_provider_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("provider base URL must be an HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("provider base URL must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("provider base URL must not contain query or fragment")
        return value.rstrip("/")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        values = os.environ if environ is None else environ
        api_key = values.get("OPENAI_COMPAT_API_KEY")
        models = values.get("OPENAI_COMPAT_ENSEMBLE_MODELS", "")
        return cls(
            database_path=Path(
                values.get("FORECAST_DATABASE_PATH", "data/forecast-agent.sqlite3")
            ),
            skills_dir=Path(values.get("FORECAST_SKILLS_DIR", "skills")),
            openai_compat_base_url=values.get(
                "OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            openai_compat_api_key=SecretStr(api_key) if api_key else None,
            openai_compat_model=values.get("OPENAI_COMPAT_MODEL") or None,
            ensemble_models=tuple(
                model.strip() for model in models.split(",") if model.strip()
            ),
            max_steps=values.get("FORECAST_MAX_STEPS", 24),
            max_tokens=values.get("FORECAST_MAX_TOKENS", 12_000),
            max_cost_usd=values.get("FORECAST_MAX_COST_USD", 1.0),
            llm_timeout_seconds=values.get("LLM_TIMEOUT_SECONDS", 30.0),
            llm_max_retries=values.get("LLM_MAX_RETRIES", 2),
            llm_initial_backoff_seconds=values.get("LLM_INITIAL_BACKOFF_SECONDS", 0.25),
            stage_max_retries=values.get("FORECAST_STAGE_MAX_RETRIES", 1),
            llm_circuit_failure_threshold=values.get(
                "LLM_CIRCUIT_FAILURE_THRESHOLD", 5
            ),
            llm_circuit_reset_seconds=values.get("LLM_CIRCUIT_RESET_SECONDS", 30.0),
        )
