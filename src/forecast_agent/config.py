import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class RunMode(StrEnum):
    """Run mode for the agent. Set once per run, cannot change mid-run."""

    BACKTEST = "backtest"
    LIVE = "live"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    database_path: Path = Path("data/forecast-agent.sqlite3")
    skills_dir: Path = Path("skills")
    # Gemini API configuration (free tier) - uses Google's native endpoint
    openai_compat_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_compat_token: SecretStr | None = None
    # Gemini free models - gemini-3.5-flash-lite and gemini-3.1-flash-lite are working
    openai_compat_model: str = "gemini-3.5-flash-lite"
    openai_compat_model_list: tuple[str, ...] = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
    ensemble_models: tuple[str, ...] = ()
    max_steps: int = Field(default=75, ge=1)
    max_tokens: int = Field(default=50_000, ge=1)
    max_cost_usd: float = Field(default=2.0, ge=0)
    llm_timeout_seconds: float = Field(default=180.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=5)
    llm_initial_backoff_seconds: float = Field(default=0.25, ge=0)
    stage_max_retries: int = Field(default=1, ge=0, le=3)
    llm_circuit_failure_threshold: int = Field(default=5, ge=1)
    llm_circuit_reset_seconds: float = Field(default=30.0, gt=0)
    run_mode: RunMode = RunMode.BACKTEST

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
        # Try OPENAI_COMPAT_TOKEN first, then OPENAI_COMPAT_API_KEY
        token = values.get("OPENAI_COMPAT_TOKEN") or values.get("OPENAI_COMPAT_API_KEY")
        # Parse model list from environment (comma-separated)
        model_list_env = values.get("OPENAI_COMPAT_MODEL_LIST", "")
        model_list = tuple(m.strip() for m in model_list_env.split(",") if m.strip()) if model_list_env else ()
        return cls(
            database_path=Path(
                values.get("FORECAST_DATABASE_PATH", "data/forecast-agent.sqlite3")
            ),
            skills_dir=Path(values.get("FORECAST_SKILLS_DIR", "skills")),
            openai_compat_base_url=values.get(
                "OPENAI_COMPAT_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            openai_compat_token=SecretStr(token) if token else None,
            openai_compat_model=values.get("OPENAI_COMPAT_MODEL") or (
                model_list[0] if model_list else None
            ),
            openai_compat_model_list=model_list,
            ensemble_models=tuple(
                m.strip() for m in values.get("OPENAI_COMPAT_ENSEMBLE_MODELS", "").split(",") if m.strip()
            ),
            max_steps=values.get("FORECAST_MAX_STEPS", 75),
            max_tokens=values.get("FORECAST_MAX_TOKENS", 50_000),
            max_cost_usd=values.get("FORECAST_MAX_COST_USD", 2.0),
            llm_timeout_seconds=values.get("LLM_TIMEOUT_SECONDS", 180.0),
            llm_max_retries=values.get("LLM_MAX_RETRIES", 3),
            llm_initial_backoff_seconds=values.get("LLM_INITIAL_BACKOFF_SECONDS", 0.25),
            stage_max_retries=values.get("FORECAST_STAGE_MAX_RETRIES", 1),
            llm_circuit_failure_threshold=values.get(
                "LLM_CIRCUIT_FAILURE_THRESHOLD", 5
            ),
            llm_circuit_reset_seconds=values.get("LLM_CIRCUIT_RESET_SECONDS", 30.0),
            run_mode=RunMode(
                values.get("RUN_MODE", RunMode.BACKTEST.value)
            ),
        )
