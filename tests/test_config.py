from pydantic import SecretStr

from forecast_agent.config import Settings


def test_settings_load_provider_and_budget_values_from_environment() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_COMPAT_API_KEY": "test-key",
            "OPENAI_COMPAT_BASE_URL": "https://openrouter.ai/api/v1",
            "OPENAI_COMPAT_MODEL": "provider/main-model",
            "OPENAI_COMPAT_ENSEMBLE_MODELS": "provider/member-a,provider/member-b",
            "FORECAST_MAX_STEPS": "12",
            "FORECAST_MAX_TOKENS": "9000",
            "FORECAST_MAX_COST_USD": "0.75",
            "FORECAST_SKILLS_DIR": "custom-skills",
        }
    )

    assert settings.openai_compat_api_key == SecretStr("test-key")
    assert settings.openai_compat_base_url == "https://openrouter.ai/api/v1"
    assert settings.openai_compat_model == "provider/main-model"
    assert settings.ensemble_models == ("provider/member-a", "provider/member-b")
    assert settings.max_steps == 12
    assert settings.max_tokens == 9000
    assert settings.max_cost_usd == 0.75
    assert settings.skills_dir.name == "custom-skills"
    assert "test-key" not in repr(settings)
