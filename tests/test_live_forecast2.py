
import asyncio
import os
import sys
sys.path.insert(0, "C:/Projects/forecast-agent/src")

from forecast_agent.config import Settings
from forecast_agent.llm.openai_compat_provider import OpenAICompatProvider
from forecast_agent.agent.pipeline import ForecastAgent

async def main():
    os.environ["RUN_MODE"] = "live"
    os.environ["OPENAI_COMPAT_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b:free"
    os.environ["OPENAI_COMPAT_API_KEY"] = "***-v1-***"
    
    settings = Settings()
    provider = await OpenAICompatProvider.from_settings(settings)
    agent = ForecastAgent(provider, settings)
    question = "What will the EUR/USD exchange rate be on 2026-12-31?"
    result = await agent.run(question, as_of=None)
    print("Forecast result:", result[:300] if result else "None")

asyncio.run(main())
