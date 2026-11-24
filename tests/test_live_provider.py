import asyncio
import os
import sys
sys.path.insert(0, "C:/Projects/forecast-agent/src")

from forecast_agent.config import Settings
from forecast_agent.llm.openai_compat_provider import OpenAICompatProvider
from forecast_agent.agent.pipeline import ForecastAgent

async def main():
    # Live mode with OpenRouter key
    os.environ["RUN_MODE"] = "live"
    os.environ["OPENAI_COMPAT_TOKEN"] = "***-v1-***"
    os.environ["OPENAI_COMPAT_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b:free"
    
    print("Settings:", Settings())
    
    # Try creating the provider
    try:
        provider = await OpenAICompatProvider.from_settings(Settings())
        print("Provider created successfully")
        print("Model:", provider.model)
        print("Max tokens:", provider.max_tokens)
    except Exception as e:
        print(f"Error creating provider: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(main())
