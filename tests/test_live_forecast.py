
import asyncio
import os
import sys
sys.path.insert(0, "C:/Projects/forecast-agent/src")

from forecast_agent.config import Settings
from forecast_agent.agent.pipeline import ForecastAgent

async def main():
    # Set up live mode
    os.environ["RUN_MODE"] = "live"
    os.environ["OPENAI_COMPAT_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b:free"
    
    settings = Settings()
    agent = ForecastAgent(settings)
    
    # Run a forecast
    question = "What will the EUR/USD exchange rate be on 2026-12-31?"
    result = await agent.run(question, as_of=None)
    
    print("Forecast question:", question)
    print("Result:" if result else "No result")
    if result:
        print(f"  Type: {type(result).__name__}")
        print(f"  Resolutions: {len(result.resolutions) if result.resolutions else 0}")
        print(f"  Questions: {len(result.questions) if result.questions else 0}")
        if result.questions:
            for q in result.questions:
                print(f"    - {q}")

asyncio.run(main())
