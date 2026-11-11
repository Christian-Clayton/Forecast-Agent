"""Test with a valid forecast question."""
import os
import asyncio
import sys
import logging

base = r'C:\Projects\forecast-agent'
sys.path.insert(0, os.path.join(base, 'src'))

os.environ['OPENAI_COMPAT_API_KEY'] = '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'
os.environ['OPENAI_COMPAT_MODEL'] = 'nvidia/nemotron-3-ultra-550b-a55b:free'
os.environ['RUN_MODE'] = 'backtest'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

from forecast_agent.config import Settings
settings = Settings.from_env()

from forecast_agent.llm.openai_compat_provider import OpenAICompatProvider
provider = OpenAICompatProvider.from_settings(settings)

from forecast_agent.agent.pipeline import ForecastAgent
from forecast_agent.data.synth.generator import generate_synthetic_data

agent = ForecastAgent(provider=provider, settings=settings, dataset=generate_synthetic_data())


async def test_agent():
    question = "Will the S&P 500 be higher on Nov 30, 2027 than on Nov 30, 2026?"
    print(f"Question: {question}")
    print("Running agent (timeout 180s)...")
    import time
    start = time.time()
    try:
        result = await asyncio.wait_for(agent.run(question), timeout=180)
        elapsed = time.time() - start
        print(f"SUCCESS: {elapsed:.1f}s")
        print(f"Refusal: {result.refusal_reason}")
        print(f"Steps: {result.steps_used}")
        if result.forecast:
            print(f"Forecast: {result.forecast.probability}")
            print(f"Quantiles: {result.forecast.quantiles}")
        print(f"Cost: ${result.cost_used_usd:.4f}")
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"TIMEOUT after {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()


asyncio.run(test_agent())
