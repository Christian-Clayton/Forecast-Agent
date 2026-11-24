import asyncio
import sys
sys.path.insert(0, 'C:/Projects/forecast-agent/src')

from forecast_agent.config import Settings
from forecast_agent.llm.multi_provider import MultiProvider
from forecast_agent.llm.gemini_provider import GeminiProvider
from forecast_agent.agent.pipeline import ForecastAgent

# Create a test agent with Gemini provider
settings = Settings.from_env()
print("Token present:", hasattr(settings, 'token') and settings.token is not None)
print("OpenAI compat model:", settings.openai_compat_model)

# Create providers
# Use the settings token (from .env)
token = settings.openai_compat_token
if isinstance(token, str):
    gemini_api_key = token
else:
    gemini_api_key = token.get_secret_value()

gemini_provider = GeminiProvider(
    api_key=gemini_api_key,
    model='gemini-3.5-flash-lite',
    temperature=0.0,
    max_tokens=512,
    timeout_seconds=180.0,
    max_retries=settings.llm_max_retries,
    initial_backoff_seconds=settings.llm_initial_backoff_seconds,
)

# Create MultiProvider
multi_provider = MultiProvider(
    providers=[
        ("gemini", gemini_provider, ["gemini-3.5-flash-lite"]),
    ],
)

# Create agent
agent = ForecastAgent(provider=multi_provider, settings=settings)

# Test a stock-related question
question = 'Will AAPL stock be higher in 7 days?'

print('')
print('Question:', question)
print('=' * 60)

try:
    result = asyncio.run(agent.run(question))
    
    print('')
    print('Forecast resolved:', result.forecast is not None)
    if result.forecast:
        print('Probability:', result.forecast.probability)
        print('Tool call IDs:', result.forecast.tool_call_ids)
        print('Baseline source:', result.forecast.base_rate_source)
        print('Domain class:', result.forecast.domain_class)
        print('Toolkit:', result.forecast.toolkit)
        print('Cost: $%.4f' % result.forecast.cost_usd)
        print('Latency: %.2fs' % result.forecast.latency_s)
    
    print('')
    print('Refusal reason:', result.refusal_reason)
    
    print('')
    print('Trace summary:')
    for trace in result.trace:
        if isinstance(trace, dict):
            if trace.get('status') == 'started':
                print('  %s: started' % trace.get('stage'))
            elif trace.get('status') == 'completed':
                print('  %s: completed' % trace.get('stage'))
            elif trace.get('status') == 'failed':
                print('  %s: failed' % trace.get('stage'))
        else:
            print('  %s' % trace)
    
    # Show model inputs
    if result.forecast and result.forecast.tool_call_ids:
        print('')
        print('Tool call IDs:', result.forecast.tool_call_ids)
    
except Exception as e:
    print('')
    print('Error:', e)
    import traceback
    traceback.print_exc()
