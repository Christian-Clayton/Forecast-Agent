import sys
sys.path.insert(0, 'C:/Projects/forecast-agent/src')
import asyncio
import json
import time
import anyio

from forecast_agent.self_growth_agent import DynamicForecastAgent
from forecast_agent.config import Settings

settings = Settings.from_env()
print(f'Token present: {settings.openai_compat_token is not None}')

agent = DynamicForecastAgent(settings)

QUESTIONS = [
    ('health', 'Will vaccination rates increase next year?'),
    ('politics', 'Will the president be re-elected?'),
    ('sports', 'Will the home team win the championship?'),
    ('economics', 'Will inflation decrease next year?'),
    ('science', 'Will fusion energy be commercially viable?'),
    ('education', 'Will online learning adoption continue to grow?'),
    ('environment', 'Will global temperatures rise further?'),
    ('technology', 'Will AI chatbots replace customer service?'),
    ('business', 'Will the stock market reach new highs?'),
    ('general', 'Will it rain more next month?'),
]

results = []

# Run the entire batch in one anyio.run() call
# This ensures the anyio async context is properly set up for all questions
async def run_all():
    for i, (domain, subject) in enumerate(QUESTIONS, 1):
        print(f'\n=== Question {i}/10: {domain} ===')
        print(f'Question: {subject}')
        
        start = time.time()
        try:
            # Run the agent within the anyio context
            result = await agent.run(subject)
            elapsed = time.time() - start
            prob = result.forecast.probability if result.forecast else None
            if prob is not None:
                print(f'Success! Probability: {prob} Elapsed: {elapsed:.1f}s')
            else:
                print(f'Refused: {result.refusal_reason} Elapsed: {elapsed:.1f}s')
            results.append({
                'domain': domain,
                'question': subject,
                'success': True,
                'probability': prob,
                'refusal_reason': result.refusal_reason,
                'bottom_line': result.answer.bottom_line if result.answer else None,
                'elapsed': elapsed,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            elapsed = time.time() - start
            print(f'ERROR: {e}')
            results.append({
                'domain': domain,
                'question': subject,
                'success': False,
                'error': str(e),
                'elapsed': elapsed,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if i < 10:
            wait_time = 180 - (time.time() - start)
            if wait_time > 0:
                print(f'Waiting {wait_time:.0f}s before next question...')
                await asyncio.sleep(wait_time)
    return results

# Run the entire batch in one anyio.run() call
# This creates a new event loop that's compatible with anyio's task groups
results = anyio.run(run_all)

print('\n=== Summary ===')
successful = [r for r in results if r['success']]
print(f'Successful: {len(successful)}/10')

with open('C:/Projects/forecast-agent/10_domain_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nResults saved to 10_domain_test_results.json')
