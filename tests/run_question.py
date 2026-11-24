import sys
sys.path.insert(0, 'C:/Projects/forecast-agent/src')
from forecast_agent.self_growth_agent import run_self_growing_agent
from forecast_agent.config import Settings

settings = Settings.from_env()
result = run_self_growing_agent('Will the president be re-elected?', settings=settings)
print('RESULT: ' + str(result))
