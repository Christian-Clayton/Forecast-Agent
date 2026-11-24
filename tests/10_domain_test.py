"""10-domain self-growth forecast test.

Runs 10 questions across 10 different domains with 150s pacing between each.
Captures detailed output for troubleshooting.
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, "/c/Projects/forecast-agent/src")

# Also add the project src
sys.path.insert(0, "C:/Projects/forecast-agent/src")

from forecast_agent.self_growth_agent import DynamicForecastAgent
from forecast_agent.config import Settings

# 10 domains, 10 questions
QUESTIONS = [
    # 1. Politics
    {
        "domain": "politics",
        "subject": "Will the Democratic Party win the 2028 US presidential election?",
        "expected_tool": "politics",
    },
    # 2. Sports
    {
        "domain": "sports",
        "subject": "Will Manchester City win the 2027 Premier League title?",
        "expected_tool": "sports",
    },
    # 3. Economics
    {
        "domain": "economics",
        "subject": "Will inflation be lower in Q4 2027?",
        "expected_tool": "economics",
    },
    # 4. Technology
    {
        "domain": "technology",
        "subject": "Will AI adoption increase significantly in the next year?",
        "expected_tool": "technology",
    },
    # 5. Health
    {
        "domain": "health",
        "subject": "Will vaccination rates increase next year?",
        "expected_tool": "health",
    },
    # 6. Environment
    {
        "domain": "environment",
        "subject": "Will global temperature rise in the next decade?",
        "expected_tool": "environment",
    },
    # 7. Business (stock market)
    {
        "domain": "business",
        "subject": "Will the stock market go up next month?",
        "expected_tool": "business",
    },
    # 8. Science
    {
        "domain": "science",
        "subject": "Will a new major scientific breakthrough occur in physics next year?",
        "expected_tool": "science",
    },
    # 9. Energy
    {
        "domain": "energy",
        "subject": "Will renewable energy share increase in the next 5 years?",
        "expected_tool": "energy",
    },
    # 10. Education
    {
        "domain": "education",
        "subject": "Will online learning adoption increase in the next 2 years?",
        "expected_tool": "education",
    },
]

# Pacing between questions
PACING_SECONDS = 150

# Output file
OUTPUT_FILE = "10_domain_test_results.json"


async def run_single_question(agent: DynamicForecastAgent, question: dict) -> dict:
    """Run a single question and capture detailed results."""
    result = {
        "domain": question["domain"],
        "subject": question["subject"],
        "expected_tool": question["expected_tool"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": {
            "tool_generation": {},
            "registration": {},
            "pipeline": {},
            "result": {},
        },
        "error": None,
        "duration_seconds": 0,
    }

    start_time = time.time()

    try:
        # Analyze query
        self_growing_agent = agent._get_self_growing_agent()
        needed_tools = self_growing_agent._analyze_query(question["subject"])
        result["steps"]["tool_generation"] = {
            "needed_tools": [t["name"] for t in needed_tools],
            "tool_count": len(needed_tools),
        }

        # Register tools
        for tool_def in needed_tools:
            server = self_growing_agent._get_or_create_server(tool_def["module"])
            self_growing_agent.registrar.register_tool_with_server(server, tool_def)
            agent.registry.register_tool(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
                module=tool_def["module"],
                identifier=tool_def["identifier"],
            )
        result["steps"]["registration"] = {
            "registered_count": len(needed_tools),
        }

        # Run pipeline
        print(f"  Running pipeline for: {question['subject']}...")
        result_obj = await agent.run(question["subject"])
        result["steps"]["pipeline"] = {
            "success": True,
            "answer": getattr(result_obj.answer, "probability", None),
            "refusal_reason": getattr(result_obj, "refusal_reason", None),
            "steps_used": getattr(result_obj, "steps_used", 0),
            "tokens_used": getattr(result_obj, "tokens_used", 0),
        }
    except Exception as e:
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }
        result["steps"]["pipeline"] = {
            "success": False,
            "error": str(e),
        }
        print(f"  ERROR: {e}")
    finally:
        result["duration_seconds"] = time.time() - start_time

    return result


async def main():
    print("=" * 60)
    print("10-DOMAIN SELF-GROWTH FORECAST TEST")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Pacing: {PACING_SECONDS}s between questions")
    print()

    # Create settings
    settings = Settings()
    # Use the default model; we only need the agent to analyze and register tools

    # Create the agent
    agent = DynamicForecastAgent(settings)
    print(f"Agent created: {agent}")
    print()

    # Run each question
    results = []
    for i, question in enumerate(QUESTIONS):
        print("-" * 60)
        print(f"Question {i+1}/{len(QUESTIONS)}: [{question['domain'].upper()}]")
        print(f"Subject: {question['subject']}")
        print("-" * 60)

        result = await run_single_question(agent, question)
        results.append(result)

        # Print summary
        if result["error"]:
            print(f"  ERROR: {result['error']['type']}: {result['error']['message']}")
        else:
            pipeline = result["steps"]["pipeline"]
            print(f"  Tool generation: {result['steps']['tool_generation']['tool_count']} tools")
            print(f"  Registration: {result['steps']['registration']['registered_count']} tools")
            if pipeline.get("success"):
                print(f"  Pipeline: SUCCESS")
                print(f"  Answer probability: {pipeline.get('answer')}")
                print(f"  Steps used: {pipeline.get('steps_used')}")
                print(f"  Tokens used: {pipeline.get('tokens_used')}")
            else:
                print(f"  Pipeline: FAILED - {pipeline.get('error')}")

        print(f"  Duration: {result['duration_seconds']:.1f}s")
        print()

        # Wait between questions (except after the last one)
        if i < len(QUESTIONS) - 1:
            print(f"  Pacing: {PACING_SECONDS}s...")
            await asyncio.sleep(PACING_SECONDS)

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {OUTPUT_FILE}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if not r["error"])
    print(f"Questions: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")

    for r in results:
        domain = r["domain"]
        if r["error"]:
            print(f"  [{domain}] FAILED: {r['error']['type']} - {r['error']['message']}")
        else:
            pipeline = r["steps"]["pipeline"]
            prob = pipeline.get("answer", "N/A")
            print(f"  [{domain}] OK: probability={prob}, steps={pipeline.get('steps_used')}")

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
