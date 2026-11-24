import asyncio
import sys
sys.path.insert(0, "src")

from forecast_agent.llm.multi_provider import MultiProvider


class SimProvider:
    """Simulates a provider that fails N times then succeeds (rate-limit test)."""

    def __init__(self, name, models, fail_after=1):
        self.name = name
        self.models = models
        self.fail_after = fail_after
        self.calls = 0

    async def complete(self, messages, *, model, max_tokens, temperature=0.0):
        self.calls += 1
        if self.calls <= self.fail_after:
            raise RuntimeError(f"{self.name} simulated 429 rate limit on model {model}")
        return type(
            "C",
            (),
            {
                "content": '{"probability":0.5}',
                "model": model,
                "input_tokens": 10,
                "output_tokens": 2,
                "cost_usd": 0.0,
            },
        )


async def main():
    # Build a MultiProvider across Gemini models + an OpenRouter-style provider.
    # First provider (gemini-a) fails twice then succeeds; second (gemini-b)
    # succeeds on first call. This tests per-model rotation AND provider fallback.
    providers = [
        (
            "gemini-a",
            SimProvider("gemini-a", ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"], fail_after=2),
            ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        ),
        (
            "gemini-b",
            SimProvider("gemini-b", ["gemini-3.5-flash"], fail_after=0),
            ["gemini-3.5-flash"],
        ),
    ]
    p = MultiProvider(providers)

    r = await p.complete([], model="gemini-3.5-flash-lite", max_tokens=200)
    print("RESULT:", r.model)
    print("ATTEMPTS:", p.attempts)
    print("LOG:")
    for a in p.log:
        print(
            f"  {a.provider:12s} {a.model:25s} {a.status:7s} {a.timing_ms:6.1f}ms "
            f"{('err='+str(a.error)[:45]) if a.error else ''}"
        )

    # Test: single provider, one model - fallback within same provider
    # Test: single provider, one model - fallback within same provider
    p2 = MultiProvider([("gemini", SimProvider("gemini", ["gemini-3.5-flash-lite"], fail_after=0), ["gemini-3.5-flash-lite"])])
    r2 = await p2.complete([], model="gemini-3.5-flash-lite", max_tokens=200)
    print("\nSINGLE-PROVIDER FALLBACK:")
    print("  RESULT model:", r2.model, "| ATTEMPTS:", p2.attempts)

asyncio.run(main())
