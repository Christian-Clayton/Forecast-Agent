"""Scheduler for the live ledger.

Periodically runs the resolver to score resolved forecasts and keep the
live ledger up to date.
"""

import time
from pathlib import Path

from forecast_agent.ledger.resolver import run_resolver


def run_once(ledger_path: str | Path = "forecast-ledger.sqlite3") -> None:
    """Run the resolver once (script entry point)."""
    result = run_resolver(ledger_path)
    print(
        f"[scheduler] Resolved {result['resolved']} questions, "
        f"scored {result['scored']}"
    )


def start_loop(
    ledger_path: str | Path = "forecast-ledger.sqlite3", interval_seconds: int = 300
) -> None:
    """Start the scheduler loop.

    Polls every second, running the resolver whenever the interval elapses.
    """
    last_run = time.time()
    print(f"[scheduler] Starting loop (interval: {interval_seconds}s)")
    while True:
        now = time.time()
        if now - last_run >= interval_seconds:
            result = run_resolver(ledger_path)
            print(
                f"[scheduler] Resolved {result['resolved']} questions, "
                f"scored {result['scored']}"
            )
            last_run = now
        time.sleep(1)
