"""Clock server.

Provides UTC time and date arithmetic for the agent pipeline.
The clock is injectable so tests and backtests can freeze it.

Usage:
    from forecast_agent.mcp_servers.clock_server import Clock, create_clock_server

    clock = Clock(frozen_time="2026-06-15T12:00:00+00:00")
    print(clock.now())           # "2026-06-15T12:00:00+00:00"
    print(clock.days_until("2026-07-01"))  # 16

    # Live mode:
    clock = Clock()
    print(clock.now())           # current UTC time
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import final


class Clock:
    """Injectable clock for time operations.

    Attributes:
        frozen_time: ISO 8601 UTC time string, or None for live mode.
    """

    def __init__(self, frozen_time: str | None = None) -> None:
        if frozen_time is not None:
            self._frozen = datetime.fromisoformat(frozen_time)
        else:
            self._frozen = None

    def now(self) -> str:
        """Return current time in ISO 8601 UTC."""
        if self._frozen is not None:
            return self._frozen.isoformat()
        return datetime.now(UTC).isoformat()

    def days_until(self, date_str: str) -> int:
        """Return number of days from now until the given date.

        Args:
            date_str: ISO 8601 date or datetime.

        Returns:
            Days until the date (negative if in the past).
        """
        target = datetime.fromisoformat(date_str)
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        now = self.now()
        now_dt = datetime.fromisoformat(now)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=UTC)
        return (target - now_dt).days

    def freeze(self, time_str: str) -> None:
        """Freeze the clock to a specific UTC ISO 8601 time."""
        self._frozen = datetime.fromisoformat(time_str)

    def unfreeze(self) -> None:
        """Unfreeze the clock (use live time)."""
        self._frozen = None


def create_clock_server() -> Clock:
    """Create a clock instance.

    The returned clock is live (no frozen time).
    """
    return Clock()


# Default clock instance for use by the pipeline.
_default_clock = Clock()


def set_default_clock(clock: Clock) -> None:
    """Set the default clock instance used by the pipeline."""
    global _default_clock
    _default_clock = clock


def get_default_clock() -> Clock:
    """Get the default clock instance."""
    return _default_clock


def now_utc() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Uses the default clock (live or frozen).
    """
    return _default_clock.now()


def days_until(date_str: str) -> int:
    """Return days until the given date using the default clock."""
    return _default_clock.days_until(date_str)
