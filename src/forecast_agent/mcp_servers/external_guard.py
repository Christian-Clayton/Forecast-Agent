"""Shared run-mode guard for external MCP servers.

Every external server (clock, search, fetch, platform) uses this guard.
In BACKTEST mode, all external calls are blocked before any network activity.
"""

from __future__ import annotations

from typing import final


class ExternalAccessBlocked(Exception):
    """Raised when an external call is attempted in backtest mode."""

    def __init__(self, message: str, mode: str) -> None:
        self.message = message
        self.mode = mode
        super().__init__(message)

    def __str__(self) -> str:
        return f"External access blocked in mode={self.mode}: {self.message}"


@final
class RunModeGuard:
    """Shared guard that blocks external calls in backtest mode."""

    def __init__(self, run_mode: str) -> None:
        self._mode = run_mode

    @property
    def mode(self) -> str:
        return self._mode

    def is_live(self) -> bool:
        return self._mode == "live"

    def check_external_call(
        self,
        tool_name: str,
        *,
        force: bool = False,
    ) -> None:
        """Raise ExternalAccessBlocked if in backtest mode and not forced.

        Args:
            tool_name: Name of the tool being called (for error messages).
            force: If True, bypass the block (for internal use only).
        """
        if force or self._mode == "live":
            return
        raise ExternalAccessBlocked(
            message=(
                f"External call to '{tool_name}' blocked in backtest mode. "
                f"External content is only available in LIVE mode. "
                f"Set RUN_MODE=live to enable."
            ),
            mode=self._mode,
        )
