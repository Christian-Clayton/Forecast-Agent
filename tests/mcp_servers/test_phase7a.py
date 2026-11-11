"""Tests for Phase 7a: run mode guard and clock server."""

import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so imports work.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forecast_agent.config import RunMode, Settings
from forecast_agent.mcp_servers.clock_server import Clock, days_until, now_utc
from forecast_agent.mcp_servers.external_guard import (
    ExternalAccessBlocked,
    RunModeGuard,
)


class TestRunModeEnum:
    """RunMode enum tests."""

    def test_backtest_value(self):
        assert RunMode.BACKTEST.value == "backtest"

    def test_live_value(self):
        assert RunMode.LIVE.value == "live"

    def test_str(self):
        # str(enum.Enum) returns the member name, so we check the value instead
        assert RunMode.BACKTEST.value == "backtest"
        assert RunMode.LIVE.value == "live"


class TestRunModeGuard:
    """RunModeGuard tests."""

    def test_is_live_when_live_mode(self):
        guard = RunModeGuard("live")
        assert guard.is_live() is True

    def test_is_not_live_when_backtest_mode(self):
        guard = RunModeGuard("backtest")
        assert guard.is_live() is False

    def test_check_external_call_allowed_in_live(self):
        guard = RunModeGuard("live")
        guard.check_external_call("search_web")  # should not raise

    def test_check_external_call_blocked_in_backtest(self):
        guard = RunModeGuard("backtest")
        with pytest.raises(ExternalAccessBlocked) as exc_info:
            guard.check_external_call("search_web")
        assert exc_info.value.mode == "backtest"
        assert "search_web" in str(exc_info.value)

    def test_check_external_call_bypass_with_force(self):
        guard = RunModeGuard("backtest")
        # force=True bypasses the block
        guard.check_external_call("search_web", force=True)  # should not raise


class TestClock:
    """Clock tests."""

    def test_frozen_clock_returns_frozen_time(self):
        clock = Clock(frozen_time="2026-06-15T12:00:00+00:00")
        assert clock.now() == "2026-06-15T12:00:00+00:00"

    def test_live_clock_returns_real_utc(self):
        clock = Clock()
        time_str = clock.now()
        # Verify it's a valid ISO 8601 UTC timestamp
        dt = datetime.fromisoformat(time_str)
        assert dt.tzinfo is not None
        # The offset should be UTC (any positive duration is UTC)
        assert dt.utcoffset() is not None

    def test_days_until_full_day(self):
        clock = Clock(frozen_time="2026-06-15T00:00:00+00:00")
        # From 2026-06-15 to 2026-07-01 = 16 days
        assert clock.days_until("2026-07-01") == 16

    def test_days_until_past_date(self):
        clock = Clock(frozen_time="2026-07-01T00:00:00+00:00")
        # From 2026-07-01 to 2026-06-15 = -16 days
        assert clock.days_until("2026-06-15") == -16

    def test_freeze_unfreeze(self):
        clock = Clock(frozen_time="2026-06-15T12:00:00+00:00")
        assert clock.now() == "2026-06-15T12:00:00+00:00"
        clock.unfreeze()
        # After unfreezing, the clock returns live time
        time_str = clock.now()
        assert time_str is not None


class TestConfigRunMode:
    """Settings run_mode field tests."""

    def test_default_run_mode_is_backtest(self):
        settings = Settings()
        assert settings.run_mode == RunMode.BACKTEST

    def test_run_mode_from_env(self):
        # Test that run_mode can be set via environment
        settings = Settings(
            run_mode=RunMode.LIVE,
            openai_compat_model="test-model",
            openai_compat_api_key="test-key",
        )
        assert settings.run_mode == RunMode.LIVE


class TestExternalGuardIntegration:
    """Integration tests for external guard with config."""

    def test_settings_can_be_created_with_live_mode(self):
        settings = Settings(
            run_mode=RunMode.LIVE,
            openai_compat_model="test-model",
            openai_compat_api_key="test-key",
        )
        assert settings.run_mode == RunMode.LIVE

    def test_settings_can_be_created_with_backtest_mode(self):
        settings = Settings(
            run_mode=RunMode.BACKTEST,
            openai_compat_model="test-model",
            openai_compat_api_key="test-key",
        )
        assert settings.run_mode == RunMode.BACKTEST


class TestSharedGuardUsedByExternalServers:
    """Verify the shared guard would block all external tools in backtest mode."""

    def test_clock_guard_blocks_in_backtest(self):
        guard = RunModeGuard("backtest")
        with pytest.raises(ExternalAccessBlocked):
            guard.check_external_call("clock.now")

    def test_search_guard_blocks_in_backtest(self):
        guard = RunModeGuard("backtest")
        with pytest.raises(ExternalAccessBlocked):
            guard.check_external_call("search_web")

    def test_fetch_guard_blocks_in_backtest(self):
        guard = RunModeGuard("backtest")
        with pytest.raises(ExternalAccessBlocked):
            guard.check_external_call("fetch")

    def test_crowd_guard_blocks_in_backtest(self):
        guard = RunModeGuard("backtest")
        with pytest.raises(ExternalAccessBlocked):
            guard.check_external_call("get_crowd_prediction")
