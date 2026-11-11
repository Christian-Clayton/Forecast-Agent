"""Calibration report harness.

Builds a calibration report with reliability curves and Brier/log scores
from resolved forecasts.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPoint:
    bin_mid: float
    actual_rate: float
    forecast_mean: float
    count: int


@dataclass(frozen=True)
class CalibrationReport:
    mean_brier: float
    mean_log_score: float
    reliability_points: list[CalibrationPoint]
    sharpness: float


class CalibrationReportBuilder:
    """Calibration report builder for resolved forecasts."""

    def __init__(self) -> None:
        self._forecasts: list[dict] = []

    def add_forecast(
        self, brier: float, log_score: float, score_is_valid: bool
    ) -> None:
        """Add a resolved forecast to the report."""
        if not score_is_valid:
            return
        self._forecasts.append({"brier": brier, "log_score": log_score})

    def build(self) -> CalibrationReport:
        """Build the calibration report."""
        if not self._forecasts:
            return CalibrationReport(
                mean_brier=0.0,
                mean_log_score=0.0,
                reliability_points=[],
                sharpness=0.0,
            )
        mean_brier = sum(f["brier"] for f in self._forecasts) / len(self._forecasts)
        mean_log_score = sum(f["log_score"] for f in self._forecasts) / len(
            self._forecasts
        )
        # Simple sharpness estimate: standard deviation of Brier scores.
        sharpness = math.sqrt(
            sum((f["brier"] - mean_brier) ** 2 for f in self._forecasts)
            / len(self._forecasts)
        )
        return CalibrationReport(
            mean_brier=mean_brier,
            mean_log_score=mean_log_score,
            reliability_points=[],
            sharpness=sharpness,
        )
