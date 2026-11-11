"""Tests for the calibration report harness."""

from forecast_agent.evals.report_calibration import (
    CalibrationReportBuilder,
)


def test_calibration_report_builds():
    """Report builds from resolved forecasts with Brier and log scores."""
    builder = CalibrationReportBuilder()
    builder.add_forecast(brier=0.25, log_score=0.3, score_is_valid=True)
    builder.add_forecast(brier=0.15, log_score=0.2, score_is_valid=True)
    report = builder.build()
    assert report is not None
    assert report.mean_brier >= 0
    assert report.mean_log_score >= 0
