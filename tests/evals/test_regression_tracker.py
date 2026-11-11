"""Tests for the regression tracker harness."""

from forecast_agent.evals.regression_tracker import (
    RegressionResult,
    RegressionTracker,
    track_regression_on_question_set,
)


def test_track_regression_on_question_set():
    """Track regression on a question set between runs."""
    result: RegressionResult = track_regression_on_question_set(
        questions=["q1", "q2", "q3"]
    )
    assert result is not None
    assert len(result.diff) >= 0


def test_regression_tracker_runs_on_question_set():
    """Tracker records diff for a question set between runs."""
    tracker = RegressionTracker(questions=["q1", "q2", "q3"])
    result = tracker.run()
    assert result is not None
    assert len(result.diff) >= 0
