"""Tests for the leakage evaluation harness."""

from forecast_agent.evals.leakage_eval import (
    LeakageEvaluator,
    evaluate_leakage_on_store,
)


def test_leakage_evaluator_runs_on_data_store():
    """Evaluator checks as_of gating on the data store."""
    evaluator = LeakageEvaluator(store_id="synthetic")
    result = evaluator.run()
    assert result is not None
    assert result.total_attempts > 0
    assert result.blocked_attempts == result.total_attempts
