"""Tests for the credibility evaluation harness."""

from forecast_agent.evals.credibility_eval import evaluate_credibility_on_groups


def test_credibility_evaluator_runs_on_synthetic_groups():
    """Evaluator runs on synthetic groups with different history lengths."""
    groups = {
        "group-a": [1, 2, 3, 4, 5],
        "group-b": [2, 4, 6, 8],
        "group-c": [0, 1],
    }
    result = evaluate_credibility_on_groups(groups)
    assert result is not None
    assert result.blended_mean >= 0
    assert result.raw_mean >= 0
    assert result.credibility >= 0
    assert result.improved in (True, False)
