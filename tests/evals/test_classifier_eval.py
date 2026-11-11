"""Tests for the classifier evaluation harness."""

from forecast_agent.evals.classifier_eval import ClassifierEvaluator


def test_classifier_evaluator_runs_on_planted_series():
    """Evaluator runs on a planted series and reports metrics."""
    evaluator = ClassifierEvaluator(series_id="cell-00")
    result = evaluator.run()
    assert result is not None
    assert result.accuracy >= 0
    assert result.confusion_matrix is not None
