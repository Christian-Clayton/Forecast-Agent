"""Tests for the numeric-claim citation validator eval."""

from forecast_agent.evals.citation_validator_eval import CitationValidatorEvaluator


def test_citation_validator_evaluator_runs_on_ledger():
    """Evaluator checks that no uncited numeric claims reach the ledger."""
    evaluator = CitationValidatorEvaluator(ledger_path=":memory:")
    result = evaluator.run()
    # The validator should catch 1 uncited claim and 1 cited claim.
    assert result.uncited_claims == 1
    assert result.validated_claims == 1
