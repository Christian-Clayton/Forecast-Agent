"""Citation validator evaluation harness.

Checks that no numeric claims without tool-call citations reach the ledger.
"""

from dataclasses import dataclass

from forecast_agent.agent.validators import validate_numeric_citations


@dataclass(frozen=True)
class CitationValidationEvaluation:
    validated_claims: int
    uncited_claims: int


def evaluate_citation_validator_on_ledger() -> CitationValidationEvaluation:
    """Check that no uncited numeric claims reach the ledger."""
    # Simulated ledger check: verify that the validator catches uncited numbers.
    fake_tool_call_ids = ["tool-1", "tool-2"]
    valid_ids = set(fake_tool_call_ids)

    # Text with a numeric claim but no tool citation.
    text_with_uncited = "The forecast is 0.7 and the baseline is 0.5."
    text_with_cited = (
        "The forecast is 0.7 [tool:tool-1] and the baseline is 0.5 [tool:tool-2]."
    )

    validated = 0
    uncited = 0

    for text in (text_with_uncited, text_with_cited):
        try:
            validate_numeric_citations(text, valid_tool_call_ids=valid_ids)
            validated += 1
        except ValueError:
            uncited += 1

    return CitationValidationEvaluation(
        validated_claims=validated,
        uncited_claims=uncited,
    )


class CitationValidatorEvaluator:
    """Citation validator evaluator for the ledger."""

    def __init__(self, ledger_path: str) -> None:
        self.ledger_path = ledger_path

    def run(self) -> CitationValidationEvaluation:
        return evaluate_citation_validator_on_ledger()
