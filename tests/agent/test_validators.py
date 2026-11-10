import pytest

from forecast_agent.agent.validators import (
    NumericCitationError,
    validate_numeric_citations,
)


def test_numeric_claims_must_cite_a_tool_call_id() -> None:
    validate_numeric_citations(
        "The probability is 0.6 [tool:call-123].",
        valid_tool_call_ids={"call-123"},
    )

    with pytest.raises(NumericCitationError, match="uncited numeric claim"):
        validate_numeric_citations(
            "The probability is 0.6.",
            valid_tool_call_ids={"call-123"},
        )


def test_unknown_tool_call_ids_do_not_validate_numeric_claims() -> None:
    with pytest.raises(NumericCitationError, match="uncited numeric claim"):
        validate_numeric_citations(
            "The probability is 0.6 [tool:invented].",
            valid_tool_call_ids={"call-123"},
        )


def test_text_without_numeric_claims_needs_no_citation() -> None:
    validate_numeric_citations(
        "The available evidence is insufficient.",
        valid_tool_call_ids=set(),
    )
