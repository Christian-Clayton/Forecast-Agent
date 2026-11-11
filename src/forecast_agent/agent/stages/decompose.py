"""Stage 4: decompose.

Breaks the falsifiable question into sub-questions that map to quantitative
tools. Also selects a series and horizon for time-series forecasts.
"""

from forecast_agent.agent.stages.schemas import DecomposeOutput

_AVAILABLE_SERIES = frozenset(
    {
        "cell-00",
        "cell-01",
        "cell-02",
        "cell-03",
        "cell-04",
        "cell-05",
    }
)


def build_decompose_prompt(
    falsifiable_text: str,
    domain_class: str,
) -> str:
    """Return the system prompt and user message for the decompose stage."""
    return (
        "Break the question into sub-questions that map to quantitative "
        "tools.\n"
        "Output ONLY JSON matching the DecomposeOutput schema.\n"
        "Rules:\n"
        "  - Each sub-question must be falsifiable and map to a tool "
        "or a series.\n"
        "  - If a time-series forecast is needed, specify series_id, "
        "start, end, and candidate_model (ets or arima).\n"
        "  - If no series is available, set series_id to null.\n"
        "  - horizon must be a positive integer.\n"
        f"  - Question: {falsifiable_text}\n"
        f"  - Domain: {domain_class}"
    )


def validate_decompose(output: dict) -> DecomposeOutput:
    """Validate decompose output."""
    return DecomposeOutput(**output)
