"""Stage 8: finalize.

Emits the actuarial-style output: bottom line first, then collapsed working.
Validates numeric citations before returning.
"""

from forecast_agent.agent.stages.schemas import FinalizeOutput


def build_finalize_prompt(
    falsifiable_text: str,
    domain_class: str,
    toolkit: str,
    probability: float | None,
    quantiles: dict | None,
    tool_call_ids: list[str],
    base_rate_source: str,
    reasoning: str,
) -> str:
    """Return the system prompt and user message for the finalize stage."""
    return (
        "Emit the forecast in actuarial style.\n"
        "Output ONLY JSON matching the FinalizeOutput schema.\n"
        "Rules:\n"
        "  - Bottom line first: the number, then the resolution date.\n"
        "  - Working after, in a collapsed block.\n"
        "  - Every number must have a tool-call citation in square brackets\n"
        "    like [tool:widget-123]. The validator rejects uncited numbers.\n"
        "  - Never invent a number.\n"
        f"  - Question: {falsifiable_text}\n"
        f"  - Domain: {domain_class}\n"
        f"  - Toolkit: {toolkit}\n"
        f"  - Probability: {probability}\n"
        f"  - Quantiles: {quantiles}\n"
        f"  - Tool call IDs: {tool_call_ids}\n"
        f"  - Base rate source: {base_rate_source}\n"
        f"  - Reasoning: {reasoning}"
    )


def validate_finalize(output: dict) -> FinalizeOutput:
    """Validate finalize output."""
    return FinalizeOutput(**output)
