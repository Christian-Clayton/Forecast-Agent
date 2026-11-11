"""Stage 7: critique.

A separate pass that runs a pre-mortem and lists disconfirming hypotheses.
It may lower confidence or widen intervals. It may not invent new numbers.
"""

from forecast_agent.agent.stages.schemas import FinalizeOutput


def build_critique_prompt(
    bottom_line: str,
    working: str,
    tool_call_ids: list[str],
    domain_class: str,
) -> str:
    """Return the system prompt and user message for the critique stage."""
    return (
        "Critique the forecast by running a pre-mortem and listing "
        "disconfirming hypotheses.\\n"
        "Output ONLY JSON matching the FinalizeOutput schema.\\n"
        "Rules:\\n"
        "  - State the most likely way the forecast could be wrong.\\n"
        "  - List disconfirming hypotheses with evidence for and against.\\n"
        "  - You may lower confidence or widen intervals, but you may not "
        "invent new numbers.\\n"
        "  - Every number must have a tool-call citation in square brackets\\n"
        "    like [tool:widget-123].\\n"
        f"  - Bottom line: {bottom_line}\\n"
        f"  - Working: {working}\\n"
        f"  - Tool call IDs: {tool_call_ids}\\n"
        f"  - Domain: {domain_class}"
    )


def validate_critique(output: dict) -> FinalizeOutput:
    """Validate critique output."""
    return FinalizeOutput(**output)
