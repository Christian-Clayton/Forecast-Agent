"""Stage 1: intake.

Restates the user question as a falsifiable prediction with a resolution
source, resolution date and type. If the question cannot be made falsifiable,
returns a refusal reason instead of inventing one.
"""

from forecast_agent.agent.stages.schemas import IntakeOutput

_TRUSTED_SOURCES = frozenset(
    {
        "metaculus",
        "good judgment open",
        "metaculus community",
        "self-authored",
    }
)

# Recognised resolution-date patterns.
_DATE_PATTERNS = (
    "2030-01-01",
    "2026-12-31",
    "2027-06-30",
    "2028-01-01",
    "2029-12-31",
)


def build_intake_prompt(raw_question: str, as_of: str) -> str:
    """Return the system prompt and user message for the intake stage."""
    return (
        "You restate a vague question as a single falsifiable prediction.\n"
        "Output ONLY JSON matching the IntakeOutput schema.\n"
        "Rules:\n"
        "  - If the question lacks a resolution source, date, or target, "
        "return refusal_reason instead of inventing them.\n"
        "  - Never fabricate a data source, base rate, or citation.\n"
        "  - falsifiable_text must be a yes/no, probability, or range "
        "question with an observable outcome.\n"
        "  - resolution_source must be one of: "
        + ", ".join(sorted(_TRUSTED_SOURCES))
        + ".\n"
    ) + ("\nUser input: " + raw_question + " (as of: " + as_of + ")")


def validate_intake(output: dict) -> IntakeOutput:
    """Validate intake output; return a refusal if the question is not falsifiable."""
    refusal_reason = output.get("refusal_reason")
    if refusal_reason:
        return IntakeOutput(refusal_reason=refusal_reason)
    required = (
        "falsifiable_text",
        "type",
        "resolution_criteria",
        "resolution_source",
        "resolves_at",
    )
    missing = [k for k in required if output.get(k) in (None, "", [])]
    if missing:
        return IntakeOutput(
            refusal_reason=f"intake incomplete: missing {', '.join(missing)}"
        )
    resolves_at = output.get("resolves_at")
    if not isinstance(resolves_at, str) or not resolves_at:
        return IntakeOutput(refusal_reason="intake incomplete: resolves_at")
    return IntakeOutput(**output)
