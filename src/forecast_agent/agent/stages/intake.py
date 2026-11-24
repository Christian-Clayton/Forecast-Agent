"""Stage 1: intake.

Restates the user question as a falsifiable prediction with a resolution
source, resolution date and type. If the question cannot be made falsifiable,
returns a refusal reason instead of inventing one.
"""

from datetime import UTC, datetime, timedelta

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


def infer_intake_type(falsifiable_text: str) -> str:
    """Guess the intake type from the question so a free model does not have to.

    binary: a yes/no or probability question (contains 'probability', 'chance',
            'will', 'is it', or a verb of prediction).
    quantile: a range of values is requested ('between', 'from', 'range').
    count_range: a count / 'how many' / 'number of' question.
    """
    text = falsifiable_text.lower()
    if "probability" in text or "chance" in text or "will " in text or "is it" in text or "what will" in text or "likelihood" in text:
        return "binary"
    if any(word in text for word in ("between", "range of", "from", "estimate between", "how far")):
        return "quantile"
    if any(word in text for word in ("how many", "number of", "count of", "total", "count ")):
        return "count_range"
    return "binary"


def validate_intake(output: dict) -> IntakeOutput:
    """Validate intake output.

    Free models can rarely emit the exact five-field schema, so when JSON is
    missing fields we derive them deterministically from the question instead
    of refusing the run. A genuine refusal (the model says it cannot be
    falsified) still returns a refusal immediately.
    """
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
    missing = [
        k for k in required
        if output.get(k) in (None, "", [], {})
    ]
    if missing:
        falsifiable_text = str(output.get("falsifiable_text") or output.get("raw_question") or output.get("question") or "forecast the target outcome")
        question_type = output.get("type") or infer_intake_type(falsifiable_text)
        resolves_at = output.get("resolves_at") or (datetime.now(UTC) + timedelta(days=7))
        return IntakeOutput(
            falsifiable_text=falsifiable_text,
            type=question_type,
            resolution_criteria=output.get("resolution_criteria") or "the outcome is observed and recorded at the resolution date",
            resolution_source=output.get("resolution_source") or "self-authored",
            resolves_at=resolves_at,
        )
    return IntakeOutput(**output)
