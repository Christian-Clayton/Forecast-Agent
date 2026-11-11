"""Stage 3: base rate.

Selects a reference class and resolves a base rate. If no trusted source is
available, records the rate as unverified_prior and flags it in the ledger.
"""

from forecast_agent.agent.stages.schemas import BaseRateOutput

_REFERENCE_CLASS_KEYWORDS = frozenset(
    {
        "telecom",
        "software",
        "cloud",
        "industrial",
        "retail",
        "finance",
        "energy",
        "healthcare",
    }
)


def build_base_rate_prompt(
    falsifiable_text: str,
    question_type: str,
) -> str:
    """Return the system prompt and user message for the base_rate stage."""
    return (
        "Select a reference class and resolve a base rate for this question.\n"
        "Output ONLY JSON matching the BaseRateOutput schema.\n"
        "Rules:\n"
        "  - If a trusted source exists, cite it and give the base rate.\n"
        "  - If no trusted source exists, return unverified_prior=0.5 "
        "and explain that the prior is unverified.\n"
        "  - reference_class_description should name the reference class "
        "or be null.\n"
        f"  - Question: {falsifiable_text}\n"
        f"  - Type: {question_type}"
    )


def resolve_reference_class(
    description: str | None,
    domain_class: str,
) -> tuple[str | None, float | None, str]:
    """Resolve a base rate from a reference class description."""
    if description is None:
        return None, None, "unverified_prior"
    text = description.lower()
    for keyword, ref in _REFERENCE_CLASS_KEYWORDS.items():
        if keyword in text:
            return ref, 0.5, f"synthetic:{keyword}"
    return None, None, "unverified_prior"


def validate_base_rate(output: dict) -> BaseRateOutput:
    """Validate base_rate output."""
    return BaseRateOutput(**output)
