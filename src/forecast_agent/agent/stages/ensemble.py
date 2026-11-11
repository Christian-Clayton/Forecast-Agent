"""Stage 6: ensemble.

Runs N independent ensemble members (differing in model, prompt framing,
and seed) and aggregates the results. If the spread ratio exceeds the
configured threshold, switches to Toolkit B.
"""

from forecast_agent.agent.stages.schemas import (
    EnsembleOutput,
)
from forecast_agent.models import EnsembleMember

# Default ensemble parameters.
_DEFAULT_MEMBERS = 3
_DEFAULT_SEED = 42
_DEFAULT_SPREAD_RATIO = 2.0


def build_ensemble_prompt(
    question: str,
    domain_class: str,
    base_rate: float | None,
    tool_call_ids: list[str],
) -> str:
    """Return the system prompt and user message for the ensemble stage."""
    return (
        "Run N independent ensemble members and aggregate the results.\n"
        "Output ONLY JSON matching the EnsembleOutput schema.\n"
        "Rules:\n"
        "  - Run at least 3 members, differing in model, prompt framing, "
        "and seed.\n"
        "  - Never invent a number; every number must come from a tool call.\n"
        "  - Report the ensemble spread explicitly.\n"
        f"  - Question: {question}\n"
        f"  - Domain: {domain_class}\n"
        f"  - Base rate: {base_rate}\n"
        f"  - Tool call IDs: {tool_call_ids}"
    )


def ensemble_aggregate(
    probabilities: list[float],
    spread_ratio: float = _DEFAULT_SPREAD_RATIO,
) -> EnsembleOutput:
    """Aggregate ensemble members and check the spread rule.

    Returns the aggregated result and a flag if the spread rule switches to
    Toolkit B.
    """
    if not probabilities:
        raise ValueError("no probabilities to aggregate")
    median = sorted(probabilities)[len(probabilities) // 2]
    trimmed = sorted(probabilities)[
        len(probabilities) // 4 : 3 * len(probabilities) // 4
    ]
    trimmed_mean = sum(trimmed) / len(trimmed) if trimmed else median
    logit_mean = sum(1 / (1 + __import__("math").exp(-p)) for p in probabilities) / len(
        probabilities
    )
    spread = max(abs(p - median) for p in probabilities)
    spread_ratio_value = spread / median if median else 0
    switch_to_toolkit_b = spread_ratio_value > spread_ratio
    return EnsembleOutput(
        members=[
            EnsembleMember(
                model_id=f"m{i}",
                prompt_id=f"p{i}",
                probability=p,
            )
            for i, p in enumerate(probabilities)
        ],
        median=median,
        trimmed_mean=trimmed_mean,
        logit_mean=logit_mean,
        spread=spread,
        spread_ratio=spread_ratio_value,
        switch_to_toolkit_b=switch_to_toolkit_b,
    )


def validate_ensemble(output: dict) -> EnsembleOutput:
    """Validate ensemble output."""
    return EnsembleOutput(**output)
