"""Stage 5: model.

Runs the quantitative tools. Always runs the baseline first; a complex model
is used only if it beats the baseline on walk-forward RMSE.
"""

from forecast_agent.agent.stages.schemas import ModelOutput


def build_model_prompt(
    falsifiable_text: str,
    domain_class: str,
    base_rate: float | None,
    base_rate_source: str,
    decomposition: dict,
    baseline_result: dict | None,
    candidate_score: dict | None,
) -> str:
    """Return the system prompt and user message for the model stage."""
    return (
        "Run the quantitative tools for this question.\n"
        "Output ONLY JSON matching the ModelOutput schema.\n"
        "Rules:\n"
        "  - Binary questions return probability only.\n"
        "  - Quantile questions return the full quantile set.\n"
        "  - Always run the baseline first; a complex model is used only "
        "if it beats the baseline on walk-forward RMSE.\n"
        "  - Never invent a number; every number must come from a tool call.\n"
        "  - State the tool call IDs in the reasoning.\n"
        f"  - Question: {falsifiable_text}\n"
        f"  - Domain: {domain_class}\n"
        f"  - Base rate: {base_rate} (source: {base_rate_source})\n"
        f"  - Decomposition: {decomposition}\n"
        f"  - Baseline result: {baseline_result}\n"
        f"  - Candidate score: {candidate_score}"
    )


def validate_model(output: dict) -> ModelOutput:
    """Validate model output."""
    return ModelOutput(**output)
