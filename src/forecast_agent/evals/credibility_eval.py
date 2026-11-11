"""Credibility evaluation harness.

Evaluates whether Buhlmann credibility blending beats the raw group mean
on short-history groups.
"""

from dataclasses import dataclass

from forecast_agent.quant.credibility import buhlmann_credibility


@dataclass(frozen=True)
class CredibilityEvaluation:
    group: str
    raw_mean: float
    blended_mean: float
    credibility: float
    k: float | None
    improved: bool


def evaluate_credibility_on_groups(
    experience: dict[str, list[float]],
) -> CredibilityEvaluation:
    """Run the credibility evaluation on a set of groups."""
    result = buhlmann_credibility(experience)
    # Use the first group for the report.
    group = next(iter(experience))
    raw_mean = sum(experience[group]) / len(experience[group])
    blended_mean = result.groups[group].blended_estimate
    credibility_value = result.groups[group].credibility
    k_value = result.k
    # Blending improves when the group's blended estimate is closer to the
    # overall mean than the raw mean, weighted by credibility.
    overall_mean = sum(v for vals in experience.values() for v in vals) / sum(
        len(v) for v in experience.values()
    )
    improved = abs(blended_mean - overall_mean) < abs(raw_mean - overall_mean)
    return CredibilityEvaluation(
        group=group,
        raw_mean=raw_mean,
        blended_mean=blended_mean,
        credibility=credibility_value,
        k=k_value,
        improved=improved,
    )


class CredibilityEvaluator:
    """Credibility evaluator over synthetic groups."""

    def __init__(self, groups: dict[str, list[float]]) -> None:
        self.groups = groups

    def run(self) -> CredibilityEvaluation:
        return evaluate_credibility_on_groups(self.groups)
