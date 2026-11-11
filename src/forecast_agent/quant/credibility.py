import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CredibilityEstimate:
    count: int
    raw_mean: float
    credibility: float
    blended_estimate: float


@dataclass(frozen=True)
class CredibilityResult:
    k: float | None
    epv: float
    vhm: float
    groups: dict[str, CredibilityEstimate]


def buhlmann_credibility(
    experience: Mapping[str, Sequence[float]],
) -> CredibilityResult:
    """Estimate Bühlmann credibility with unequal group sizes by moments."""
    if len(experience) < 2:
        raise ValueError("at least two groups are required")

    observations: dict[str, list[float]] = {}
    for group, values in experience.items():
        group_values = [float(value) for value in values]
        if not group or len(group_values) < 2:
            raise ValueError("each named group needs at least two observations")
        if not all(math.isfinite(value) for value in group_values):
            raise ValueError("experience values must be finite")
        observations[group] = group_values

    means = {
        group: math.fsum(values) / len(values) for group, values in observations.items()
    }
    total_count = sum(map(len, observations.values()))
    overall_mean = (
        math.fsum(math.fsum(values) for values in observations.values()) / total_count
    )

    within_sum_squares = math.fsum(
        (value - means[group]) ** 2
        for group, values in observations.items()
        for value in values
    )
    degrees_of_freedom = total_count - len(observations)
    epv = within_sum_squares / degrees_of_freedom

    weighted_between_sum_squares = math.fsum(
        len(observations[group]) * (mean - overall_mean) ** 2
        for group, mean in means.items()
    )
    between_degrees_of_freedom = (
        total_count
        - math.fsum(len(values) ** 2 for values in observations.values()) / total_count
    )
    vhm = max(
        0.0,
        (weighted_between_sum_squares - epv * (len(observations) - 1))
        / between_degrees_of_freedom,
    )
    k = epv / vhm if vhm > 0 else None

    group_estimates = {}
    for group, values in observations.items():
        credibility = len(values) / (len(values) + k) if k is not None else 0.0
        raw_mean = means[group]
        group_estimates[group] = CredibilityEstimate(
            count=len(values),
            raw_mean=raw_mean,
            credibility=credibility,
            blended_estimate=credibility * raw_mean + (1 - credibility) * overall_mean,
        )

    return CredibilityResult(k=k, epv=epv, vhm=vhm, groups=group_estimates)
