import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleAggregate:
    median: float
    trimmed_mean: float
    logit_mean: float
    spread: float
    spread_ratio: float


def ensemble_aggregate(
    probabilities: Sequence[float],
    trim_fraction: float = 0.1,
    epsilon: float = 1e-15,
) -> EnsembleAggregate:
    """Aggregate probabilities using median, trimmed, and logit means."""
    values = [float(probability) for probability in probabilities]
    if not values:
        raise ValueError("probabilities must not be empty")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("probabilities must be finite and between 0 and 1")
    if not math.isfinite(trim_fraction) or not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be between 0 and 0.5")
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")

    ordered = sorted(values)
    trim_count = int(len(ordered) * trim_fraction)
    trimmed = ordered[trim_count:-trim_count] if trim_count else ordered
    trimmed_mean = math.fsum(trimmed) / len(trimmed)

    if all(value == 0 for value in values):
        logit_mean = 0.0
    elif all(value == 1 for value in values):
        logit_mean = 1.0
    else:
        clipped = [min(max(value, epsilon), 1 - epsilon) for value in values]
        mean_logit = math.fsum(
            math.log(probability / (1 - probability)) for probability in clipped
        ) / len(clipped)
        if mean_logit >= 0:
            logit_mean = 1 / (1 + math.exp(-mean_logit))
        else:
            exp_logit = math.exp(mean_logit)
            logit_mean = exp_logit / (1 + exp_logit)

    spread = ordered[-1] - ordered[0]
    mean_probability = math.fsum(values) / len(values)
    spread_ratio = spread / mean_probability if mean_probability else 0.0
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return EnsembleAggregate(
        median=median,
        trimmed_mean=trimmed_mean,
        logit_mean=logit_mean,
        spread=spread,
        spread_ratio=spread_ratio,
    )
