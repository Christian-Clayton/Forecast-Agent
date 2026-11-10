import math
from collections.abc import Sequence
from dataclasses import dataclass

from forecast_agent.scoring.rules import _validate_binary_outcome, _validate_probability


@dataclass(frozen=True)
class CalibrationBin:
    lower_bound: float
    upper_bound: float
    mean_probability: float
    observed_frequency: float
    count: int


def reliability_bins(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if len(probabilities) == 0:
        raise ValueError("probabilities and outcomes must not be empty")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    grouped: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for probability, outcome in zip(probabilities, outcomes, strict=True):
        _validate_probability(probability)
        observed = _validate_binary_outcome(outcome)
        bin_index = min(int(probability * n_bins), n_bins - 1)
        grouped[bin_index].append((probability, observed))

    return [
        CalibrationBin(
            lower_bound=index / n_bins,
            upper_bound=(index + 1) / n_bins,
            mean_probability=math.fsum(p for p, _ in values) / len(values),
            observed_frequency=math.fsum(y for _, y in values) / len(values),
            count=len(values),
        )
        for index, values in enumerate(grouped)
        if values
    ]


def sharpness(intervals: Sequence[tuple[float, float]]) -> float:
    if len(intervals) == 0:
        raise ValueError("intervals must not be empty")

    widths = []
    for lower, upper in intervals:
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("interval bounds must be finite and ordered")
        widths.append(upper - lower)
    return math.fsum(widths) / len(widths)
