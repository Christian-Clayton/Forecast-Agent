import math
from collections.abc import Sequence


def _validate_probability(probability: float) -> None:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and between 0 and 1")


def _validate_binary_outcome(outcome: bool | int) -> int:
    if outcome not in (False, True, 0, 1):
        raise ValueError("outcome must be binary")
    return int(outcome)


def brier_score(probability: float, outcome: bool | int) -> float:
    _validate_probability(probability)
    observed = _validate_binary_outcome(outcome)
    return (probability - observed) ** 2


def log_score(
    probability: float, outcome: bool | int, epsilon: float = 1e-15
) -> float:
    """Return clipped negative log likelihood to keep zero-probability cases finite."""
    _validate_probability(probability)
    observed = _validate_binary_outcome(outcome)
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and between 0 and 0.5")
    likelihood = probability if observed else 1 - probability
    return -math.log(max(likelihood, epsilon))


def pinball_loss(actual: float, forecast: float, quantile: float) -> float:
    if not all(math.isfinite(value) for value in (actual, forecast, quantile)):
        raise ValueError("actual, forecast, and quantile must be finite")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    error = actual - forecast
    return quantile * error if error >= 0 else (quantile - 1) * error


def crps_approx(samples: Sequence[float], observation: float) -> float:
    if len(samples) == 0:
        raise ValueError("samples must not be empty")
    if not math.isfinite(observation) or not all(
        math.isfinite(sample) for sample in samples
    ):
        raise ValueError("samples and observation must be finite")

    ordered = sorted(samples)
    sample_count = len(ordered)
    first_term = math.fsum(abs(sample - observation) for sample in ordered) / (
        sample_count
    )
    pairwise_half_term = math.fsum(
        (2 * index - sample_count + 1) * sample
        for index, sample in enumerate(ordered)
    ) / (sample_count**2)
    return first_term - pairwise_half_term
