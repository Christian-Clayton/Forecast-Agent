import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_distribution


@dataclass(frozen=True)
class FrequencyParameters:
    distribution: Literal["poisson", "negative_binomial"]
    rate: float
    dispersion: float


@dataclass(frozen=True)
class SeverityParameters:
    distribution: Literal["gamma", "lognormal"]
    shape: float
    scale: float


@dataclass(frozen=True)
class FrequencySeverityFit:
    frequency: FrequencyParameters
    severity: SeverityParameters
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class AggregateLossResult:
    quantiles: dict[str, float]
    mean: float
    seed: int
    n_sims: int


def fit_frequency_severity(
    event_table: pd.DataFrame,
    exposure_column: str,
    frequency_column: str,
    severity_column: str,
    frequency_distribution: Literal["poisson", "negative_binomial"] = "poisson",
    severity_distribution: Literal["gamma", "lognormal"] = "lognormal",
) -> FrequencySeverityFit:
    """Fit counts per exposure cell and positive severity observations."""
    required = {exposure_column, frequency_column, severity_column}
    missing = required - set(event_table.columns)
    if missing:
        raise ValueError(f"event table is missing columns: {sorted(missing)}")
    if event_table.empty:
        raise ValueError("event table must not be empty")

    exposures = event_table[exposure_column].to_numpy(dtype=float)
    counts = event_table[frequency_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(exposures)) or np.any(exposures <= 0):
        raise ValueError("exposures must be finite and positive")
    if (
        not np.all(np.isfinite(counts))
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError("frequency values must be finite nonnegative integers")

    total_exposure = float(np.sum(exposures))
    total_count = float(np.sum(counts))
    rate = total_count / total_exposure
    dispersion = 0.0
    if frequency_distribution == "negative_binomial" and rate > 0:
        expected = exposures * rate
        denominator = float(np.sum(expected**2))
        if denominator > 0:
            excess_variance = float(np.sum((counts - expected) ** 2 - expected))
            dispersion = max(0.0, excess_variance / denominator)
    elif frequency_distribution != "poisson":
        raise ValueError(
            f"unsupported frequency distribution: {frequency_distribution}"
        )

    severity_observations = []
    for observation in event_table[severity_column].dropna():
        if isinstance(observation, (list, tuple, np.ndarray)):
            severity_observations.extend(float(value) for value in observation)
        else:
            severity_observations.append(float(observation))
    severities = np.asarray(severity_observations, dtype=float)
    if severities.size == 0 or not np.all(np.isfinite(severities)):
        raise ValueError("severity column must contain finite observations")
    if np.any(severities <= 0):
        raise ValueError("severity observations must be positive")

    if severity_distribution == "gamma":
        shape, _, scale = gamma_distribution.fit(severities, floc=0)
        shape = float(shape)
        scale = float(scale)
    elif severity_distribution == "lognormal":
        log_severities = np.log(severities)
        shape = float(np.std(log_severities, ddof=0))
        scale = float(np.exp(np.mean(log_severities)))
    else:
        raise ValueError(f"unsupported severity distribution: {severity_distribution}")

    return FrequencySeverityFit(
        frequency=FrequencyParameters(
            distribution=frequency_distribution,
            rate=rate,
            dispersion=dispersion,
        ),
        severity=SeverityParameters(
            distribution=severity_distribution,
            shape=shape,
            scale=scale,
        ),
        diagnostics={
            "total_count": total_count,
            "total_exposure": total_exposure,
            "severity_observations": float(severities.size),
        },
    )


def aggregate_loss_mc(
    frequency: FrequencyParameters,
    severity: SeverityParameters,
    horizon: float,
    n_sims: int,
    seed: int,
) -> AggregateLossResult:
    """Simulate aggregate loss using the supplied distributions and seed."""
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    if n_sims < 1:
        raise ValueError("n_sims must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if frequency.distribution not in ("poisson", "negative_binomial"):
        raise ValueError(
            f"unsupported frequency distribution: {frequency.distribution}"
        )
    if severity.distribution not in ("gamma", "lognormal"):
        raise ValueError(
            f"unsupported severity distribution: {severity.distribution}"
        )
    if (
        not math.isfinite(frequency.rate)
        or frequency.rate < 0
        or not math.isfinite(frequency.dispersion)
        or frequency.dispersion < 0
    ):
        raise ValueError("frequency parameters must be finite and nonnegative")
    if (
        not math.isfinite(severity.shape)
        or severity.shape < 0
        or not math.isfinite(severity.scale)
        or severity.scale <= 0
        or (severity.distribution == "gamma" and severity.shape == 0)
    ):
        raise ValueError("severity parameters must be finite with positive scale")

    rng = np.random.default_rng(seed)
    expected_count = frequency.rate * horizon
    if not math.isfinite(expected_count):
        raise ValueError("expected event count must be finite")
    total_losses = np.zeros(n_sims, dtype=float)
    chunk_size = max(1, min(10_000, int(1_000_000 / max(expected_count, 1))))
    for start in range(0, n_sims, chunk_size):
        stop = min(start + chunk_size, n_sims)
        size = stop - start
        if (
            frequency.distribution == "negative_binomial"
            and frequency.dispersion > 1e-12
            and expected_count > 0
        ):
            shape = 1 / frequency.dispersion
            probability = shape / (shape + expected_count)
            counts = rng.negative_binomial(shape, probability, size=size)
        elif frequency.distribution in ("poisson", "negative_binomial"):
            counts = rng.poisson(expected_count, size=size)
        else:
            raise ValueError(
                f"unsupported frequency distribution: {frequency.distribution}"
            )

        event_count = int(np.sum(counts))
        if event_count == 0:
            continue
        if severity.distribution == "gamma":
            losses = rng.gamma(severity.shape, severity.scale, size=event_count)
        elif severity.distribution == "lognormal":
            losses = rng.lognormal(
                mean=math.log(severity.scale),
                sigma=severity.shape,
                size=event_count,
            )
        else:
            raise ValueError(
                f"unsupported severity distribution: {severity.distribution}"
            )
        simulation_indices = np.repeat(np.arange(size), counts)
        total_losses[start:stop] = np.bincount(
            simulation_indices, weights=losses, minlength=size
        )

    q05, q50, q95 = np.quantile(total_losses, [0.05, 0.5, 0.95])
    return AggregateLossResult(
        quantiles={"q05": float(q05), "q50": float(q50), "q95": float(q95)},
        mean=float(np.mean(total_losses)),
        seed=seed,
        n_sims=n_sims,
    )
