import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto, kstest


@dataclass(frozen=True)
class EVTResult:
    threshold: float
    shape: float
    scale: float
    exceedances: int
    return_levels: dict[float, float]
    diagnostics: dict[str, float]
    warnings: list[str]


def evt_pot_fit(
    series: Sequence[float],
    threshold: float,
    return_periods: Sequence[float],
) -> EVTResult:
    """Fit a GPD to threshold excesses and estimate observation-based levels."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("series must contain finite one-dimensional values")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    periods = [float(period) for period in return_periods]
    if not periods or not all(
        math.isfinite(period) and period > 0 for period in periods
    ):
        raise ValueError("return periods must be finite and positive")

    excesses = values[values > threshold] - threshold
    if excesses.size < 3:
        raise ValueError("at least three threshold exceedances are required")
    shape, _, scale = genpareto.fit(excesses, floc=0)
    shape = float(shape)
    scale = float(scale)
    exceedance_rate = excesses.size / values.size
    return_levels = {}
    for period in periods:
        exceedance_probability = 1 - 1 / (period * exceedance_rate)
        if not 0 < exceedance_probability < 1:
            raise ValueError(
                "each return period must exceed the reciprocal exceedance rate"
            )
        return_level = float(
            threshold
            + genpareto.ppf(
                exceedance_probability,
                c=shape,
                loc=0,
                scale=scale,
            )
        )
        if not math.isfinite(return_level):
            raise ValueError("return level is not finite for the requested period")
        return_levels[period] = return_level

    goodness_of_fit = kstest(excesses, genpareto.cdf, args=(shape, 0, scale))
    warnings = []
    if excesses.size < 50:
        warnings.append("fewer than 50 exceedances; tail estimates may be unstable")
    return EVTResult(
        threshold=threshold,
        shape=shape,
        scale=scale,
        exceedances=int(excesses.size),
        return_levels=return_levels,
        diagnostics={
            "log_likelihood": float(
                np.sum(genpareto.logpdf(excesses, c=shape, loc=0, scale=scale))
            ),
            "ks_statistic": float(goodness_of_fit.statistic),
        },
        warnings=warnings,
    )
