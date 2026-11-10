import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class BaselineForecast:
    forecast: list[float]
    lower: list[float]
    upper: list[float]


def baseline_forecast(
    series: Sequence[float],
    horizon: int,
    method: Literal["naive", "seasonal_naive"],
    season_length: int | None = None,
) -> BaselineForecast:
    """Forecast from the last value or season with empirical residual intervals."""
    values = [float(value) for value in series]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("series must contain finite values")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    if method == "naive":
        if season_length is not None:
            raise ValueError("season_length applies only to seasonal_naive")
        if len(values) <= horizon:
            raise ValueError(
                "naive intervals require historical errors at the requested horizon"
            )
        forecast = [values[-1]] * horizon
        residuals = np.asarray(values[horizon:]) - np.asarray(values[:-horizon])
    elif method == "seasonal_naive":
        if season_length is None or season_length < 1:
            raise ValueError("season_length must be positive for seasonal_naive")
        if len(values) <= season_length:
            raise ValueError("seasonal_naive requires at least one full prior season")
        last_season = values[-season_length:]
        forecast = [last_season[index % season_length] for index in range(horizon)]
        residuals = np.asarray(values[season_length:]) - np.asarray(
            values[:-season_length]
        )
    else:
        raise ValueError(f"unsupported baseline method: {method}")

    lower_error, upper_error = (
        float(error) for error in np.quantile(residuals, [0.05, 0.95])
    )
    return BaselineForecast(
        forecast=forecast,
        lower=[value + lower_error for value in forecast],
        upper=[value + upper_error for value in forecast],
    )
