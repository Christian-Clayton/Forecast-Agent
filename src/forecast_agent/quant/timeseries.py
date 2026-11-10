import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from forecast_agent.quant.baselines import baseline_forecast


@dataclass(frozen=True)
class TimeSeriesForecast:
    forecast: list[float]
    lower: list[float]
    upper: list[float]
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class STLDecomposition:
    trend: list[float]
    seasonal: list[float]
    residual: list[float]


@dataclass(frozen=True)
class FoldError:
    train_end: int
    actual: list[float]
    forecast: list[float]
    mae: float
    rmse: float


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldError]
    mae: float
    rmse: float


def _validated_series(series: Sequence[float], horizon: int) -> list[float]:
    values = [float(value) for value in series]
    if len(values) < 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("series must contain at least three finite values")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return values


def fit_ets(
    series: Sequence[float],
    horizon: int,
    seasonal_periods: int | None = None,
) -> TimeSeriesForecast:
    """Fit additive ETS and use fitted residual quantiles for approximate bands."""
    values = _validated_series(series, horizon)
    if seasonal_periods is not None and (
        seasonal_periods < 2 or len(values) < 2 * seasonal_periods
    ):
        raise ValueError("seasonal ETS requires at least two complete seasons")

    model = ExponentialSmoothing(
        values,
        trend="add",
        seasonal="add" if seasonal_periods is not None else None,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    result = model.fit(optimized=True)
    forecast = [float(value) for value in result.forecast(horizon)]
    residuals = np.asarray(values) - np.asarray(result.fittedvalues)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise ValueError("ETS fit produced no finite residuals for intervals")
    lower_error, upper_error = (
        float(value) for value in np.quantile(residuals, [0.05, 0.95])
    )
    return TimeSeriesForecast(
        forecast=forecast,
        lower=[value + lower_error for value in forecast],
        upper=[value + upper_error for value in forecast],
        diagnostics={
            "aic": float(result.aic),
            "bic": float(result.bic),
            "residual_std": float(np.std(residuals, ddof=1))
            if residuals.size > 1
            else 0.0,
        },
    )


def fit_arima(
    series: Sequence[float],
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 1),
) -> TimeSeriesForecast:
    """Fit ARIMA and return its 90 percent prediction interval."""
    values = _validated_series(series, horizon)
    if len(order) != 3 or any(component < 0 for component in order):
        raise ValueError("ARIMA order must contain three nonnegative integers")

    result = ARIMA(values, order=order).fit()
    prediction = result.get_forecast(steps=horizon)
    forecast = [float(value) for value in prediction.predicted_mean]
    intervals = np.asarray(prediction.conf_int(alpha=0.1), dtype=float)
    return TimeSeriesForecast(
        forecast=forecast,
        lower=[float(value) for value in intervals[:, 0]],
        upper=[float(value) for value in intervals[:, 1]],
        diagnostics={
            "aic": float(result.aic),
            "bic": float(result.bic),
            "residual_std": float(np.std(result.resid, ddof=1)),
        },
    )


def stl_decompose(
    series: Sequence[float],
    period: int,
    robust: bool = False,
) -> STLDecomposition:
    """Return additive trend, seasonal, and residual components."""
    values = [float(value) for value in series]
    if period < 2 or len(values) < 2 * period:
        raise ValueError("STL requires at least two periods of length two or more")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("series must contain finite values")

    result = STL(values, period=period, robust=robust).fit()
    return STLDecomposition(
        trend=[float(value) for value in result.trend],
        seasonal=[float(value) for value in result.seasonal],
        residual=[float(value) for value in result.resid],
    )


def walk_forward_score(
    series: Sequence[float],
    model: Literal["naive", "seasonal_naive", "ets", "arima"],
    min_train_size: int,
    horizon: int,
    step: int = 1,
    season_length: int | None = None,
) -> WalkForwardResult:
    values = [float(value) for value in series]
    if len(values) < 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("series must contain at least three finite values")
    if model not in ("naive", "seasonal_naive", "ets", "arima"):
        raise ValueError(f"unsupported walk-forward model: {model}")
    if min_train_size < 2 or horizon < 1 or step < 1:
        raise ValueError("min_train_size, horizon, and step must be positive")
    if model in ("ets", "arima") and min_train_size < 3:
        raise ValueError(f"{model} requires a minimum training size of three")
    if model == "seasonal_naive":
        if season_length is None or min_train_size <= season_length:
            raise ValueError(
                "seasonal_naive requires a season length below the training size"
            )
    elif season_length is not None:
        raise ValueError("season_length applies only to seasonal_naive")

    folds = []
    for train_end in range(min_train_size, len(values) - horizon + 1, step):
        training = values[:train_end]
        actual = values[train_end : train_end + horizon]
        if model in ("naive", "seasonal_naive"):
            prediction = baseline_forecast(
                training,
                horizon=horizon,
                method=model,
                season_length=season_length,
            ).forecast
        elif model == "ets":
            prediction = fit_ets(training, horizon=horizon).forecast
        else:
            prediction = fit_arima(training, horizon=horizon).forecast

        errors = [observed - predicted for observed, predicted in zip(
            actual, prediction, strict=True
        )]
        folds.append(
            FoldError(
                train_end=train_end,
                actual=actual,
                forecast=prediction,
                mae=math.fsum(abs(error) for error in errors) / len(errors),
                rmse=math.sqrt(math.fsum(error**2 for error in errors) / len(errors)),
            )
        )

    if not folds:
        raise ValueError("window specification produces no walk-forward folds")
    all_errors = [
        observed - predicted
        for fold in folds
        for observed, predicted in zip(fold.actual, fold.forecast, strict=True)
    ]
    return WalkForwardResult(
        folds=folds,
        mae=math.fsum(abs(error) for error in all_errors) / len(all_errors),
        rmse=math.sqrt(
            math.fsum(error**2 for error in all_errors) / len(all_errors)
        ),
    )
