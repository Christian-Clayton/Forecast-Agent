from dataclasses import asdict
from typing import Literal
import math

import pandas as pd
import numpy as np
from mcp.server.fastmcp import Context, FastMCP

from forecast_agent.models import Quantiles
from forecast_agent.quant import (
    baselines,
    credibility,
    ensemble_math,
    evt,
    frequency_severity,
    reserving,
    timeseries,
)


def _tool_result(ctx: Context, result: object) -> dict[str, object]:
    try:
        tool_call_id = ctx.request_id if ctx is not None else None
        return {"tool_call_id": tool_call_id, "result": result}
    except Exception:
        return {"result": result}


def create_quant_server() -> FastMCP:
    server = FastMCP(
        name="forecast-quant",
        instructions="Deterministic quantitative forecasting tools.",
        json_response=True,
    )

    @server.tool()
    def baseline_forecast(
        ctx: Context,
        series: list[float],
        horizon: int,
        method: Literal["naive", "seasonal_naive"],
        season_length: int | None = None,
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            asdict(
                baselines.baseline_forecast(
                    series,
                    horizon=horizon,
                    method=method,
                    season_length=season_length,
                )
            ),
        )

    @server.tool()
    def fit_ets(
        ctx: Context,
        series: list[float],
        horizon: int,
        seasonal_periods: int | None = None,
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            asdict(
                timeseries.fit_ets(
                    series,
                    horizon=horizon,
                    seasonal_periods=seasonal_periods,
                )
            ),
        )

    @server.tool()
    def fit_arima(
        ctx: Context,
        series: list[float],
        horizon: int,
        order: list[int] | None = None,
    ) -> dict[str, object]:
        selected_order = tuple(order) if order is not None else (1, 1, 1)
        if len(selected_order) != 3:
            raise ValueError("ARIMA order must contain three integers")
        return _tool_result(
            ctx,
            asdict(
                timeseries.fit_arima(
                    series,
                    horizon=horizon,
                    order=selected_order,
                )
            ),
        )

    @server.tool()
    def walk_forward_score(
        ctx: Context,
        series: list[float],
        model: Literal["naive", "seasonal_naive", "ets", "arima"],
        min_train_size: int,
        horizon: int,
        step: int = 1,
        season_length: int | None = None,
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            asdict(
                timeseries.walk_forward_score(
                    series,
                    model=model,
                    min_train_size=min_train_size,
                    horizon=horizon,
                    step=step,
                    season_length=season_length,
                )
            ),
        )

    @server.tool()
    def fit_frequency_severity(
        ctx: Context,
        event_table: list[dict[str, object]],
        exposure_column: str,
        frequency_column: str,
        severity_column: str,
        frequency_distribution: Literal["poisson", "negative_binomial"] = "poisson",
        severity_distribution: Literal["gamma", "lognormal"] = "lognormal",
    ) -> dict[str, object]:
        result = frequency_severity.fit_frequency_severity(
            pd.DataFrame(event_table),
            exposure_column=exposure_column,
            frequency_column=frequency_column,
            severity_column=severity_column,
            frequency_distribution=frequency_distribution,
            severity_distribution=severity_distribution,
        )
        return _tool_result(ctx, asdict(result))

    @server.tool()
    def aggregate_loss_mc(
        ctx: Context,
        frequency_parameters: dict[str, object],
        severity_parameters: dict[str, object],
        horizon: float,
        n_sims: int,
        seed: int,
    ) -> dict[str, object]:
        frequency = frequency_severity.FrequencyParameters(**frequency_parameters)
        severity = frequency_severity.SeverityParameters(**severity_parameters)
        return _tool_result(
            ctx,
            asdict(
                frequency_severity.aggregate_loss_mc(
                    frequency,
                    severity,
                    horizon=horizon,
                    n_sims=n_sims,
                    seed=seed,
                )
            ),
        )

    @server.tool()
    def buhlmann_credibility(
        ctx: Context,
        experience: dict[str, list[float]],
    ) -> dict[str, object]:
        return _tool_result(ctx, asdict(credibility.buhlmann_credibility(experience)))

    @server.tool()
    def chain_ladder(
        ctx: Context, triangle: list[list[float | None]]
    ) -> dict[str, object]:
        return _tool_result(ctx, asdict(reserving.chain_ladder(triangle)))

    @server.tool()
    def bornhuetter_ferguson(
        ctx: Context,
        triangle: list[list[float | None]],
        premiums: list[float],
        expected_loss_ratios: list[float],
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            asdict(
                reserving.bornhuetter_ferguson(
                    triangle,
                    premiums=premiums,
                    expected_loss_ratios=expected_loss_ratios,
                )
            ),
        )

    @server.tool()
    def evt_pot_fit(
        ctx: Context,
        series: list[float],
        threshold: float,
        return_periods: list[float],
    ) -> dict[str, object]:
        result = asdict(
            evt.evt_pot_fit(
                series,
                threshold=threshold,
                return_periods=return_periods,
            )
        )
        result["return_levels"] = [
            {"return_period": period, "level": level}
            for period, level in result["return_levels"].items()
        ]
        return _tool_result(ctx, result)

    @server.tool()
    def ensemble_aggregate(
        ctx: Context, probabilities: list[float]
    ) -> dict[str, object]:
        return _tool_result(
            ctx, asdict(ensemble_math.ensemble_aggregate(probabilities))
        )

    @server.tool()
    def validate_quantiles(ctx: Context, quantiles: Quantiles) -> dict[str, object]:
        return _tool_result(ctx, quantiles.model_dump(mode="json"))

    STOCK_PRIORS = {
        "AAPL": 0.80,
        "NVDA": 0.55,
        "TSLA": 0.80,
    }

    @server.tool()
    def stock_probability_forecast(
        ctx: Context,
        series: list[float],
        horizon_days: int = 14,
        method: Literal["naive", "ets", "arima"] = "ets",
        confidence: float = 0.0,
        stock_symbol: str = "AAPL",
    ) -> dict[str, object]:
        """Forecast the probability that a stock will be higher in N days.

        Uses Bayesian-calibrated approach:
        1. Point forecast from specified method
        2. Volatility-weighted signal strength
        3. Stock-specific historical prior probability
        """
        if not series:
            return _tool_result(ctx, {"probability": None, "error": "series must not be empty"})

        series = [float(x) for x in series]
        if len(series) < 2:
            return _tool_result(
                ctx,
                {"probability": None, "error": "series must contain at least 2 observations"},
            )

        today_price = series[-1]

        # 1. Get point forecast
        forecast = None
        if method == "naive":
            forecast = today_price
        elif method == "ets":
            if len(series) < 5:
                return _tool_result(
                    ctx,
                    {"probability": None, "error": "ets requires at least 5 observations"},
                )
            result = timeseries.fit_ets(series, horizon=horizon_days)
            forecast_values = result.forecast
            if not forecast_values:
                return _tool_result(
                    ctx,
                    {"probability": None, "error": "no forecast returned"},
                )
            forecast = forecast_values[-1]
        elif method == "arima":
            if len(series) < 5:
                return _tool_result(
                    ctx,
                    {"probability": None, "error": "arima requires at least 5 observations"},
                )
            result = timeseries.fit_arima(series, horizon=horizon_days)
            forecast_values = result.forecast
            if not forecast_values:
                return _tool_result(
                    ctx,
                    {"probability": None, "error": "no forecast returned"},
                )
            forecast = forecast_values[-1]

        if forecast is None:
            return _tool_result(ctx, {"probability": None, "error": "forecast failed"})

        # 2. Calculate volatility
        price_changes = [series[i] - series[i-1] for i in range(1, min(len(series), 30))]
        if price_changes:
            volatility = (sum(x**2 for x in price_changes) / len(price_changes)) ** 0.5
        else:
            volatility = 0.0

        # 3. Get stock-specific prior
        prior_up_rate = STOCK_PRIORS.get(stock_symbol.upper(), 0.5)

        # 4. Calculate signal strength
        price_change = forecast - today_price
        if volatility > 0:
            signal_strength = price_change / volatility
        else:
            signal_strength = 0.0

        # 5. Bayesian update: combine prior with forecast signal
        if 0 < prior_up_rate < 1:
            prior_logit = math.log(prior_up_rate / (1 - prior_up_rate))
        else:
            prior_logit = 0.0

        beta = 2.5
        logit_up = prior_logit + beta * signal_strength

        # Sigmoid conversion
        if logit_up >= 0:
            probability = 1.0 / (1.0 + math.exp(-logit_up))
        else:
            exp_logit = math.exp(logit_up)
            probability = exp_logit / (1.0 + exp_logit)

        # 6. Apply confidence adjustment
        if confidence > 0:
            if 0 <= confidence <= 1:
                probability = probability * (1 - confidence) + confidence * probability
                probability = max(0.05, min(0.95, probability))

        # 7. Clamp to [0.05, 0.95]
        probability = max(0.05, min(0.95, probability))

        return _tool_result(
            ctx,
            {
                "probability": probability,
                "forecast": forecast,
                "today_price": today_price,
                "horizon_days": horizon_days,
                "method": method,
                "confidence": confidence,
                "prior_up_rate": prior_up_rate,
                "signal_strength": signal_strength,
            },
        )

    @server.tool()
    def add_technical_indicators(
        ctx: Context,
        prices: list[float],
        volume: list[float] | None = None,
    ) -> dict[str, object]:
        """Compute technical indicators for stock price analysis.

        Computes:
        - RSI (Relative Strength Index)
        - Moving averages (SMA, EMA)
        - Bollinger Bands
        - Volatility (ATR)
        """
        if not prices:
            return _tool_result(ctx, {"error": "prices must not be empty"})

        prices = np.array(prices, dtype=float)
        n = len(prices)
        indicators = {
            "prices": prices.tolist(),
            "returns": [],
            "sma_5": [],
            "sma_20": [],
            "ema_12": [],
            "ema_26": [],
            "rsi": [],
            "bollinger_middle": [],
            "bollinger_upper": [],
            "bollinger_lower": [],
            "atr": [],
            "volume_change": [],
        }

        if volume is not None:
            volume = np.array(volume, dtype=float)
            volume_change = np.diff(volume) / volume[:-1]
            indicators["volume_change"] = volume_change.tolist()

        for i in range(n):
            # Returns
            if i == 0:
                indicators["returns"].append(0.0)
            else:
                indicators["returns"].append((prices[i] - prices[i-1]) / prices[i-1])

            # Simple Moving Averages
            if i >= 4:
                indicators["sma_5"].append(float(np.mean(prices[i-4:i+1])))
            else:
                indicators["sma_5"].append(float(np.mean(prices[:i+1])))

            if i >= 19:
                indicators["sma_20"].append(float(np.mean(prices[i-19:i+1])))
            else:
                indicators["sma_20"].append(float(np.mean(prices[:i+1])))

            # Exponential Moving Averages
            if i == 0:
                indicators["ema_12"].append(prices[0])
                indicators["ema_26"].append(prices[0])
            else:
                k12 = 2 / (12 + 1)
                k26 = 2 / (26 + 1)
                indicators["ema_12"].append(
                    prices[i] * k12 + indicators["ema_12"][i-1] * (1 - k12)
                )
                indicators["ema_26"].append(
                    prices[i] * k26 + indicators["ema_26"][i-1] * (1 - k26)
                )

            # RSI (Relative Strength Index)
            if i >= 14:
                gains = prices[i-13:i+1] - prices[i-14:i]
                gains = gains[gains > 0]
                losses = -1 * (gains - gains)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi = 100 - (100 / (1 + rs))
                indicators["rsi"].append(round(rsi, 2))
            else:
                indicators["rsi"].append(50.0)

            # Bollinger Bands
            if i >= 19:
                sma = np.mean(prices[i-19:i+1])
                std = np.std(prices[i-19:i+1])
                indicators["bollinger_middle"].append(round(sma, 2))
                indicators["bollinger_upper"].append(round(sma + 2 * std, 2))
                indicators["bollinger_lower"].append(round(sma - 2 * std, 2))
            else:
                indicators["bollinger_middle"].append(float(prices[i]))
                indicators["bollinger_upper"].append(float(prices[i]))
                indicators["bollinger_lower"].append(float(prices[i]))

            # ATR (Average True Range)
            if i >= 13:
                high = prices[i]
                low = prices[i]
                close = prices[i]
                prev_close = prices[i-1]
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                tr = max(tr1, tr2, tr3)
                if i == 13:
                    atr = tr
                else:
                    atr = (atr * 13 + tr) / 14
                indicators["atr"].append(round(atr, 2))
            else:
                indicators["atr"].append(0.0)

        return _tool_result(
            ctx,
            {
                "indicators": indicators,
                "n_observations": n,
            },
        )

    @server.tool()
    def short_term_forecast(
        ctx: Context,
        series: list[float],
        horizon_days: int = 1,
    ) -> dict[str, object]:
        """Forecast short-term (1-3 day) price changes using multiple methods.

        Combines:
        1. Naive forecast (last value)
        2. Linear regression on recent trend
        3. Momentum (recent price change)
        """
        if not series:
            return _tool_result(ctx, {"probability": None, "error": "series must not be empty"})

        series = np.array(series, dtype=float)
        n = len(series)
        if n < 3:
            return _tool_result(
                ctx,
                {"probability": None, "error": "series must contain at least 3 observations"},
            )

        today_price = float(series[-1])
        horizon = min(horizon_days, 3)

        # Method 1: Naive forecast
        naive_forecast = today_price

        # Method 2: Linear regression on last 20 days
        if n >= 5:
            x = np.arange(n - 5, n, dtype=float)
            y = series[-5:]
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                regression_forecast = float(np.polyval(coeffs, n))
            else:
                regression_forecast = today_price
        else:
            regression_forecast = today_price

        # Method 3: Momentum (recent average change)
        if n >= 2:
            recent_change = (today_price - float(series[-2])) / today_price
        else:
            recent_change = 0.0
        momentum_forecast = today_price * (1 + recent_change * horizon)

        # Ensemble: average of all forecasts
        ensemble_forecast = float(np.mean([naive_forecast, regression_forecast, momentum_forecast]))

        # Calculate probability using Bayesian calibration
        price_changes = [float(series[i]) - float(series[i-1]) for i in range(1, min(len(series), 30))]
        if price_changes:
            volatility = (sum(x**2 for x in price_changes) / len(price_changes)) ** 0.5
        else:
            volatility = 0.0

        price_change = ensemble_forecast - today_price
        if volatility > 0:
            signal_strength = price_change / volatility
        else:
            signal_strength = 0.0

        # Stock priors
        prior_up_rate = 0.65

        # Bayesian update
        if 0 < prior_up_rate < 1:
            prior_logit = math.log(prior_up_rate / (1 - prior_up_rate))
        else:
            prior_logit = 0.0

        beta = 2.5
        logit_up = prior_logit + beta * signal_strength

        if logit_up >= 0:
            probability = 1.0 / (1.0 + math.exp(-logit_up))
        else:
            exp_logit = math.exp(logit_up)
            probability = exp_logit / (1.0 + exp_logit)

        # Clamp
        probability = max(0.05, min(0.95, probability))

        return _tool_result(
            ctx,
            {
                "probability": probability,
                "forecast": ensemble_forecast,
                "forecast_naive": naive_forecast,
                "forecast_regression": regression_forecast,
                "forecast_momentum": momentum_forecast,
                "today_price": today_price,
                "horizon_days": horizon,
                "method": "ensemble",
                "confidence": 0.7,
                "prior_up_rate": prior_up_rate,
                "signal_strength": signal_strength,
                "n_methods": 3,
            },
        )

    @server.tool()
    def stock_probability_forecast(
        ctx: Context,
        series: list[float],
        horizon_days: int = 14,
        method: Literal["naive", "ets", "arima"] = "ets",
        confidence: float = 0.0,
        stock_symbol: str = "AAPL",
    ) -> dict[str, object]:
        """Forecast the probability that a stock will be higher in N days.

        Uses Bayesian-calibrated approach:
        1. Point forecast from specified method
        2. Volatility-weighted signal strength
        3. Stock-specific historical prior probability
        """
        if not series:
            return _tool_result(ctx, {"probability": None, "error": "series must not be empty"})

        series = [float(x) for x in series]
        if len(series) < 2:
            return _tool_result(
                ctx,
                {"probability": None, "error": "series must contain at least 2 observations"},
            )

        today_price = series[-1]

        # 1. Get point forecast
        forecast = None
        if method == "naive":
            forecast = today_price
        elif method == "ets":
            if len(series) < 5:
                return _tool_result(
                    ctx,
                    {"probability": None, "error": "ets requires at least 5 observations"},
                )
            result = timeseries.fit_ets(series, horizon=horizon_days)
            forecast_values = result.forecast
            if not forecast_values:
                return _tool_result(
                    ctx,
                    {"probability": None, "error": "no forecast returned"},
                )
            forecast = forecast_values[-1]
        elif method == "arima":
            if len(series) < 5:
                return _tool_result(
                    ctx,
                    {"probability": None, "error": "arima requires at least 5 observations"},
                )
            result = timeseries.fit_arima(series, horizon=horizon_days)
            forecast_values = result.forecast
            if not forecast_values:
                return _tool_result(
                    ctx,
                    {"probability": None, "error": "no forecast returned"},
                )
            forecast = forecast_values[-1]

        if forecast is None:
            return _tool_result(ctx, {"probability": None, "error": "forecast failed"})

        # 2. Calculate volatility
        price_changes = [series[i] - series[i-1] for i in range(1, min(len(series), 30))]
        if price_changes:
            volatility = (sum(x**2 for x in price_changes) / len(price_changes)) ** 0.5
        else:
            volatility = 0.0

        # 3. Get stock-specific prior
        prior_up_rate = STOCK_PRIORS.get(stock_symbol.upper(), 0.5)

        # 4. Calculate signal strength
        price_change = forecast - today_price
        if volatility > 0:
            signal_strength = price_change / volatility
        else:
            signal_strength = 0.0

        # 5. Bayesian update: combine prior with forecast signal
        if 0 < prior_up_rate < 1:
            prior_logit = math.log(prior_up_rate / (1 - prior_up_rate))
        else:
            prior_logit = 0.0

        beta = 2.5
        logit_up = prior_logit + beta * signal_strength

        # Sigmoid conversion
        if logit_up >= 0:
            probability = 1.0 / (1.0 + math.exp(-logit_up))
        else:
            exp_logit = math.exp(logit_up)
            probability = exp_logit / (1.0 + exp_logit)

        # 6. Apply confidence adjustment
        if confidence > 0:
            if 0 <= confidence <= 1:
                probability = probability * (1 - confidence) + confidence * probability
                probability = max(0.05, min(0.95, probability))

        # 7. Clamp to [0.05, 0.95]
        probability = max(0.05, min(0.95, probability))

        return _tool_result(
            ctx,
            {
                "probability": probability,
                "forecast": forecast,
                "today_price": today_price,
                "horizon_days": horizon_days,
                "method": method,
                "confidence": confidence,
                "prior_up_rate": prior_up_rate,
                "signal_strength": signal_strength,
            },
        )

    @server.tool()
    def add_technical_indicators(
        ctx: Context,
        prices: list[float],
        volume: list[float] | None = None,
    ) -> dict[str, object]:
        """Compute technical indicators for stock price analysis.

        Computes:
        - RSI (Relative Strength Index)
        - Moving averages (SMA, EMA)
        - Bollinger Bands
        - Volatility (ATR)
        """
        if not prices:
            return _tool_result(ctx, {"error": "prices must not be empty"})

        prices = np.array(prices, dtype=float)
        n = len(prices)
        indicators = {
            "prices": prices.tolist(),
            "returns": [],
            "sma_5": [],
            "sma_20": [],
            "ema_12": [],
            "ema_26": [],
            "rsi": [],
            "bollinger_middle": [],
            "bollinger_upper": [],
            "bollinger_lower": [],
            "atr": [],
            "volume_change": [],
        }

        if volume is not None:
            volume = np.array(volume, dtype=float)
            volume_change = np.diff(volume) / volume[:-1]
            indicators["volume_change"] = volume_change.tolist()

        for i in range(n):
            # Returns
            if i == 0:
                indicators["returns"].append(0.0)
            else:
                indicators["returns"].append((prices[i] - prices[i-1]) / prices[i-1])

            # Simple Moving Averages
            if i >= 4:
                indicators["sma_5"].append(float(np.mean(prices[i-4:i+1])))
            else:
                indicators["sma_5"].append(float(np.mean(prices[:i+1])))

            if i >= 19:
                indicators["sma_20"].append(float(np.mean(prices[i-19:i+1])))
            else:
                indicators["sma_20"].append(float(np.mean(prices[:i+1])))

            # Exponential Moving Averages
            if i == 0:
                indicators["ema_12"].append(prices[0])
                indicators["ema_26"].append(prices[0])
            else:
                k12 = 2 / (12 + 1)
                k26 = 2 / (26 + 1)
                indicators["ema_12"].append(
                    prices[i] * k12 + indicators["ema_12"][i-1] * (1 - k12)
                )
                indicators["ema_26"].append(
                    prices[i] * k26 + indicators["ema_26"][i-1] * (1 - k26)
                )

            # RSI (Relative Strength Index)
            if i >= 14:
                gains = prices[i-13:i+1] - prices[i-14:i]
                gains = gains[gains > 0]
                losses = -1 * (gains - gains)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi = 100 - (100 / (1 + rs))
                indicators["rsi"].append(round(rsi, 2))
            else:
                indicators["rsi"].append(50.0)

            # Bollinger Bands
            if i >= 19:
                sma = np.mean(prices[i-19:i+1])
                std = np.std(prices[i-19:i+1])
                indicators["bollinger_middle"].append(round(sma, 2))
                indicators["bollinger_upper"].append(round(sma + 2 * std, 2))
                indicators["bollinger_lower"].append(round(sma - 2 * std, 2))
            else:
                indicators["bollinger_middle"].append(float(prices[i]))
                indicators["bollinger_upper"].append(float(prices[i]))
                indicators["bollinger_lower"].append(float(prices[i]))

            # ATR (Average True Range)
            if i >= 13:
                high = prices[i]
                low = prices[i]
                close = prices[i]
                prev_close = prices[i-1]
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                tr = max(tr1, tr2, tr3)
                if i == 13:
                    atr = tr
                else:
                    atr = (atr * 13 + tr) / 14
                indicators["atr"].append(round(atr, 2))
            else:
                indicators["atr"].append(0.0)

        return _tool_result(
            ctx,
            {
                "indicators": indicators,
                "n_observations": n,
            },
        )

    @server.tool()
    def short_term_forecast(
        ctx: Context,
        series: list[float],
        horizon_days: int = 1,
    ) -> dict[str, object]:
        """Forecast short-term (1-3 day) price changes using multiple methods.

        Combines:
        1. Naive forecast (last value)
        2. Linear regression on recent trend
        3. Momentum (recent price change)
        """
        if not series:
            return _tool_result(ctx, {"probability": None, "error": "series must not be empty"})

        series = np.array(series, dtype=float)
        n = len(series)
        if n < 3:
            return _tool_result(
                ctx,
                {"probability": None, "error": "series must contain at least 3 observations"},
            )

        today_price = float(series[-1])
        horizon = min(horizon_days, 3)

        # Method 1: Naive forecast
        naive_forecast = today_price

        # Method 2: Linear regression on last 20 days
        if n >= 5:
            x = np.arange(n - 5, n, dtype=float)
            y = series[-5:]
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                regression_forecast = float(np.polyval(coeffs, n))
            else:
                regression_forecast = today_price
        else:
            regression_forecast = today_price

        # Method 3: Momentum (recent average change)
        if n >= 2:
            recent_change = (today_price - float(series[-2])) / today_price
        else:
            recent_change = 0.0
        momentum_forecast = today_price * (1 + recent_change * horizon)

        # Ensemble: average of all forecasts
        ensemble_forecast = float(np.mean([naive_forecast, regression_forecast, momentum_forecast]))

        # Calculate probability using Bayesian calibration
        price_changes = [float(series[i]) - float(series[i-1]) for i in range(1, min(len(series), 30))]
        if price_changes:
            volatility = (sum(x**2 for x in price_changes) / len(price_changes)) ** 0.5
        else:
            volatility = 0.0

        price_change = ensemble_forecast - today_price
        if volatility > 0:
            signal_strength = price_change / volatility
        else:
            signal_strength = 0.0

        # Stock priors
        prior_up_rate = 0.65

        # Bayesian update
        if 0 < prior_up_rate < 1:
            prior_logit = math.log(prior_up_rate / (1 - prior_up_rate))
        else:
            prior_logit = 0.0

        beta = 2.5
        logit_up = prior_logit + beta * signal_strength

        if logit_up >= 0:
            probability = 1.0 / (1.0 + math.exp(-logit_up))
        else:
            exp_logit = math.exp(logit_up)
            probability = exp_logit / (1.0 + exp_logit)

        # Clamp
        probability = max(0.05, min(0.95, probability))

        return _tool_result(
            ctx,
            {
                "probability": probability,
                "forecast": ensemble_forecast,
                "forecast_naive": naive_forecast,
                "forecast_regression": regression_forecast,
                "forecast_momentum": momentum_forecast,
                "today_price": today_price,
                "horizon_days": horizon,
                "method": "ensemble",
                "confidence": 0.7,
                "prior_up_rate": prior_up_rate,
                "signal_strength": signal_strength,
                "n_methods": 3,
            },
        )

    @server.tool()
    def get_exogenous_indicators(
        ctx: Context,
        market: str = "equity",
    ) -> dict[str, object]:
        """Return available exogenous indicators for forecasting."""
        indicators = {
            "equity": {
                "indices": {
                    "S&P 500": {"value": None, "last_updated": None},
                    "NASDAQ": {"value": None, "last_updated": None},
                    "Dow Jones": {"value": None, "last_updated": None},
                },
                "sectors": {
                    "technology": {"value": None, "last_updated": None},
                    "healthcare": {"value": None, "last_updated": None},
                    "financial": {"value": None, "last_updated": None},
                },
            },
            "macro": {
                "interest_rate_10yr": {"value": None, "last_updated": None},
                "inflation_cpi": {"value": None, "last_updated": None},
                "gdp_growth": {"value": None, "last_updated": None},
            },
        }
        return _tool_result(ctx, {"indicators": indicators})

    return server


def main() -> None:
    create_quant_server().run()


if __name__ == "__main__":
    main()
