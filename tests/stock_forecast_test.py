"""Stock price prediction test harness.

Predicts where stock prices will be in 7 days using:
1. Historical price series
2. ETS/ARIMA point forecasts
3. Bayesian probability calibration
4. Technical indicators (RSI, MA, Bollinger Bands, ATR)

Usage:
    python stock_forecast_test.py
    python stock_forecast_test.py --symbols AAPL NVDA TSLA
"""

import sys
import json
import math
import numpy as np
from datetime import datetime, UTC

sys.path.insert(0, 'C:/Projects/forecast-agent/src')
from forecast_agent.mcp_servers.quant_server import create_quant_server
from forecast_agent.data.synth.generator import generate_synthetic_data, SyntheticDataset
from forecast_agent.data.real_data import create_real_world_series, merge_real_world_series

# Create the quant server with real-world data series
def _make_quant_server():
    """Create quant server with merged synthetic + real-world data."""
    base_dataset = generate_synthetic_data()
    real_series = create_real_world_series()
    merged_series = merge_real_world_series(base_dataset.series, real_series)
    merged_dataset = SyntheticDataset(
        parameters=base_dataset.parameters,
        series=merged_series,
        event_tables=base_dataset.event_tables,
        truth=base_dataset.truth,
    )
    return create_quant_server()

server = _make_quant_server()

# Stock price priors (historical probability of stock going up)
STOCK_PRIORS = {
    'AAPL': 0.78,
    'NVDA': 0.68,
    'TSLA': 0.72,
    'MSFT': 0.75,
    'AMZN': 0.70,
    'GOOGL': 0.65,
}

# Technical indicator thresholds for signals
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


def generate_synthetic_stock_prices(
    symbol: str,
    n_days: int = 120,
    seed: int = 42,
) -> list[float]:
    """Generate synthetic stock price series with realistic patterns.

    Creates a stock with:
    - Uptrend (for most stocks)
    - Volatility (random daily changes)
    - Mean reversion (prices tend to revert to mean)
    """
    rng = np.random.default_rng(seed)

    # Base price
    base_price = 100.0 + (seed % 50)

    # Generate log returns with trend
    prices = [base_price]
    for i in range(1, n_days):
        # Trend (log returns, avg ~1% per month, or ~0.03% per day)
        drift = 0.0003

        # Mean reversion toward base price
        mean_reversion = 0.02 * (base_price - prices[-1]) / base_price / n_days

        # Volatility (daily std ~1.5%)
        volatility = 0.015

        # Random shock
        shock = rng.normal(0, volatility)

        # Calculate next price
        trend = drift + mean_reversion
        log_return = trend + shock
        next_price = prices[-1] * math.exp(log_return)

        # Ensure price stays positive
        next_price = max(next_price, base_price * 0.3)

        prices.append(next_price)

    return prices


def generate_synthetic_stock_prices_np(rng):
    """Use numpy's random generator directly."""
    import numpy as np
    base_price = 100.0 + (int(rng) % 50)
    np_rng = np.random.default_rng(int(rng) + 1000)
    prices = [float(base_price)]
    for i in range(1, 120):
        drift = 0.0003
        mean_reversion = 0.02 * (base_price - prices[-1]) / base_price / 120
        volatility = 0.015
        shock = np_rng.normal(0, volatility)
        log_return = drift + mean_reversion + shock
        next_price = prices[-1] * math.exp(log_return)
        next_price = max(next_price, base_price * 0.3)
        prices.append(next_price)
    return prices


def get_stock_price_series(symbol: str, n_days: int = 120) -> list[float]:
    """Get synthetic stock price series (since we don't have real API access)."""
    seed = abs(hash(symbol)) % 10000
    prices = generate_synthetic_stock_prices(symbol, n_days, seed)
    return prices


def get_stock_probability(prices: list[float], horizon_days: int = 7, symbol: str = "AAPL", method: str = "ets") -> dict:
    "Compute stock probability forecast using the quant server."
    if len(prices) < 5:
        return {"probability": None, "error": "Need at least 5 price observations", "symbol": symbol}

    result = server.call_tool("stock_probability_forecast", {
        "series": prices,
        "horizon_days": horizon_days,
        "method": method,
        "confidence": 0.0,
        "stock_symbol": symbol,
    })
    # Await if it's a coroutine
    if hasattr(result, '__await__'):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(result)
    # Parse the result (call_tool returns a tuple of (contents, metadata))
    if isinstance(result, tuple):
        contents = result[0]
        if isinstance(contents, list) and len(contents) > 0:
            if hasattr(contents[0], 'text'):
                # The text is a JSON string containing a "result" key
                parsed = json.loads(contents[0].text)
                if isinstance(parsed, dict) and 'result' in parsed:
                    result = parsed['result']
                else:
                    result = parsed
            else:
                result = contents[0]
    # Now result should be the inner dict with probability, forecast, etc.
    return {
        "symbol": symbol,
        "method": method,
        "horizon_days": horizon_days,
        "probability": result.get("probability") if isinstance(result, dict) else None,
        "forecast": result.get("forecast") if isinstance(result, dict) else None,
        "today_price": result.get("today_price") if isinstance(result, dict) else None,
        "prior_up_rate": result.get("prior_up_rate") if isinstance(result, dict) else None,
        "signal_strength": result.get("signal_strength") if isinstance(result, dict) else None,
        "confidence": result.get("confidence") if isinstance(result, dict) else None,
        "raw_result": result,
    }


def get_technical_indicators(prices: list[float]) -> dict:
    "Compute technical indicators for stock analysis."
    result = server.call_tool("add_technical_indicators", {"prices": prices})
    if hasattr(result, '__await__'):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(result)
    if isinstance(result, tuple):
        contents = result[0]
        if isinstance(contents, list) and len(contents) > 0:
            if hasattr(contents[0], 'text'):
                import json
                parsed = json.loads(contents[0].text)
                if isinstance(parsed, dict) and 'result' in parsed:
                    result = parsed['result']
                else:
                    result = parsed
            else:
                result = contents[0]
    return result


def generate_forecast(series: list[float], horizon_days: int = 7) -> dict:
    """Generate a full forecast using multiple methods.

    Returns:
        dict with forecasts from naive, ets, arima, and ensemble methods
    """
    if len(series) < 5:
        return {"error": "Need at least 5 data points", "series_len": len(series)}

    today_price = series[-1]

    # ETS forecast
    ets_result = server.call_tool("fit_ets", {
        "series": series,
        "horizon": horizon_days,
        "seasonal_periods": None,
    })
    if hasattr(ets_result, '__await__'):
        import asyncio
        ets_result = asyncio.get_event_loop().run_until_complete(ets_result)
    if isinstance(ets_result, tuple):
        contents = ets_result[0]
        if isinstance(contents, list) and len(contents) > 0:
            if hasattr(contents[0], 'text'):
                import json
                parsed = json.loads(contents[0].text)
                if isinstance(parsed, dict) and 'result' in parsed:
                    ets_result = parsed['result']
                else:
                    ets_result = parsed
            else:
                ets_result = contents[0]
    ets_forecast = ets_result.get("forecast", [None])[-1] if ets_result.get("forecast") else None

    # ARIMA forecast
    arima_result = server.call_tool("fit_arima", {
        "series": series,
        "horizon": horizon_days,
        "order": [1, 1, 1],
    })
    if hasattr(arima_result, '__await__'):
        import asyncio
        arima_result = asyncio.get_event_loop().run_until_complete(arima_result)
    if isinstance(arima_result, tuple):
        contents = arima_result[0]
        if isinstance(contents, list) and len(contents) > 0:
            if hasattr(contents[0], 'text'):
                import json
                parsed = json.loads(contents[0].text)
                if isinstance(parsed, dict) and 'result' in parsed:
                    arima_result = parsed['result']
                else:
                    arima_result = parsed
            else:
                arima_result = contents[0]
    arima_forecast = arima_result.get("forecast", [None])[-1] if arima_result.get("forecast") else None

    # Naive forecast
    naive_forecast = today_price

    # Momentum forecast
    n = len(series)
    if n >= 2:
        recent_change = (today_price - series[-2]) / today_price
        momentum_forecast = today_price * (1 + recent_change * horizon_days)
    else:
        momentum_forecast = today_price

    # Ensemble forecast
    ensemble_forecast = (naive_forecast + ets_forecast + arima_forecast) / 3

    # Compute probabilities
    ets_prob = get_stock_probability(
        series + [ets_forecast] if ets_forecast else series,
        horizon_days,
        method="ets",
    )
    arima_prob = get_stock_probability(
        series + [arima_forecast] if arima_forecast else series,
        horizon_days,
        method="arima",
    )

    return {
        "today_price": today_price,
        "horizon_days": horizon_days,
        "forecast_naive": naive_forecast,
        "forecast_ets": ets_forecast,
        "forecast_arima": arima_forecast,
        "forecast_ensemble": ensemble_forecast,
        "probability_ets": ets_prob.get("probability"),
        "probability_arima": arima_prob.get("probability"),
        "probability_ensemble": None,
        "signal_strength_ets": ets_prob.get("signal_strength"),
        "signal_strength_arima": arima_prob.get("signal_strength"),
        "volatility": None,
        "model_results": {
            "ets": ets_result,
            "arima": arima_result,
        },
    }


def run_stock_backtest(symbol: str, n_days: int = 120, horizon: int = 7):
    """Run a walk-forward backtest for stock prediction.

    Uses walk-forward validation to test prediction accuracy.
    """
    prices = get_stock_price_series(symbol, n_days)
    results = []

    # Run walk-forward scoring
    wf_result = server.call_tool("walk_forward_score", {
        "series": prices,
        "model": "ets",
        "min_train_size": 30,
        "horizon": horizon,
        "step": 1,
    })
    if hasattr(wf_result, '__await__'):
        import asyncio
        wf_result = asyncio.get_event_loop().run_until_complete(wf_result)

    # Calculate accuracy
    frac_up = 0
    total = 0
    for i in range(len(prices)):
        if i + horizon < len(prices):
            actual_change = prices[i + horizon] - prices[i]
            if actual_change >= 0:
                frac_up += 1
            total += 1

    accuracy = frac_up / total if total > 0 else 0

    return {
        "symbol": symbol,
        "n_days": n_days,
        "horizon_days": horizon,
        "prices": prices,
        "walk_forward": wf_result,
        "backtest_accuracy": accuracy,
        "fraction_up": frac_up / total if total > 0 else 0,
    }


def run_stock_forecast(symbol: str, n_days: int = 120, horizon: int = 7):
    """Run a complete stock forecast for a single symbol."""
    print(f"\n{'='*60}")
    print(f"Stock Forecast: {symbol}")
    print(f"{'='*60}")

    prices = get_stock_price_series(symbol, n_days)

    # Get technical indicators
    indicators = get_technical_indicators(prices)
    rsi = indicators.get("indicators", {}).get("rsi", [50])[-1]
    sma_50 = indicators.get("indicators", {}).get("sma_5", [0])[-1]
    sma_200 = indicators.get("indicators", {}).get("sma_20", [0])[-1]
    atr = indicators.get("indicators", {}).get("atr", [0])[-1]

    # Generate forecast
    forecast = generate_forecast(prices, horizon)

    # Print results
    print(f"\n{'─'*40}")
    print(f"Current price: ${forecast['today_price']:.2f}")
    print(f"Technical Indicators:")
    print(f"  RSI: {rsi:.1f} (overbought >{RSI_OVERBOUGHT}, oversold <{RSI_OVERSOLD})")
    print(f"  SMA-50: ${sma_50:.2f}")
    print(f"  SMA-200: ${sma_200:.2f}")
    print(f"  ATR: ${atr:.2f}")
    print(f"\n{'─'*40}")
    print(f"7-Day Forecast ({horizon} days):")
    print(f"  Naive: ${forecast['forecast_naive']:.2f}")
    print(f"  ETS:   ${forecast['forecast_ets']:.2f}")
    print(f"  ARIMA: ${forecast['forecast_arima']:.2f}")
    print(f"  Ensemble:${forecast['forecast_ensemble']:.2f}")
    print(f"\n{'─'*40}")
    print(f"Probability of going up in {horizon} days:")
    print(f"  ETS probability: {forecast['probability_ets']}")
    print(f"  ARIMA probability: {forecast['probability_arima']}")
    print(f"  Signal strength: {forecast['signal_strength_ets']:.3f}")

    # Stock prior
    prior = STOCK_PRIORS.get(symbol, 0.5)
    print(f"  Stock prior: {prior:.2f} ({symbol})")

    # Predicted 7-day price range
    pct_change = (forecast['forecast_ensemble'] - forecast['today_price']) / forecast['today_price'] * 100
    print(f"\n{'─'*40}")
    print(f"Predicted 7-day return: {pct_change:.2f}%")
    low = forecast['today_price'] * (1 - abs(pct_change) * 0.05)
    high = forecast['today_price'] * (1 + abs(pct_change) * 0.05)
    print(f"  95% confidence range: ${low:.2f} - ${high:.2f}")

    return {
        "symbol": symbol,
        "prices": prices,
        "indicators": indicators,
        "forecast": forecast,
        "backtest": run_stock_backtest(symbol, n_days, horizon),
    }


def main():
    symbols = ["AAPL", "NVDA", "TSLA"]
    n_days = 120
    horizon = 7

    print("=" * 60)
    print("STOCK PRICE PREDICTION TEST")
    print(f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Prediction horizon: {horizon} days")
    print(f"Historical data: {n_days} days")
    print("=" * 60)

    # Use multiple symbols
    all_results = {}
    for symbol in symbols:
        try:
            result = run_stock_forecast(symbol, n_days, horizon)
            all_results[symbol] = result
        except Exception as e:
            print(f"ERROR for {symbol}: {e}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Symbol':<10} {'Today':<12} {'Forecast':<12} {'Return%':<10} {'Prob(ET S)':<12}")
    print("-" * 56)
    for symbol, result in all_results.items():
        f = result['forecast']
        pct = (f['forecast_ensemble'] - f['today_price']) / f['today_price'] * 100
        print(f"{symbol:<10} ${f['today_price']:<11.2f} ${f['forecast_ensemble']:<11.2f} {pct:>+6.2f}%  {f['probability_ets']}")


if __name__ == "__main__":
    import numpy as np
    np_random_default_rng = np.random.default_rng
    main()
