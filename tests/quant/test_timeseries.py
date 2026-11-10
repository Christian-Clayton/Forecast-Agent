import pytest

from forecast_agent.quant.timeseries import (
    fit_arima,
    fit_ets,
    stl_decompose,
    walk_forward_score,
)


def test_ets_returns_forecast_interval_and_diagnostics() -> None:
    result = fit_ets([float(value) for value in range(1, 21)], horizon=3)

    assert len(result.forecast) == 3
    assert len(result.lower) == len(result.upper) == 3
    assert all(lower <= value <= upper for lower, value, upper in zip(
        result.lower, result.forecast, result.upper, strict=True
    ))
    assert "aic" in result.diagnostics
    assert all(value == pytest.approx(expected) for value, expected in zip(
        result.forecast, [21.0, 22.0, 23.0], strict=True
    ))


def test_arima_random_walk_returns_last_value_forecast() -> None:
    result = fit_arima(
        [float(value) for value in range(1, 21)],
        horizon=2,
        order=(0, 1, 0),
    )

    assert result.forecast == pytest.approx([20.0, 20.0])
    assert len(result.lower) == len(result.upper) == 2
    assert all(lower <= value <= upper for lower, value, upper in zip(
        result.lower, result.forecast, result.upper, strict=True
    ))
    assert "aic" in result.diagnostics


def test_walk_forward_reports_known_naive_errors() -> None:
    result = walk_forward_score(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        model="naive",
        min_train_size=3,
        horizon=1,
    )

    assert [fold.mae for fold in result.folds] == pytest.approx([1.0, 1.0, 1.0])
    assert result.mae == pytest.approx(1.0)
    assert result.rmse == pytest.approx(1.0)


def test_walk_forward_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model"):
        walk_forward_score(
            [1.0, 2.0, 3.0, 4.0],
            model="unknown",
            min_train_size=3,
            horizon=1,
        )


def test_stl_decomposition_reconstructs_input_series() -> None:
    series = [
        10 + 0.5 * index + (2 if index % 4 == 0 else -2 if index % 4 == 2 else 0)
        for index in range(40)
    ]

    result = stl_decompose(series, period=4)

    assert len(result.trend) == len(result.seasonal) == len(result.residual) == 40
    assert [
        trend + seasonal + residual
        for trend, seasonal, residual in zip(
            result.trend, result.seasonal, result.residual, strict=True
        )
    ] == pytest.approx(series)
