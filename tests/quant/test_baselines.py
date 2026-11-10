import pytest

from forecast_agent.quant.baselines import baseline_forecast


def test_naive_baseline_repeats_last_observation() -> None:
    result = baseline_forecast(
        [10.0, 12.0, 11.0, 14.0, 10.0, 13.0],
        horizon=2,
        method="naive",
    )

    assert result.forecast == [13.0, 13.0]
    assert result.lower == pytest.approx([12.0, 12.0])
    assert result.upper == pytest.approx([14.85, 14.85])


def test_naive_baseline_requires_historical_errors_for_requested_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        baseline_forecast([1.0, 2.0], horizon=2, method="naive")


def test_seasonal_naive_repeats_last_season() -> None:
    result = baseline_forecast(
        [1.0, 2.0, 1.0, 2.0],
        horizon=3,
        method="seasonal_naive",
        season_length=2,
    )

    assert result.forecast == [1.0, 2.0, 1.0]
    assert result.lower == [1.0, 2.0, 1.0]
    assert result.upper == [1.0, 2.0, 1.0]


def test_seasonal_naive_requires_season_length() -> None:
    with pytest.raises(ValueError, match="season_length"):
        baseline_forecast([1.0, 2.0], horizon=1, method="seasonal_naive")
