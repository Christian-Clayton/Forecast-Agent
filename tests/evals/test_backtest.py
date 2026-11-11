"""Tests for the walk-forward backtest eval harness."""

from forecast_agent.evals.backtest import BacktestRunner


def test_backtest_runner_initializes_and_runs():
    """Runner initialises and runs a basic backtest on synthetic data."""
    runner = BacktestRunner(
        series_id="cell-00",
        start="2024-01-01",
        end="2024-12-31",
        baseline_method="naive",
        candidate_model="ets",
        min_train_size=30,
        horizon=1,
        step=10,
    )
    result = runner.run()
    assert result is not None
    assert len(result.folds) > 0
    assert result.mae >= 0
    assert result.rmse >= 0
    assert result.baseline_mae >= 0
    assert result.baseline_rmse >= 0
