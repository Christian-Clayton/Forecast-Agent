"""Walk-forward backtest eval harness.

Runs a walk-forward backtest on synthetic series and reports error metrics
against naive and seasonal-naive baselines.
"""

from dataclasses import dataclass

from forecast_agent.data.synth.generator import generate_synthetic_data
from forecast_agent.mcp_servers.data_server import create_data_server
from forecast_agent.mcp_servers.quant_server import create_quant_server


@dataclass(frozen=True)
class BacktestResult:
    folds: list[dict]
    mae: float
    rmse: float
    baseline_mae: float
    baseline_rmse: float
    candidate_selected: bool


def run_walk_forward_backtest(
    series_id: str,
    start: str,
    end: str,
    baseline_method: str = "naive",
    candidate_model: str = "ets",
    min_train_size: int = 30,
    horizon: int = 1,
    step: int = 10,
) -> BacktestResult:
    """Run a walk-forward backtest and return the result."""
    dataset = generate_synthetic_data()
    create_data_server(dataset)
    create_quant_server()

    # Simulate MCP client session (in-process).
    # We use the quant server directly for walk-forward scoring.
    from forecast_agent.quant.timeseries import walk_forward_score

    values = dataset.series[series_id]["count"].tolist()
    result = walk_forward_score(
        series=values,
        model=candidate_model,
        min_train_size=min_train_size,
        horizon=horizon,
        step=step,
    )
    baseline_result = walk_forward_score(
        series=values,
        model="naive",
        min_train_size=min_train_size,
        horizon=horizon,
        step=step,
    )
    candidate_selected = result.rmse < baseline_result.rmse
    return BacktestResult(
        folds=result.folds,
        mae=result.mae,
        rmse=result.rmse,
        baseline_mae=baseline_result.mae,
        baseline_rmse=baseline_result.rmse,
        candidate_selected=candidate_selected,
    )


class BacktestRunner:
    """Walk-forward backtest runner for a single series."""

    def __init__(
        self,
        series_id: str,
        start: str,
        end: str,
        baseline_method: str = "naive",
        candidate_model: str = "ets",
        min_train_size: int = 30,
        horizon: int = 1,
        step: int = 10,
    ) -> None:
        self.series_id = series_id
        self.start = start
        self.end = end
        self.baseline_method = baseline_method
        self.candidate_model = candidate_model
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.step = step

    def run(self) -> BacktestResult:
        return run_walk_forward_backtest(
            series_id=self.series_id,
            start=self.start,
            end=self.end,
            baseline_method=self.baseline_method,
            candidate_model=self.candidate_model,
            min_train_size=self.min_train_size,
            horizon=self.horizon,
            step=self.step,
        )
