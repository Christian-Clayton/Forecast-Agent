"""Leakage evaluation harness.

Checks that the as_of gate correctly refuses data requests past the cutoff.
"""

from dataclasses import dataclass

from forecast_agent.data.store import DataStore
from forecast_agent.data.synth.generator import generate_synthetic_data
from forecast_agent.mcp_servers.data_server import create_data_server


@dataclass(frozen=True)
class LeakageEvaluation:
    total_attempts: int
    blocked_attempts: int
    blocked_rate: float


def evaluate_leakage_on_store() -> LeakageEvaluation:
    """Check that the as_of gate blocks all data requests past the cutoff."""
    dataset = generate_synthetic_data()
    store = DataStore(dataset)
    create_data_server(store)
    # Simulate MCP client session for data server tools.
    # We use the store directly for the leakage check.

    total_attempts = 0
    blocked_attempts = 0
    series_id = "cell-00"

    # Attempt to request data past the series end (leakage attempt).
    series_frame = dataset.series[series_id]
    earliest_date = series_frame["date"].min()
    latest_date = series_frame["date"].max()
    as_of = latest_date  # as_of is at the very end, no leakage

    # This is a leakage attempt: request data past the actual data end.
    try:
        from datetime import timedelta
        from forecast_agent.data.store import DataStore as _Store

        future_as_of = latest_date + timedelta(days=10)
        _Store._get_frame(
            dataset.series,
            series_id,
            start=earliest_date,
            end=future_as_of,
            as_of=as_of,
        )
        blocked = False
    except ValueError:
        blocked = True

    total_attempts = 1
    blocked_attempts = 1 if blocked else 0
    return LeakageEvaluation(
        total_attempts=total_attempts,
        blocked_attempts=blocked_attempts,
        blocked_rate=blocked_attempts / total_attempts if total_attempts else 0,
    )


class LeakageEvaluator:
    """Leakage evaluator for the data store."""

    def __init__(self, store_id: str) -> None:
        self.store_id = store_id

    def run(self) -> LeakageEvaluation:
        return evaluate_leakage_on_store()
