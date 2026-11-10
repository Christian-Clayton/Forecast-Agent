import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticParameters:
    seed: int
    n_groups: int
    days: int
    start_date: datetime
    severity_tail_probability: float
    severity_log_mean: float
    severity_log_sigma: float
    severity_tail_shape: float
    severity_tail_scale: float


@dataclass
class SyntheticDataset:
    parameters: SyntheticParameters
    series: dict[str, pd.DataFrame]
    event_tables: dict[str, pd.DataFrame]
    truth: dict[str, object]


def generate_synthetic_data(
    seed: int = 42,
    n_groups: int = 6,
    days: int = 365,
    start_date: datetime = datetime(2024, 1, 1, tzinfo=UTC),
) -> SyntheticDataset:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if n_groups < 4:
        raise ValueError("at least four groups are required")
    if days < 7 * n_groups:
        raise ValueError("days must provide at least one week per group")
    if start_date.tzinfo is None or start_date.utcoffset() is None:
        raise ValueError("start_date must be timezone-aware")
    start_date = start_date.astimezone(UTC)

    parameters = SyntheticParameters(
        seed=seed,
        n_groups=n_groups,
        days=days,
        start_date=start_date,
        severity_tail_probability=0.02,
        severity_log_mean=4.5,
        severity_log_sigma=0.7,
        severity_tail_shape=2.5,
        severity_tail_scale=5000.0,
    )
    rng = np.random.default_rng(seed)
    history_lengths = np.linspace(days // n_groups, days, n_groups, dtype=int)
    series = {}
    event_tables = {}
    baseline_rates = {}
    domain_labels = {}
    changepoints = {}

    for group_index, history_length in enumerate(history_lengths):
        group_id = f"cell-{group_index:02d}"
        fat_tailed = group_index >= n_groups - 2
        baseline_rate = 0.9 + 0.2 * group_index
        first_change = max(1, int(history_length / 3))
        second_change = max(first_change + 1, int(2 * history_length / 3))
        dates = pd.date_range(start_date, periods=int(history_length), freq="D")
        daily_counts = []
        daily_severities = []
        daily_tail_flags = []

        for day_index in range(int(history_length)):
            change_factor = 1.0
            if day_index >= first_change:
                change_factor *= 1.3
            if day_index >= second_change:
                change_factor *= 0.75
            weekly_factor = 1 + 0.25 * math.sin(2 * math.pi * day_index / 7)
            trend_factor = 1 + 0.0005 * day_index
            expected_count = (
                baseline_rate * change_factor * weekly_factor * trend_factor
            )
            count = int(rng.poisson(expected_count))
            if fat_tailed and rng.random() < 0.01:
                count += int(rng.integers(10, 31))

            tail_flags = rng.random(count) < parameters.severity_tail_probability
            severities = rng.lognormal(
                mean=parameters.severity_log_mean,
                sigma=parameters.severity_log_sigma,
                size=count,
            )
            tail_count = int(np.sum(tail_flags))
            if tail_count:
                severities[tail_flags] = parameters.severity_tail_scale * (
                    1 + rng.pareto(parameters.severity_tail_shape, size=tail_count)
                )
            daily_counts.append(count)
            daily_severities.append([float(value) for value in severities])
            daily_tail_flags.append([bool(value) for value in tail_flags])

        series[group_id] = pd.DataFrame(
            {
                "date": dates,
                "count": daily_counts,
                "exposure": np.ones(int(history_length), dtype=float),
            }
        )
        event_tables[group_id] = pd.DataFrame(
            {
                "date": dates,
                "event_count": daily_counts,
                "exposure": np.ones(int(history_length), dtype=float),
                "severity": daily_severities,
                "tail_flags": daily_tail_flags,
            }
        )
        baseline_rates[group_id] = baseline_rate
        domain_labels[group_id] = "fat" if fat_tailed else "thin"
        changepoints[group_id] = [
            dates[first_change].to_pydatetime(),
            dates[second_change].to_pydatetime(),
        ]

    development_triangle = [
        [100.0, 150.0, 180.0, 200.0],
        [120.0, 180.0, 216.0, None],
        [150.0, 225.0, None, None],
        [170.0, None, None, None],
    ]
    return SyntheticDataset(
        parameters=parameters,
        series=series,
        event_tables=event_tables,
        truth={
            "baseline_rates": baseline_rates,
            "domain_labels": domain_labels,
            "changepoints": changepoints,
            "severity": {
                "tail_probability": parameters.severity_tail_probability,
                "log_mean": parameters.severity_log_mean,
                "log_sigma": parameters.severity_log_sigma,
                "tail_shape": parameters.severity_tail_shape,
                "tail_scale": parameters.severity_tail_scale,
            },
            "development_triangle": development_triangle,
            "development_ultimates": [200.0, 240.0, 300.0, 340.0],
        },
    )
