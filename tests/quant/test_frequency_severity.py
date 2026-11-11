import math

import pandas as pd
import pytest

from forecast_agent.quant.frequency_severity import (
    FrequencyParameters,
    SeverityParameters,
    aggregate_loss_mc,
    fit_frequency_severity,
)


def test_frequency_and_lognormal_severity_fit_known_parameters() -> None:
    event_table = pd.DataFrame(
        {
            "exposure": [1.0, 2.0, 1.0],
            "count": [0, 2, 1],
            "severity": [math.exp(-1), 1.0, math.exp(1)],
        }
    )

    result = fit_frequency_severity(
        event_table,
        exposure_column="exposure",
        frequency_column="count",
        severity_column="severity",
        frequency_distribution="poisson",
        severity_distribution="lognormal",
    )

    assert result.frequency.rate == pytest.approx(0.75)
    assert result.frequency.dispersion == 0.0
    assert result.severity.shape == pytest.approx(math.sqrt(2 / 3))
    assert result.severity.scale == pytest.approx(1.0)


def test_negative_binomial_fit_detects_overdispersion() -> None:
    event_table = pd.DataFrame(
        {
            "exposure": [1.0, 1.0, 1.0, 1.0],
            "count": [0, 0, 10, 10],
            "severity": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = fit_frequency_severity(
        event_table,
        exposure_column="exposure",
        frequency_column="count",
        severity_column="severity",
        frequency_distribution="negative_binomial",
        severity_distribution="gamma",
    )

    assert result.frequency.rate == pytest.approx(5.0)
    assert result.frequency.dispersion == pytest.approx(0.8)
    assert result.severity.shape > 0
    assert result.severity.scale > 0


def test_aggregate_loss_mc_is_reproducible_from_seed() -> None:
    frequency = FrequencyParameters("poisson", rate=2.0, dispersion=0.0)
    severity = SeverityParameters("gamma", shape=1.0, scale=10.0)

    first = aggregate_loss_mc(frequency, severity, horizon=3, n_sims=1000, seed=17)
    second = aggregate_loss_mc(frequency, severity, horizon=3, n_sims=1000, seed=17)

    assert first == second
    assert first.seed == 17
    assert first.quantiles["q05"] <= first.quantiles["q50"]
    assert first.quantiles["q50"] <= first.quantiles["q95"]


def test_aggregate_loss_mc_rejects_nonpositive_simulation_count() -> None:
    with pytest.raises(ValueError, match="n_sims"):
        aggregate_loss_mc(
            FrequencyParameters("poisson", rate=1.0, dispersion=0.0),
            SeverityParameters("gamma", shape=1.0, scale=1.0),
            horizon=1,
            n_sims=0,
            seed=1,
        )


def test_aggregate_loss_mc_rejects_unknown_severity_distribution_without_events() -> (
    None
):
    with pytest.raises(ValueError, match="severity distribution"):
        aggregate_loss_mc(
            FrequencyParameters("poisson", rate=0.0, dispersion=0.0),
            SeverityParameters("unsupported", shape=1.0, scale=1.0),
            horizon=1,
            n_sims=1,
            seed=1,
        )
