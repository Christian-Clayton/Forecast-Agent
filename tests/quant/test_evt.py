import numpy as np
import pytest
from scipy.stats import genpareto

from forecast_agent.quant.evt import evt_pot_fit


def test_pot_fit_recovers_planted_gpd_parameters() -> None:
    shape = 0.25
    scale = 2.0
    rng = np.random.default_rng(42)
    series = genpareto.rvs(
        c=shape,
        loc=0.0,
        scale=scale,
        size=20_000,
        random_state=rng,
    )

    result = evt_pot_fit(series, threshold=0.0, return_periods=[100.0])

    assert result.shape == pytest.approx(shape, abs=0.04)
    assert result.scale == pytest.approx(scale, abs=0.08)
    assert result.return_levels[100.0] > 0.0
    assert result.exceedances == len(series)


def test_pot_fit_rejects_too_few_exceedances() -> None:
    with pytest.raises(ValueError, match="exceedances"):
        evt_pot_fit([0.0, 1.0, 2.0], threshold=0.0, return_periods=[10.0])


def test_pot_fit_rejects_return_period_below_exceedance_rate_limit() -> None:
    series = np.random.default_rng(42).pareto(2.0, size=100)
    with pytest.raises(ValueError, match="return period"):
        evt_pot_fit(series, threshold=0.0, return_periods=[1.0])
