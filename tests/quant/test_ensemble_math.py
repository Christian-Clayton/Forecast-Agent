import pytest

from forecast_agent.quant.ensemble_math import ensemble_aggregate


def test_ensemble_aggregation_matches_symmetric_fixture() -> None:
    result = ensemble_aggregate([0.2, 0.5, 0.8])

    assert result.median == pytest.approx(0.5)
    assert result.trimmed_mean == pytest.approx(0.5)
    assert result.logit_mean == pytest.approx(0.5)
    assert result.spread == pytest.approx(0.6)
    assert result.spread_ratio == pytest.approx(1.2)


def test_ensemble_aggregation_handles_zero_mean_spread_ratio() -> None:
    result = ensemble_aggregate([0.0, 0.0])

    assert result.spread_ratio == 0.0
    assert result.logit_mean == 0.0


def test_ensemble_aggregation_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probabilit"):
        ensemble_aggregate([0.5, 1.1])
