import pytest

from forecast_agent.scoring.calibration import reliability_bins, sharpness


def test_reliability_bins_report_mean_forecast_and_observed_frequency() -> None:
    bins = reliability_bins(
        probabilities=[0.1, 0.2, 0.9],
        outcomes=[False, True, True],
        n_bins=2,
    )

    assert len(bins) == 2
    assert bins[0].mean_probability == pytest.approx(0.15)
    assert bins[0].observed_frequency == pytest.approx(0.5)
    assert bins[0].count == 2
    assert bins[0].lower_bound == 0.0
    assert bins[0].upper_bound == 0.5
    assert bins[1].mean_probability == pytest.approx(0.9)
    assert bins[1].observed_frequency == 1.0
    assert bins[1].lower_bound == 0.5
    assert bins[1].upper_bound == 1.0


def test_reliability_bins_reject_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        reliability_bins([0.5], [True, False], n_bins=2)


def test_sharpness_is_mean_interval_width() -> None:
    assert sharpness([(1.0, 5.0), (2.0, 8.0)]) == pytest.approx(5.0)
