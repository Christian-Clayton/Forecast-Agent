import pytest

from forecast_agent.quant.credibility import buhlmann_credibility


def test_buhlmann_matches_hand_computed_balanced_groups() -> None:
    result = buhlmann_credibility(
        {
            "low": [10.0, 12.0],
            "middle": [20.0, 22.0],
            "high": [30.0, 32.0],
        }
    )

    assert result.epv == pytest.approx(2.0)
    assert result.vhm == pytest.approx(99.0)
    assert result.k == pytest.approx(2.0 / 99.0)
    assert result.groups["low"].credibility == pytest.approx(0.99)
    assert result.groups["low"].blended_estimate == pytest.approx(11.1)
    assert result.groups["middle"].blended_estimate == pytest.approx(21.0)


def test_buhlmann_adjusts_between_variance_for_unequal_history_lengths() -> None:
    result = buhlmann_credibility(
        {
            "short": [10.0, 12.0],
            "medium": [20.0, 22.0, 24.0],
            "long": [30.0, 32.0, 34.0, 36.0],
        }
    )

    assert result.epv == pytest.approx(5.0)
    assert result.vhm == pytest.approx(5960 / 52)
    assert result.groups["short"].credibility < result.groups["long"].credibility
