import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from forecast_agent.scoring.rules import (
    brier_score,
    crps_approx,
    log_score,
    pinball_loss,
)


def test_brier_score_matches_hand_computed_case() -> None:
    assert brier_score(0.8, True) == pytest.approx(0.04)


def test_brier_score_is_zero_for_a_perfect_forecast() -> None:
    assert brier_score(1.0, True) == 0.0
    assert brier_score(0.0, False) == 0.0


@given(
    probability=st.floats(min_value=0, max_value=1, allow_nan=False),
    outcome=st.booleans(),
)
def test_brier_score_is_always_in_unit_interval(
    probability: float, outcome: bool
) -> None:
    assert 0.0 <= brier_score(probability, outcome) <= 1.0


def test_brier_score_rejects_probability_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="probability"):
        brier_score(1.01, True)


def test_log_score_matches_negative_log_likelihood() -> None:
    assert log_score(0.25, True) == pytest.approx(-math.log(0.25))


def test_log_score_is_finite_for_zero_probability() -> None:
    assert math.isfinite(log_score(0.0, True))


def test_pinball_loss_matches_hand_computed_cases() -> None:
    assert pinball_loss(actual=10, forecast=8, quantile=0.25) == pytest.approx(0.5)
    assert pinball_loss(actual=8, forecast=10, quantile=0.25) == pytest.approx(1.5)


def test_empirical_crps_matches_two_point_fixture() -> None:
    assert crps_approx([0.0, 2.0], observation=1.0) == pytest.approx(0.5)
