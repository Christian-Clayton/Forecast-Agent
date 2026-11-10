import pytest

from forecast_agent.quant.reserving import bornhuetter_ferguson, chain_ladder

TRIANGLE = [
    [100.0, 150.0, 180.0],
    [120.0, 180.0, None],
    [150.0, None, None],
]


def test_chain_ladder_matches_standard_cumulative_triangle() -> None:
    result = chain_ladder(TRIANGLE)

    assert result.factors == pytest.approx([1.5, 1.2])
    assert result.ultimates == pytest.approx([180.0, 216.0, 270.0])
    assert result.ibnr == pytest.approx([0.0, 36.0, 120.0])


def test_bornhuetter_ferguson_matches_expected_ultimate_fixture() -> None:
    result = bornhuetter_ferguson(
        TRIANGLE,
        premiums=[1800.0, 2160.0, 2700.0],
        expected_loss_ratios=[0.1, 0.1, 0.1],
    )

    assert result.ibnr == pytest.approx([0.0, 36.0, 120.0])
    assert result.ultimates == pytest.approx([180.0, 216.0, 270.0])
