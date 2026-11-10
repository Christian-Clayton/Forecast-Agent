import pytest

from forecast_agent.scoring.skill_score import skill_score


def test_skill_score_measures_improvement_over_baseline() -> None:
    assert skill_score(model_score=0.2, baseline_score=0.5) == pytest.approx(0.6)


def test_skill_score_rejects_zero_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        skill_score(model_score=0.0, baseline_score=0.0)
