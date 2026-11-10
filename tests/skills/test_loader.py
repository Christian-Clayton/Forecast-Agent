import pytest

from forecast_agent.skills.loader import SkillLoader


def test_skill_loader_reads_a_requested_skill_and_counts_tokens(tmp_path) -> None:
    skill_dir = tmp_path / "forecasting-and-measurement"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("Make calibrated forecasts from evidence.", encoding="utf-8")
    loader = SkillLoader(tmp_path)

    skill = loader.load("forecasting-and-measurement")

    assert skill.skill_id == "forecasting-and-measurement"
    assert skill.content == "Make calibrated forecasts from evidence."
    assert skill.token_estimate > 0


def test_skill_loader_rejects_path_traversal_and_missing_skills(tmp_path) -> None:
    loader = SkillLoader(tmp_path)

    with pytest.raises(ValueError, match="invalid skill id"):
        loader.load("..")

    with pytest.raises(FileNotFoundError, match="missing-skill"):
        loader.load("missing-skill")
