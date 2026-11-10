from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from forecast_agent.config import Settings
from forecast_agent.models import Forecast, Quantiles, Question


def test_question_keeps_resolution_contract_and_timestamps() -> None:
    resolves_at = datetime(2026, 12, 31, tzinfo=UTC)
    question = Question(
        id="question-1",
        raw_text="Will the event occur?",
        falsifiable_text="Will the event occur by 2026-12-31?",
        type="binary",
        resolution_criteria="Resolve yes if the event occurs by the date.",
        resolution_source="https://example.test/source",
        resolves_at=resolves_at,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert question.resolves_at == resolves_at
    assert question.type == "binary"


def test_forecast_keeps_tool_lineage_and_unverified_prior_label() -> None:
    forecast = Forecast(
        id="forecast-1",
        question_id="question-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        as_of=datetime(2025, 12, 31, tzinfo=UTC),
        domain_class="thin",
        classification_reasoning="Stable event rate with no heavy-tail mechanism.",
        toolkit="A",
        probability=0.6,
        base_rate=0.5,
        base_rate_source="unverified_prior",
        tool_call_ids=["tool-1"],
    )

    assert forecast.probability == 0.6
    assert forecast.base_rate_source == "unverified_prior"
    assert forecast.tool_call_ids == ["tool-1"]


def test_forecast_rejects_probability_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Forecast(
            id="forecast-1",
            question_id="question-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            as_of=datetime(2025, 12, 31, tzinfo=UTC),
            domain_class="thin",
            classification_reasoning="Stable event rate.",
            toolkit="A",
            probability=1.1,
        )


def test_forecast_rejects_nonfinite_probability() -> None:
    with pytest.raises(ValidationError):
        Forecast(
            id="forecast-1",
            question_id="question-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            as_of=datetime(2025, 12, 31, tzinfo=UTC),
            domain_class="thin",
            classification_reasoning="Stable event rate.",
            toolkit="A",
            probability=float("nan"),
        )


def test_quantiles_must_be_in_nondecreasing_order() -> None:
    with pytest.raises(ValidationError):
        Quantiles(q05=10, q25=20, q50=15, q75=30, q95=40)


def test_quantiles_reject_nonfinite_values() -> None:
    with pytest.raises(ValidationError):
        Quantiles(q05=float("nan"), q25=2, q50=3, q75=4, q95=5)


def test_forecast_requires_an_estimate() -> None:
    with pytest.raises(ValidationError):
        Forecast(
            id="forecast-1",
            question_id="question-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            as_of=datetime(2025, 12, 31, tzinfo=UTC),
            domain_class="thin",
            classification_reasoning="Stable event rate.",
            toolkit="A",
        )


def test_forecast_rejects_both_probability_and_quantiles() -> None:
    with pytest.raises(ValidationError):
        Forecast(
            id="forecast-1",
            question_id="question-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            as_of=datetime(2025, 12, 31, tzinfo=UTC),
            domain_class="thin",
            classification_reasoning="Stable event rate.",
            toolkit="A",
            probability=0.5,
            quantiles=Quantiles(q05=1, q25=2, q50=3, q75=4, q95=5),
        )


def test_settings_uses_sqlite_database_path_by_default() -> None:
    assert Settings().database_path.as_posix() == "data/forecast-agent.sqlite3"
