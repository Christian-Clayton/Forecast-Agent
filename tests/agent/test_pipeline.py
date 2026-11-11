import asyncio
import json

from forecast_agent.agent.pipeline import ForecastAgent
from forecast_agent.config import Settings
from forecast_agent.ledger.ledger import connect
from forecast_agent.llm.provider import ChatMessage, Completion


class FakeProvider:
    def __init__(
        self,
        *,
        fat_tailed: bool = False,
        refuse: bool = False,
        use_series: bool = False,
    ) -> None:
        self.fat_tailed = fat_tailed
        self.refuse = refuse
        self.use_series = use_series
        self.calls = 0
        self.model_inputs: list[dict[str, object]] = []

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Completion:
        self.calls += 1
        stage = messages[0].content.splitlines()[0].removeprefix("Stage: ")
        stage_input = json.loads(messages[-1].content)
        if stage == "intake":
            if self.refuse:
                output = {"refusal_reason": "Specify a resolution source and date."}
            else:
                output = {
                    "falsifiable_text": stage_input["raw_question"],
                    "type": "binary",
                    "resolution_criteria": "Resolve from the stated outcome.",
                    "resolution_source": "self-authored",
                    "resolves_at": "2030-01-01T00:00:00Z",
                }
        elif stage == "classify":
            output = {
                "domain_class": "fat" if self.fat_tailed else "thin",
                "reasoning": "Synthetic test classification.",
            }
        elif stage == "base_rate":
            output = {
                "reference_class_description": None,
                "unverified_prior": 0.5,
                "reasoning": "No sourced reference class was supplied.",
            }
        elif stage == "decompose":
            output = {"subquestions": []}
            if self.use_series:
                output.update(
                    {
                        "series_id": "cell-00",
                        "start": "2024-01-01T00:00:00Z",
                        "end": "2024-01-30T00:00:00Z",
                        "candidate_model": "ets",
                    }
                )
        elif stage == "model":
            self.model_inputs.append(stage_input)
            if self.fat_tailed:
                output = {
                    "quantiles": {
                        "q05": 0.05,
                        "q25": 0.25,
                        "q50": 0.5,
                        "q75": 0.75,
                        "q95": 0.95,
                    },
                    "reasoning": "A distribution is more appropriate for this tail.",
                }
            else:
                output = {
                    "probability": 0.6,
                    "reasoning": "A calibrated judgment based on available evidence.",
                }
        elif stage == "finalize":
            tool_call_id = stage_input["tool_call_ids"][-1]
            output = {
                "bottom_line": f"Forecast estimate 0.6 [tool:{tool_call_id}].",
                "working": (
                    f"Source classification: unverified_prior [tool:{tool_call_id}]."
                ),
            }
        else:
            raise AssertionError(f"unexpected stage: {stage}")
        return Completion(
            content=json.dumps(output),
            model=model,
            request_id=f"request-{self.calls}",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )


def _settings(tmp_path) -> Settings:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "forecasting-and-measurement"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "Use calibrated evidence and state uncertainty.",
        encoding="utf-8",
    )
    return Settings(
        database_path=tmp_path / "ledger.sqlite3",
        skills_dir=skills_dir,
        openai_compat_model="openrouter/test-model",
        max_steps=20,
        max_tokens=5000,
        max_cost_usd=0.1,
    )


def test_pipeline_records_ten_self_authored_forecasts_in_ledger(tmp_path) -> None:
    settings = _settings(tmp_path)
    provider = FakeProvider()
    agent = ForecastAgent(provider, settings)

    async def run_questions():
        return [
            await agent.run(
                f"Will event {index} occur by 2030-01-01, "
                "resolved using self-authored evidence?"
            )
            for index in range(10)
        ]

    runs = asyncio.run(run_questions())
    connection = connect(settings.database_path)
    try:
        question_count = connection.execute(
            "SELECT COUNT(*) FROM questions"
        ).fetchone()[0]
        forecast_count = connection.execute(
            "SELECT COUNT(*) FROM forecasts"
        ).fetchone()[0]
    finally:
        connection.close()

    assert question_count == 10
    assert forecast_count == 10
    assert provider.calls == 60
    assert all(run.forecast is not None for run in runs)
    assert all(run.forecast.tool_call_ids for run in runs)
    assert all(run.forecast.base_rate_source == "unverified_prior" for run in runs)
    assert all("[tool:" in run.answer.bottom_line for run in runs)


def test_fat_tailed_forecast_uses_toolkit_b_and_quantiles(tmp_path) -> None:
    agent = ForecastAgent(FakeProvider(fat_tailed=True), _settings(tmp_path))

    run = asyncio.run(
        agent.run(
            "Will a highly uncertain event occur by 2030-01-01, "
            "resolved using self-authored evidence?"
        )
    )

    assert run.forecast is not None
    assert run.forecast.toolkit == "B"
    assert run.forecast.probability is None
    assert run.forecast.quantiles.q95 == 0.95
    assert run.forecast.tool_call_ids


def test_pipeline_refuses_questions_without_resolution_details(tmp_path) -> None:
    provider = FakeProvider(refuse=True)
    agent = ForecastAgent(provider, _settings(tmp_path))

    run = asyncio.run(agent.run("Will some event occur?"))

    assert run.refusal_reason == "Specify a resolution source and date."
    assert run.question is None
    assert run.forecast is None
    assert provider.calls == 1


def test_pipeline_uses_complex_model_only_when_it_beats_baseline(tmp_path) -> None:
    from datetime import UTC, datetime

    provider = FakeProvider(use_series=True)
    agent = ForecastAgent(provider, _settings(tmp_path))

    run = asyncio.run(
        agent.run(
            "Will event 1 occur by 2030-01-01, resolved using self-authored evidence?",
            as_of=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )
    model_input = provider.model_inputs[0]
    selection = model_input["baseline_result"]["selected_model"]
    candidate_selected = model_input["candidate_score"]["candidate_selected"]

    assert run.forecast is not None
    assert (selection == "ets") is candidate_selected
