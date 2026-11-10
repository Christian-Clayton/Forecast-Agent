import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from forecast_agent.ledger.ledger import connect
from forecast_agent.models import Forecast, Question, Resolution, Score
from forecast_agent.scoring.rules import brier_score, log_score, pinball_loss
from forecast_agent.scoring.skill_score import skill_score


@contextmanager
def _database(path: str | Path) -> Iterator[sqlite3.Connection]:
    connection = connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _question_from_row(row: tuple[object, ...]) -> Question:
    return Question.model_validate(
        dict(
            zip(
                (
                    "id",
                    "raw_text",
                    "falsifiable_text",
                    "type",
                    "resolution_criteria",
                    "resolution_source",
                    "resolves_at",
                    "created_at",
                ),
                row,
                strict=True,
            )
        )
    )


def _forecast_from_row(row: tuple[object, ...]) -> Forecast:
    (
        forecast_id,
        question_id,
        created_at,
        as_of,
        domain_class,
        classification_reasoning,
        toolkit,
        probability,
        quantiles_json,
        base_rate,
        base_rate_source,
        ensemble_members_json,
        ensemble_spread,
        tool_call_ids_json,
        skill_ids_loaded_json,
        cost_usd,
        latency_s,
        trace_id,
    ) = row
    return Forecast.model_validate(
        {
            "id": forecast_id,
            "question_id": question_id,
            "created_at": created_at,
            "as_of": as_of,
            "domain_class": domain_class,
            "classification_reasoning": classification_reasoning,
            "toolkit": toolkit,
            "probability": probability,
            "quantiles": (
                json.loads(quantiles_json) if quantiles_json is not None else None
            ),
            "base_rate": base_rate,
            "base_rate_source": base_rate_source,
            "ensemble_members": json.loads(ensemble_members_json),
            "ensemble_spread": ensemble_spread,
            "tool_call_ids": json.loads(tool_call_ids_json),
            "skill_ids_loaded": json.loads(skill_ids_loaded_json),
            "cost_usd": cost_usd,
            "latency_s": latency_s,
            "trace_id": trace_id,
        }
    )


def _tool_result(ctx: Context, result: object) -> dict[str, object]:
    return {"tool_call_id": ctx.request_id, "result": result}


def create_ledger_server(database_path: str | Path) -> FastMCP:
    server = FastMCP(
        name="forecast-ledger",
        instructions="Append-only forecast ledger, resolution, and scoring tools.",
        json_response=True,
    )

    @server.tool()
    def write_question(ctx: Context, question: Question) -> dict[str, object]:
        data = question.model_dump(mode="json")
        with _database(database_path) as connection:
            connection.execute(
                """
                INSERT INTO questions (
                    id, raw_text, falsifiable_text, type, resolution_criteria,
                    resolution_source, resolves_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["id"],
                    data["raw_text"],
                    data["falsifiable_text"],
                    data["type"],
                    data["resolution_criteria"],
                    data["resolution_source"],
                    data["resolves_at"],
                    data["created_at"],
                ),
            )
        return _tool_result(ctx, data)

    @server.tool()
    def write_forecast(ctx: Context, forecast: Forecast) -> dict[str, object]:
        data = forecast.model_dump(mode="json")
        with _database(database_path) as connection:
            connection.execute(
                """
                INSERT INTO forecasts (
                    id, question_id, created_at, as_of, domain_class,
                    classification_reasoning, toolkit, probability,
                    quantiles_json, base_rate, base_rate_source,
                    ensemble_members_json, ensemble_spread, tool_call_ids_json,
                    skill_ids_loaded_json, cost_usd, latency_s, trace_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["id"],
                    data["question_id"],
                    data["created_at"],
                    data["as_of"],
                    data["domain_class"],
                    data["classification_reasoning"],
                    data["toolkit"],
                    data["probability"],
                    (
                        json.dumps(data["quantiles"])
                        if data["quantiles"] is not None
                        else None
                    ),
                    data["base_rate"],
                    data["base_rate_source"],
                    json.dumps(data["ensemble_members"]),
                    data["ensemble_spread"],
                    json.dumps(data["tool_call_ids"]),
                    json.dumps(data["skill_ids_loaded"]),
                    data["cost_usd"],
                    data["latency_s"],
                    data["trace_id"],
                ),
            )
        return _tool_result(ctx, data)

    @server.tool()
    def read_forecast(ctx: Context, forecast_id: str) -> dict[str, object]:
        with _database(database_path) as connection:
            row = connection.execute(
                """
                SELECT id, question_id, created_at, as_of, domain_class,
                       classification_reasoning, toolkit, probability,
                       quantiles_json, base_rate, base_rate_source,
                       ensemble_members_json, ensemble_spread, tool_call_ids_json,
                       skill_ids_loaded_json, cost_usd, latency_s, trace_id
                FROM forecasts WHERE id = ?
                """,
                (forecast_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"forecast not found: {forecast_id}")
        return _tool_result(ctx, _forecast_from_row(row).model_dump(mode="json"))

    @server.tool()
    def list_open_questions(ctx: Context) -> dict[str, object]:
        with _database(database_path) as connection:
            rows = connection.execute(
                """
                SELECT q.id, q.raw_text, q.falsifiable_text, q.type,
                       q.resolution_criteria, q.resolution_source, q.resolves_at,
                       q.created_at
                FROM questions AS q
                LEFT JOIN resolutions AS r ON r.question_id = q.id
                WHERE r.question_id IS NULL
                ORDER BY q.resolves_at, q.id
                """
            ).fetchall()
        return _tool_result(
            ctx, [_question_from_row(row).model_dump(mode="json") for row in rows]
        )

    @server.tool()
    def record_resolution(ctx: Context, resolution: Resolution) -> dict[str, object]:
        data = resolution.model_dump(mode="json")
        with _database(database_path) as connection:
            connection.execute(
                """
                INSERT INTO resolutions (
                    question_id, resolved_at, outcome_json, resolver_note
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    data["question_id"],
                    data["resolved_at"],
                    json.dumps(data["outcome"]),
                    data["resolver_note"],
                ),
            )
        return _tool_result(ctx, data)

    @server.tool()
    def score_forecast(ctx: Context, forecast_id: str) -> dict[str, object]:
        with _database(database_path) as connection:
            row = connection.execute(
                """
                SELECT f.probability, f.quantiles_json, f.base_rate,
                       q.type, r.outcome_json
                FROM forecasts AS f
                JOIN questions AS q ON q.id = f.question_id
                LEFT JOIN resolutions AS r ON r.question_id = q.id
                WHERE f.id = ?
                """,
                (forecast_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"forecast not found: {forecast_id}")
            probability, quantiles_json, base_rate, question_type, outcome_json = row
            if outcome_json is None:
                raise ValueError(f"question for forecast {forecast_id} is unresolved")

            outcome = json.loads(outcome_json)
            brier = None
            log_loss = None
            pinball = None
            baseline_brier = None
            skill = None
            if question_type == "binary":
                if not isinstance(outcome, bool):
                    raise ValueError("binary resolution outcome must be true or false")
                if probability is None:
                    raise ValueError("binary question requires a probability forecast")
                brier = brier_score(probability, outcome)
                log_loss = log_score(probability, outcome)
                if base_rate is not None:
                    baseline_brier = brier_score(base_rate, outcome)
                    if baseline_brier > 0:
                        skill = skill_score(brier, baseline_brier)
            elif question_type == "quantile":
                if not isinstance(outcome, (int, float)) or isinstance(outcome, bool):
                    raise ValueError("quantile resolution outcome must be numeric")
                if quantiles_json is None:
                    raise ValueError("quantile question requires quantile forecasts")
                quantiles = json.loads(quantiles_json)
                pinball = (
                    sum(
                        pinball_loss(outcome, quantiles[f"q{level:02d}"], level / 100)
                        for level in (5, 25, 50, 75, 95)
                    )
                    / 5
                )
            else:
                raise ValueError(f"scoring is not implemented for {question_type}")

            score = Score(
                forecast_id=forecast_id,
                brier=brier,
                log_score=log_loss,
                pinball=pinball,
                baseline_brier=baseline_brier,
                skill_score=skill,
            )
            data = score.model_dump(mode="json")
            connection.execute(
                """
                INSERT INTO scores (
                    forecast_id, brier, log_score, pinball, baseline_brier,
                    skill_score
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    data["forecast_id"],
                    data["brier"],
                    data["log_score"],
                    data["pinball"],
                    data["baseline_brier"],
                    data["skill_score"],
                ),
            )
        return _tool_result(ctx, data)

    return server


def main() -> None:
    import os

    database_path = os.environ.get("FORECAST_LEDGER_PATH", "forecast-ledger.sqlite3")
    create_ledger_server(database_path).run()


if __name__ == "__main__":
    main()
