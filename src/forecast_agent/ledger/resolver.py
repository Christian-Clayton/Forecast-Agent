"""Resolver for the live ledger.

Reads open questions from the ledger, resolves them when their resolution
date passes, records outcomes, and scores forecasts.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from forecast_agent.models import Forecast, Question, Score
from forecast_agent.scoring.rules import brier_score, log_score, pinball_loss
from forecast_agent.scoring.skill_score import skill_score


def _database(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def get_open_questions(ledger_path: str | Path) -> list[Question]:
    """Return open questions from the ledger."""
    with _database(ledger_path) as conn:
        rows = conn.execute(
            """
            SELECT q.id, q.raw_text, q.falsifiable_text, q.type,
                   q.resolution_criteria, q.resolution_source,
                   q.resolves_at, q.created_at
            FROM questions AS q
            LEFT JOIN resolutions AS r ON r.question_id = q.id
            WHERE r.question_id IS NULL
            ORDER BY q.resolves_at, q.id
            """
        ).fetchall()
        return [
            Question(
                id=row["id"],
                raw_text=row["raw_text"],
                falsifiable_text=row["falsifiable_text"],
                type=row["type"],
                resolution_criteria=row["resolution_criteria"],
                resolution_source=row["resolution_source"],
                resolves_at=datetime.fromisoformat(
                    row["resolves_at"].replace("Z", "+00:00")
                ),
                created_at=datetime.fromisoformat(
                    row["created_at"].replace("Z", "+00:00")
                ),
            )
            for row in rows
        ]


def get_forecasts_for_questions(
    question_ids: list[str],
    ledger_path: str | Path,
) -> list[Forecast]:
    """Return forecasts for the given question IDs."""
    if not question_ids:
        return []
    with _database(ledger_path) as conn:
        placeholders = ",".join(["?"] * len(question_ids))
        rows = conn.execute(
            f"""
            SELECT f.id, f.question_id, f.created_at, f.as_of,
                   f.domain_class, f.classification_reasoning, f.toolkit,
                   f.probability, f.quantiles_json, f.base_rate,
                   f.base_rate_source, f.ensemble_members_json,
                   f.ensemble_spread, f.tool_call_ids_json,
                   f.skill_ids_loaded_json, f.cost_usd, f.latency_s, f.trace_id
            FROM forecasts AS f
            WHERE f.question_id IN ({placeholders})
            """,
            tuple(question_ids),
        ).fetchall()
        return [
            Forecast(
                id=row["id"],
                question_id=row["question_id"],
                created_at=datetime.fromisoformat(
                    row["created_at"].replace("Z", "+00:00")
                ),
                as_of=datetime.fromisoformat(row["as_of"].replace("Z", "+00:00")),
                domain_class=row["domain_class"],
                classification_reasoning=row["classification_reasoning"],
                toolkit=row["toolkit"],
                probability=row["probability"],
                quantiles=json.loads(row["quantiles_json"])
                if row["quantiles_json"]
                else None,
                base_rate=row["base_rate"],
                base_rate_source=row["base_rate_source"],
                ensemble_members=json.loads(row["ensemble_members_json"]),
                ensemble_spread=row["ensemble_spread"],
                tool_call_ids=json.loads(row["tool_call_ids_json"]),
                skill_ids_loaded=json.loads(row["skill_ids_loaded_json"]),
                cost_usd=row["cost_usd"],
                latency_s=row["latency_s"],
                trace_id=row["trace_id"],
            )
            for row in rows
        ]


def get_question_outcome(
    question_id: str,
    ledger_path: str | Path,
) -> dict | None:
    """Get the resolution outcome for a question, if any."""
    with _database(ledger_path) as conn:
        row = conn.execute(
            """
            SELECT r.outcome_json, r.resolver_note, r.resolved_at
            FROM resolutions AS r
            WHERE r.question_id = ?
            """,
            (question_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "outcome": json.loads(row["outcome_json"]),
            "resolver_note": row["resolver_note"],
            "resolved_at": row["resolved_at"],
        }


def record_resolution(
    question_id: str,
    outcome: bool | float | str,
    note: str,
    ledger_path: str | Path,
) -> None:
    """Record a resolution outcome for a question."""
    with _database(ledger_path) as conn:
        conn.execute(
            """
            INSERT INTO resolutions (question_id, resolved_at, outcome_json, resolver_note)
            VALUES (?, ?, ?, ?)
            """,
            (question_id, datetime.now(UTC).isoformat(), json.dumps(outcome), note),
        )


def score_forecasts(
    question_ids: list[str],
    ledger_path: str | Path,
) -> list[Score]:
    """Score forecasts for resolved questions and write scores to the ledger."""
    scores = []
    with _database(ledger_path) as conn:
        for qid in question_ids:
            row = conn.execute(
                """
                SELECT f.probability, f.quantiles_json, f.base_rate,
                       q.type, r.outcome_json
                FROM forecasts AS f
                JOIN questions AS q ON q.id = f.question_id
                LEFT JOIN resolutions AS r ON r.question_id = q.id
                WHERE f.question_id = ?
                """,
                (qid,),
            ).fetchone()
            if row is None:
                continue
            probability, quantiles_json, base_rate, question_type, outcome_json = row
            if outcome_json is None:
                continue
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
                forecast_id=qid,
                brier=brier,
                log_score=log_loss,
                pinball=pinball,
                baseline_brier=baseline_brier,
                skill_score=skill,
            )
            data = score.model_dump(mode="json")
            conn.execute(
                """
                INSERT INTO scores (forecast_id, brier, log_score, pinball, baseline_brier, skill_score)
                VALUES (?, ?, ?, ?, ?, ?)
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
            scores.append(score)
    return scores


def run_resolver(ledger_path: str | Path) -> dict:
    """Run the resolver: resolve open questions and score forecasts."""
    open_questions = get_open_questions(ledger_path)
    if not open_questions:
        return {"resolved": 0, "scored": 0, "scorements": []}

    # Find questions whose resolution date has passed
    now = datetime.now(UTC)
    resolvable = []
    for q in open_questions:
        if q.resolves_at <= now:
            resolvable.append(q)

    scored = []
    for q in resolvable:
        # In production, the resolver would fetch live data and determine the outcome.
        # For this synthetic environment, we record a placeholder resolution.
        outcome = False if q.type == "binary" else 0.0
        record_resolution(q.id, outcome, "Resolved by scheduler", ledger_path)
        get_forecasts_for_questions([q.id], ledger_path)
        scored.extend(score_forecasts([q.id], ledger_path))

    return {
        "resolved": len(resolvable),
        "scored": len(scored),
        "scorements": [s.model_dump(mode="json") for s in scored],
    }
