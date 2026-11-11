import sqlite3
from pathlib import Path

import pytest

SCHEMA_PATH = (
    Path(__file__).parents[1] / "src" / "forecast_agent" / "ledger" / "schema.sql"
)


def create_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    connection.execute(
        "INSERT INTO questions (id, raw_text, falsifiable_text, type, "
        "resolution_criteria, resolution_source, resolves_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "question-1",
            "Will it occur?",
            "Will it occur by 2026-01-01?",
            "binary",
            "Resolve yes if it occurs.",
            "fixture",
            "2026-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z",
        ),
    )
    connection.execute(
        "INSERT INTO forecasts (id, question_id, created_at, as_of, domain_class, "
        "classification_reasoning, toolkit) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "forecast-1",
            "question-1",
            "2026-01-01T00:00:00Z",
            "2025-12-31T23:59:59Z",
            "thin",
            "Fixture reasoning.",
            "A",
        ),
    )
    return connection


def test_forecast_update_is_rejected() -> None:
    connection = create_database()
    with pytest.raises(sqlite3.IntegrityError, match="forecasts are append-only"):
        connection.execute(
            "UPDATE forecasts SET as_of = ? WHERE id = ?",
            ("2026-01-01T00:00:00Z", "forecast-1"),
        )


def test_forecast_delete_is_rejected() -> None:
    connection = create_database()
    with pytest.raises(sqlite3.IntegrityError, match="forecasts are append-only"):
        connection.execute(
            "DELETE FROM forecasts WHERE id = ?",
            ("forecast-1",),
        )
