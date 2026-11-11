"""Tests for the live ledger resolver."""

import os
import sqlite3
import json
import pytest

from forecast_agent.ledger.resolver import (
    get_open_questions,
    get_forecasts_for_questions,
    record_resolution,
    run_resolver,
)


def _schema_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "forecast_agent",
        "ledger",
        "schema.sql",
    )


def _create_test_ledger(tmp_path, db_name="test.sqlite3"):
    """Create a test ledger with questions and forecasts."""
    db_path = tmp_path / db_name
    conn = sqlite3.connect(db_path)
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    now = "2026-01-01T00:00:00Z"
    # Insert an open question with resolution date in the past
    conn.execute(
        "INSERT INTO questions (id, raw_text, falsifiable_text, type, "
        "resolution_criteria, resolution_source, resolves_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "q1",
            "Test question",
            "Test falsifiable",
            "binary",
            "Resolve from data",
            "test-source",
            "2026-01-01T00:00:00Z",
            now,
        ),
    )
    # Insert a forecast for q1
    conn.execute(
        "INSERT INTO forecasts (id, question_id, created_at, as_of, domain_class, "
        "classification_reasoning, toolkit, probability, base_rate) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("f1", "q1", now, now, "thin", "Test reasoning", "A", 0.6, 0.5),
    )
    conn.commit()
    conn.close()
    return db_path


def test_get_open_questions(tmp_path):
    """Returns questions that have no resolution."""
    db_path = _create_test_ledger(tmp_path)
    open_qs = get_open_questions(db_path)
    assert len(open_qs) == 1
    assert open_qs[0].id == "q1"
    assert open_qs[0].type == "binary"


def test_get_forecasts_for_questions(tmp_path):
    """Return forecasts for given question IDs."""
    db_path = _create_test_ledger(tmp_path)
    forecasts = get_forecasts_for_questions(["q1"], db_path)
    assert len(forecasts) == 1
    assert forecasts[0].id == "f1"
    assert forecasts[0].probability == 0.6


def test_record_resolution(tmp_path):
    """Records a resolution in the ledger."""
    db_path = _create_test_ledger(tmp_path)
    record_resolution("q1", True, "Resolved", db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT outcome_json FROM resolutions WHERE question_id = ?", ("q1",)
    ).fetchone()
    assert row is not None
    assert json.loads(row["outcome_json"]) is True
    conn.close()


def test_run_resolver_resolves_and_scores(tmp_path):
    """Resolver resolves pending questions and scores forecasts."""
    db_path = _create_test_ledger(tmp_path)
    result = run_resolver(db_path)
    assert result["resolved"] == 1
    assert result["scored"] == 1
    # Check resolution was recorded
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT outcome_json FROM resolutions WHERE question_id = ?", ("q1",)
    ).fetchone()
    assert row is not None
    conn.close()
