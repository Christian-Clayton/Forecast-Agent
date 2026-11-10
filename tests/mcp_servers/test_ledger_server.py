import math

import pytest

from forecast_agent.mcp_servers.ledger_server import create_ledger_server
from tests.mcp_servers.client_helpers import result_payload, run_client


def test_ledger_server_writes_reads_resolves_and_scores_without_mutation_tools(
    tmp_path,
) -> None:
    server = create_ledger_server(tmp_path / "ledger.sqlite3")
    question = {
        "id": "question-1",
        "raw_text": "Will the incident occur?",
        "falsifiable_text": "Will an incident occur by 2026-12-31?",
        "type": "binary",
        "resolution_criteria": "Resolve yes if at least one incident occurs.",
        "resolution_source": "synthetic:test",
        "resolves_at": "2026-12-31T00:00:00Z",
        "created_at": "2026-01-01T00:00:00Z",
    }
    forecast = {
        "id": "forecast-1",
        "question_id": "question-1",
        "created_at": "2026-01-01T00:00:00Z",
        "as_of": "2025-12-31T00:00:00Z",
        "domain_class": "thin",
        "classification_reasoning": "Stable event rate.",
        "toolkit": "A",
        "probability": 0.8,
        "base_rate": 0.5,
        "base_rate_source": "unverified_prior",
        "tool_call_ids": ["tool-1"],
    }
    resolution = {
        "question_id": "question-1",
        "resolved_at": "2026-12-31T00:00:00Z",
        "outcome": True,
        "resolver_note": "The synthetic event occurred.",
    }
    quantile_question = dict(
        question,
        id="question-2",
        type="quantile",
        falsifiable_text="What is the incident cost by 2026-12-31?",
    )
    quantile_forecast = {
        "id": "forecast-2",
        "question_id": "question-2",
        "created_at": "2026-01-01T00:00:00Z",
        "as_of": "2025-12-31T00:00:00Z",
        "domain_class": "thin",
        "classification_reasoning": "Stable severity distribution.",
        "toolkit": "A",
        "quantiles": {
            "q05": 0.05,
            "q25": 0.25,
            "q50": 0.5,
            "q75": 0.75,
            "q95": 0.95,
        },
    }
    quantile_resolution = {
        "question_id": "question-2",
        "resolved_at": "2026-12-31T00:00:00Z",
        "outcome": 1.0,
        "resolver_note": "The synthetic observed value was 1.",
    }

    async def use_ledger(session):
        tools = await session.list_tools()
        names = {tool.name for tool in tools.tools}
        await session.call_tool("write_question", {"question": question})
        await session.call_tool("write_forecast", {"forecast": forecast})
        read_result = await session.call_tool(
            "read_forecast", {"forecast_id": "forecast-1"}
        )
        open_before = await session.call_tool("list_open_questions", {})
        await session.call_tool("record_resolution", {"resolution": resolution})
        open_after = await session.call_tool("list_open_questions", {})
        score_result = await session.call_tool(
            "score_forecast", {"forecast_id": "forecast-1"}
        )
        await session.call_tool("write_question", {"question": quantile_question})
        await session.call_tool("write_forecast", {"forecast": quantile_forecast})
        await session.call_tool(
            "record_resolution", {"resolution": quantile_resolution}
        )
        quantile_score_result = await session.call_tool(
            "score_forecast", {"forecast_id": "forecast-2"}
        )
        invalid_forecast = dict(forecast, id="forecast-invalid", probability=1.2)
        invalid_result = await session.call_tool(
            "write_forecast", {"forecast": invalid_forecast}
        )
        return (
            names,
            read_result,
            open_before,
            open_after,
            score_result,
            quantile_score_result,
            invalid_result,
        )

    (
        names,
        read_result,
        open_before,
        open_after,
        score_result,
        quantile_score_result,
        invalid_result,
    ) = run_client(server, use_ledger)
    stored_forecast = result_payload(read_result)
    first_open_list = result_payload(open_before)
    final_open_list = result_payload(open_after)
    score = result_payload(score_result)
    quantile_score = result_payload(quantile_score_result)

    assert not any("update" in name or "delete" in name for name in names)
    assert stored_forecast["base_rate_source"] == "unverified_prior"
    assert len(first_open_list) == 1
    assert final_open_list == []
    assert score["brier"] == pytest.approx(0.04)
    assert score["log_score"] == pytest.approx(-math.log(0.8))
    assert score["baseline_brier"] == pytest.approx(0.25)
    assert score["skill_score"] == pytest.approx(0.84)
    assert quantile_score["pinball"] == pytest.approx(0.144)
    assert invalid_result.isError


def test_ledger_server_has_no_forecast_update_or_delete_tools(tmp_path) -> None:
    server = create_ledger_server(tmp_path / "ledger.sqlite3")

    async def list_names(session):
        result = await session.list_tools()
        return {tool.name for tool in result.tools}

    names = run_client(server, list_names)
    assert "write_forecast" in names
    assert not any("update" in name or "delete" in name for name in names)
