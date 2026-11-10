from datetime import UTC, datetime, timedelta

from forecast_agent.data.store import DataStore
from forecast_agent.data.synth.generator import generate_synthetic_data
from forecast_agent.mcp_servers.data_server import create_data_server
from tests.mcp_servers.client_helpers import result_payload, run_client


def test_data_server_serializes_records_and_rejects_leakage() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    as_of = start + timedelta(days=10)
    dataset = generate_synthetic_data(
        seed=3,
        n_groups=4,
        days=80,
        start_date=start,
    )
    server = create_data_server(DataStore(dataset))

    async def call_data_tools(session):
        series_list_result = await session.call_tool("list_series", {})
        series_result = await session.call_tool(
            "get_series",
            {
                "series_id": "cell-00",
                "start": start.isoformat(),
                "end": as_of.isoformat(),
                "as_of": as_of.isoformat(),
            },
        )
        leaking_result = await session.call_tool(
            "get_series",
            {
                "series_id": "cell-00",
                "start": start.isoformat(),
                "end": (as_of + timedelta(days=1)).isoformat(),
                "as_of": as_of.isoformat(),
            },
        )
        event_table_result = await session.call_tool(
            "get_event_table",
            {
                "series_id": "cell-00",
                "start": start.isoformat(),
                "end": as_of.isoformat(),
                "as_of": as_of.isoformat(),
            },
        )
        reference_result = await session.call_tool(
            "get_reference_class",
            {
                "description": "daily telecom incident count",
                "as_of": as_of.isoformat(),
            },
        )
        return (
            series_list_result,
            series_result,
            leaking_result,
            event_table_result,
            reference_result,
        )

    (
        series_list_result,
        series_result,
        leaking_result,
        event_table_result,
        reference_result,
    ) = run_client(server, call_data_tools)
    assert result_payload(series_list_result)
    records = result_payload(series_result)
    event_records = result_payload(event_table_result)
    reference = result_payload(reference_result)

    assert records
    assert all(record["date"] <= as_of.isoformat() for record in records)
    assert event_records
    assert reference["source"]
    assert leaking_result.isError
